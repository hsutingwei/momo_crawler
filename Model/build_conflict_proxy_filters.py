#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model/build_conflict_proxy_filters.py
======================================
衝突型 Proxy Filter 產生器

根據 v2_error_prod surrogate analysis 結果，
產生 3 個「高訊號但不一致型」商品過濾版本並寫入 public.ml_data_filters：

  1. v8_conflict_volume_entropy_prod   - 高聲量 + 高語意複雜度
  2. v8_conflict_emotion_mix_prod      - 高正向情緒 + 高負向情緒
  3. v8_conflict_union_prod            - Rule 1 與 Rule 2 的聯集

這三個版本均只針對 y = 1 的正樣本商品，
捕捉「高訊號但訊號方向不一致、模型難以穩定學習」的樣本。

Null 規則：
  - 分位數只在該欄位非 null 的商品母體上計算
  - 若必要欄位為 null，該列不納入候選

Usage:
  python Model/build_conflict_proxy_filters.py --date-cutoff 2025-06-25
  python Model/build_conflict_proxy_filters.py --date-cutoff 2025-06-25 \\
      --modes v8_conflict_volume_entropy_prod v8_conflict_emotion_mix_prod
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
from psycopg2.extras import execute_values

# --------------------------------------------------------------------------
# 路徑設定
# --------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseConfig
from Model.data_loader import load_product_level_training_set

ALL_MODES = [
    "v8_conflict_volume_entropy_prod",
    "v8_conflict_emotion_mix_prod",
    "v8_conflict_union_prod",
]

REQUIRED_COLS_R1 = ["comment_count_90d", "feat_entropy_emb"]
REQUIRED_COLS_R2 = ["clean_arousal_score", "bert_negative_mean"]


# --------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------

def get_db_connection():
    return DatabaseConfig().get_connection()


def delete_version(conn, version_tag: str):
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ml_data_filters WHERE version_tag = %s;",
            (version_tag,),
        )
    conn.commit()
    print(f"  🗑️  Cleared old records for version_tag='{version_tag}'")


def insert_filters(conn, version_tag: str, rows: list) -> int:
    """
    rows: list of (filter_value:str, reason:str, score:float)
    """
    if not rows:
        return 0
    sql = """
    INSERT INTO ml_data_filters
        (version_tag, filter_level, filter_value, reason, score, created_at)
    VALUES %s
    ON CONFLICT (version_tag, filter_level, filter_value) DO NOTHING;
    """
    now = datetime.utcnow()
    values = [
        (version_tag, "product_id", fv, reason, score, now)
        for fv, reason, score in rows
    ]
    with conn.cursor() as cur:
        execute_values(cur, sql, values)
        inserted = cur.rowcount
    conn.commit()
    return inserted


# --------------------------------------------------------------------------
# 資料載入
# --------------------------------------------------------------------------

def load_data(date_cutoff: str) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"📊 載入資料 (date_cutoff={date_cutoff})")
    print(f"{'='*70}")

    X_dense_df, _, y, meta, _ = load_product_level_training_set(
        date_cutoff=date_cutoff,
        label_strategy="absolute",
        label_delta_threshold=10,
        skip_tfidf_matrix=True,
    )
    df = X_dense_df.copy()
    df["product_id"] = meta["product_id"].values
    df["y"] = y.values

    all_needed = ["product_id", "y"] + REQUIRED_COLS_R1 + REQUIRED_COLS_R2
    missing = [c for c in all_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"  ✅ Loaded {len(df)} products  (y=1: {(df['y']==1).sum()}, y=0: {(df['y']==0).sum()})")
    for col in REQUIRED_COLS_R1 + REQUIRED_COLS_R2:
        null_n = df[col].isna().sum()
        print(f"    {col}: {null_n} nulls ({null_n/len(df)*100:.1f}%)")
    return df


# --------------------------------------------------------------------------
# Rule 1: 高聲量 + 高語意複雜度
# --------------------------------------------------------------------------

def build_volume_entropy(df: pd.DataFrame, conn) -> tuple:
    """
    v8_conflict_volume_entropy_prod
    篩選：y=1, comment_count_90d >= P80, feat_entropy_emb >= P80
    分位數母體：兩欄均非 null 的全體商品
    """
    version_tag = "v8_conflict_volume_entropy_prod"
    print(f"\n{'='*70}")
    print(f"🔨 Building [{version_tag}]")

    base = df[df["comment_count_90d"].notna() & df["feat_entropy_emb"].notna()]
    cc90_p80    = base["comment_count_90d"].quantile(0.80)
    entropy_p80 = base["feat_entropy_emb"].quantile(0.80)

    mask = (
        (df["y"] == 1)
        & df["comment_count_90d"].notna()
        & df["feat_entropy_emb"].notna()
        & (df["comment_count_90d"] >= cc90_p80)
        & (df["feat_entropy_emb"] >= entropy_p80)
    )
    filtered = df[mask].copy()
    filtered["_score"] = filtered["comment_count_90d"] + filtered["feat_entropy_emb"]

    print(f"\n  [Summary]")
    print(f"  cc90_p80    = {cc90_p80:.4f}")
    print(f"  entropy_p80 = {entropy_p80:.4f}")
    print(f"  filtered_count = {len(filtered)}")
    if len(filtered) > 0:
        print(filtered[["product_id", "comment_count_90d", "feat_entropy_emb", "_score"]]
              .sort_values("_score", ascending=False).head(5).to_string(index=False))

    delete_version(conn, version_tag)
    rows = [
        (str(int(r["product_id"])), "high_volume_high_entropy_conflict", float(r["_score"]))
        for _, r in filtered.iterrows()
    ]
    inserted = insert_filters(conn, version_tag, rows)
    print(f"  ✅ Inserted {inserted} records")

    return set(filtered["product_id"].astype(int).tolist()), filtered.set_index("product_id")["_score"]


# --------------------------------------------------------------------------
# Rule 2: 高正向情緒 + 高負向情緒
# --------------------------------------------------------------------------

def build_emotion_mix(df: pd.DataFrame, conn) -> tuple:
    """
    v8_conflict_emotion_mix_prod
    篩選：y=1, clean_arousal_score >= P80, bert_negative_mean >= P80
    分位數母體：兩欄均非 null 的全體商品
    """
    version_tag = "v8_conflict_emotion_mix_prod"
    print(f"\n{'='*70}")
    print(f"🔨 Building [{version_tag}]")

    base = df[df["clean_arousal_score"].notna() & df["bert_negative_mean"].notna()]
    arousal_p80 = base["clean_arousal_score"].quantile(0.80)
    neg_p80     = base["bert_negative_mean"].quantile(0.80)

    mask = (
        (df["y"] == 1)
        & df["clean_arousal_score"].notna()
        & df["bert_negative_mean"].notna()
        & (df["clean_arousal_score"] >= arousal_p80)
        & (df["bert_negative_mean"] >= neg_p80)
    )
    filtered = df[mask].copy()
    filtered["_score"] = filtered["clean_arousal_score"] + filtered["bert_negative_mean"]

    print(f"\n  [Summary]")
    print(f"  arousal_p80 = {arousal_p80:.4f}")
    print(f"  neg_p80     = {neg_p80:.4f}")
    print(f"  filtered_count = {len(filtered)}")
    if len(filtered) > 0:
        print(filtered[["product_id", "clean_arousal_score", "bert_negative_mean", "_score"]]
              .sort_values("_score", ascending=False).head(5).to_string(index=False))

    delete_version(conn, version_tag)
    rows = [
        (str(int(r["product_id"])), "high_arousal_high_negative_conflict", float(r["_score"]))
        for _, r in filtered.iterrows()
    ]
    inserted = insert_filters(conn, version_tag, rows)
    print(f"  ✅ Inserted {inserted} records")

    return set(filtered["product_id"].astype(int).tolist()), filtered.set_index("product_id")["_score"]


# --------------------------------------------------------------------------
# Rule 3: Union of Rule 1 and Rule 2
# --------------------------------------------------------------------------

def build_union(df: pd.DataFrame, conn,
                id_r1: set, score_r1: pd.Series,
                id_r2: set, score_r2: pd.Series):
    """
    v8_conflict_union_prod
    聯集（OR）：符合 Rule 1 或 Rule 2 的任一條件即納入。
    Score: R1-only = R1 score; R2-only = R2 score; both = sum
    """
    version_tag = "v8_conflict_union_prod"
    print(f"\n{'='*70}")
    print(f"🔨 Building [{version_tag}]  (Union of Rule 1 ∪ Rule 2)")

    only_r1 = id_r1 - id_r2
    only_r2 = id_r2 - id_r1
    both    = id_r1 & id_r2
    union   = id_r1 | id_r2

    print(f"\n  [Summary]")
    print(f"  union_count              = {len(union)}")
    print(f"  matched_rule1_only_count = {len(only_r1)}")
    print(f"  matched_rule2_only_count = {len(only_r2)}")
    print(f"  matched_both_count       = {len(both)}")

    # Build score series for union
    rows = []
    for pid in union:
        s1 = score_r1.get(pid, 0.0)
        s2 = score_r2.get(pid, 0.0)
        if pid in both:
            score_val = float(s1) + float(s2)
        elif pid in id_r1:
            score_val = float(s1)
        else:
            score_val = float(s2)
        rows.append((str(int(pid)), "conflict_union_rule", score_val))

    delete_version(conn, version_tag)
    inserted = insert_filters(conn, version_tag, rows)
    print(f"  ✅ Inserted {inserted} records")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Build conflict-type proxy filters from surrogate analysis"
    )
    ap.add_argument("--date-cutoff", type=str, required=True)
    ap.add_argument(
        "--modes", nargs="+", default=ALL_MODES,
        choices=ALL_MODES,
        help="Which filter versions to build (default: all)"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*70}")
    print(f"🚀 build_conflict_proxy_filters.py")
    print(f"   date_cutoff = {args.date_cutoff}")
    print(f"   modes       = {args.modes}")
    print(f"{'='*70}")

    # 1. Load data
    df = load_data(args.date_cutoff)

    # 2. DB connection
    conn = get_db_connection()

    id_r1    = set()
    score_r1 = pd.Series(dtype=float)
    id_r2    = set()
    score_r2 = pd.Series(dtype=float)

    # 3. Rule 1
    if "v8_conflict_volume_entropy_prod" in args.modes:
        id_r1, score_r1 = build_volume_entropy(df, conn)

    # 4. Rule 2
    if "v8_conflict_emotion_mix_prod" in args.modes:
        id_r2, score_r2 = build_emotion_mix(df, conn)

    # 5. Rule 3 (union) – requires both R1 and R2 to have been computed
    if "v8_conflict_union_prod" in args.modes:
        # If only union was requested alone, compute R1 & R2 masks without writing
        if not id_r1 and not id_r2:
            print("\n  ℹ️  Computing R1/R2 masks for union (without writing individually)...")
            base1 = df[df["comment_count_90d"].notna() & df["feat_entropy_emb"].notna()]
            cc90_p80    = base1["comment_count_90d"].quantile(0.80)
            entropy_p80 = base1["feat_entropy_emb"].quantile(0.80)
            mask1 = (
                (df["y"] == 1)
                & df["comment_count_90d"].notna() & df["feat_entropy_emb"].notna()
                & (df["comment_count_90d"] >= cc90_p80)
                & (df["feat_entropy_emb"] >= entropy_p80)
            )
            f1 = df[mask1].copy()
            f1["_score"] = f1["comment_count_90d"] + f1["feat_entropy_emb"]
            id_r1 = set(f1["product_id"].astype(int).tolist())
            score_r1 = f1.set_index("product_id")["_score"]

            base2 = df[df["clean_arousal_score"].notna() & df["bert_negative_mean"].notna()]
            arousal_p80 = base2["clean_arousal_score"].quantile(0.80)
            neg_p80     = base2["bert_negative_mean"].quantile(0.80)
            mask2 = (
                (df["y"] == 1)
                & df["clean_arousal_score"].notna() & df["bert_negative_mean"].notna()
                & (df["clean_arousal_score"] >= arousal_p80)
                & (df["bert_negative_mean"] >= neg_p80)
            )
            f2 = df[mask2].copy()
            f2["_score"] = f2["clean_arousal_score"] + f2["bert_negative_mean"]
            id_r2 = set(f2["product_id"].astype(int).tolist())
            score_r2 = f2.set_index("product_id")["_score"]

        build_union(df, conn, id_r1, score_r1, id_r2, score_r2)

    conn.close()

    print(f"\n✅ Done!")
    print("""
Verification SQL:
  SELECT version_tag, COUNT(*) AS cnt
  FROM public.ml_data_filters
  WHERE version_tag IN (
    'v8_conflict_volume_entropy_prod',
    'v8_conflict_emotion_mix_prod',
    'v8_conflict_union_prod'
  )
  GROUP BY version_tag ORDER BY version_tag;
""")


if __name__ == "__main__":
    main()
