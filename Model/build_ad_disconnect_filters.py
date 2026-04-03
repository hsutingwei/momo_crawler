#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model/build_ad_disconnect_filters.py
=====================================
基於「Consumer Skepticism / 評價脫鉤」概念的商品過濾條件產生器

產生 2 個 version_tag 並寫入 public.ml_data_filters：
  1. v6_ad_disconnect_prod        - 基礎版：高業配感爆發樣本
  2. v6_ad_disconnect_strict_prod - 強化版：高業配感 + 低品質支撐 + 爆發

兩個版本均只針對 y = 1 的正樣本商品進行過濾，
用來標記「業配感極高，但其成長可能不是由平台內評論自然驅動」的樣本。

Null / 母體規則：
  - 分位數只在「該欄位非 null」的商品母體上計算
  - 若某列必要欄位為 null，該列不納入候選

Usage:
  python Model/build_ad_disconnect_filters.py --date-cutoff 2025-06-25
  python Model/build_ad_disconnect_filters.py --date-cutoff 2025-06-25 \\
      --modes v6_ad_disconnect_prod
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
from psycopg2.extras import execute_values

# --------------------------------------------------------------------------
# 路徑設定（與 build_filters.py 相同）
# --------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseConfig
from Model.data_loader import load_product_level_training_set

ALL_MODES = [
    "v6_ad_disconnect_prod",
    "v6_ad_disconnect_strict_prod",
]

REQUIRED_COLS = ["bert_advertisement_mean", "quality_driven_momentum", "clean_arousal_score"]

# --------------------------------------------------------------------------
# DB helpers
# --------------------------------------------------------------------------

def get_db_connection():
    """Get psycopg2 connection using DatabaseConfig."""
    return DatabaseConfig().get_connection()


def delete_version(conn, version_tag: str):
    """刪除同 version_tag 的舊資料，確保冪等性。"""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM ml_data_filters WHERE version_tag = %s;",
            (version_tag,),
        )
    conn.commit()
    print(f"  🗑️  Cleared old records for version_tag='{version_tag}'")


def insert_filters(conn, version_tag: str, rows: list) -> int:
    """
    Batch-insert rows into ml_data_filters.
    rows: list of (filter_value:str, reason:str, score:float)
    Returns: number of rows inserted.
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
        (version_tag, "product_id", filter_value, reason, score, now)
        for filter_value, reason, score in rows
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
    """
    載入訓練資料並合併 product_id 與 y。
    回傳欄位包含：product_id, y, bert_advertisement_mean,
                  quality_driven_momentum, clean_arousal_score
    """
    print(f"\n{'='*70}")
    print(f"📊 載入資料 (date_cutoff={date_cutoff})")
    print(f"{'='*70}")

    X_dense_df, _X_tfidf, y, meta, _vocab = load_product_level_training_set(
        date_cutoff=date_cutoff,
        label_strategy="absolute",
        label_delta_threshold=10,
        skip_tfidf_matrix=True,
    )

    df = X_dense_df.copy()
    df["product_id"] = meta["product_id"].values
    df["y"] = y.values

    # 確認必要欄位存在
    missing = [c for c in REQUIRED_COLS + ["product_id", "y"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"  ✅ Loaded {len(df)} products  (y=1: {(df['y']==1).sum()}, y=0: {(df['y']==0).sum()})")
    print(f"  Null rates:")
    for col in REQUIRED_COLS:
        null_n = df[col].isna().sum()
        print(f"    {col}: {null_n} nulls ({null_n/len(df)*100:.1f}%)")

    return df


# --------------------------------------------------------------------------
# 2 個過濾版本
# --------------------------------------------------------------------------

def build_ad_disconnect(df: pd.DataFrame, conn) -> tuple:
    """
    v6_ad_disconnect_prod - 基礎版：高業配感爆發樣本

    條件：
      - y = 1
      - bert_advertisement_mean >= P90 (在非 null 母體上計算)
    """
    version_tag = "v6_ad_disconnect_prod"
    print(f"\n{'='*70}")
    print(f"🔨 Building [{version_tag}]")
    print(f"  Rule: y=1 AND bert_advertisement_mean >= P90")

    # 母體：y=1 且 bert_advertisement_mean 非 null
    base = df[df["bert_advertisement_mean"].notna()].copy()

    # 計算分位數門檻
    ad_p90 = base["bert_advertisement_mean"].quantile(0.90)

    # 過濾：限定 y = 1
    mask = (
        (df["y"] == 1)
        & (df["bert_advertisement_mean"].notna())
        & (df["bert_advertisement_mean"] >= ad_p90)
    )
    filtered = df[mask].copy()

    print(f"\n  [Summary]")
    print(f"  ad_p90 = {ad_p90:.4f}")
    print(f"  filtered_count = {len(filtered)}")
    if len(filtered) > 0:
        preview = filtered[["product_id", "bert_advertisement_mean"]].head(5)
        preview = preview.rename(columns={"bert_advertisement_mean": "score"})
        print(f"  Top 5 samples:\n{preview.to_string(index=False)}")

    delete_version(conn, version_tag)
    rows = [
        (str(int(row["product_id"])), "high_advertisement_disconnect",
         float(row["bert_advertisement_mean"]))
        for _, row in filtered.iterrows()
    ]
    inserted = insert_filters(conn, version_tag, rows)
    print(f"  ✅ Inserted {inserted} records")

    return set(filtered["product_id"].astype(int).tolist()), ad_p90


def build_ad_disconnect_strict(df: pd.DataFrame, conn) -> tuple:
    """
    v6_ad_disconnect_strict_prod - 強化版：高業配感 + 低品質支撐 + 爆發

    條件：
      - y = 1
      - bert_advertisement_mean >= P90
      - quality_driven_momentum <= P30
      - clean_arousal_score <= P30
      (所有欄位均非 null 的母體上計算分位數)
    """
    version_tag = "v6_ad_disconnect_strict_prod"
    print(f"\n{'='*70}")
    print(f"🔨 Building [{version_tag}]")
    print(f"  Rule: y=1 AND bert_advertisement_mean>=P90 AND"
          f" quality_driven_momentum<=P30 AND clean_arousal_score<=P30")

    # 母體：三欄均非 null
    base_mask = (
        df["bert_advertisement_mean"].notna()
        & df["quality_driven_momentum"].notna()
        & df["clean_arousal_score"].notna()
    )
    base = df[base_mask].copy()
    print(f"  Quantile population: {len(base)} products (all 3 columns non-null)")

    if len(base) < 20:
        print("  ⚠️  Population too small, skipping.")
        return set(), None, None, None

    # 計算分位數門檻
    ad_p90  = base["bert_advertisement_mean"].quantile(0.90)
    qdm_p30 = base["quality_driven_momentum"].quantile(0.30)
    cas_p30 = base["clean_arousal_score"].quantile(0.30)

    # 過濾：限定 y = 1 + 三條件
    mask = (
        (df["y"] == 1)
        & base_mask
        & (df["bert_advertisement_mean"] >= ad_p90)
        & (df["quality_driven_momentum"] <= qdm_p30)
        & (df["clean_arousal_score"] <= cas_p30)
    )
    filtered = df[mask].copy()

    # score = bert_advertisement_mean - quality_driven_momentum - clean_arousal_score
    filtered["_score"] = (
        filtered["bert_advertisement_mean"]
        - filtered["quality_driven_momentum"]
        - filtered["clean_arousal_score"]
    )

    print(f"\n  [Summary]")
    print(f"  ad_p90  = {ad_p90:.4f}")
    print(f"  qdm_p30 = {qdm_p30:.4f}")
    print(f"  cas_p30 = {cas_p30:.4f}")
    print(f"  filtered_count = {len(filtered)}")
    if len(filtered) > 0:
        preview = filtered[["product_id", "_score"]].head(5)
        preview = preview.rename(columns={"_score": "score"})
        print(f"  Top 5 samples (sorted by score desc):\n"
              f"{filtered.sort_values('_score', ascending=False)[['product_id','_score']].head(5).to_string(index=False)}")

    delete_version(conn, version_tag)
    rows = [
        (str(int(row["product_id"])), "ad_disconnect_low_quality_support",
         float(row["_score"]))
        for _, row in filtered.iterrows()
    ]
    inserted = insert_filters(conn, version_tag, rows)
    print(f"  ✅ Inserted {inserted} records")

    return set(filtered["product_id"].astype(int).tolist()), ad_p90, qdm_p30, cas_p30


# --------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------

def print_overlap(id_basic: set, id_strict: set):
    print(f"\n{'='*70}")
    print("📊 Overlap Summary")
    print(f"{'='*70}")
    overlap = id_basic & id_strict
    print(f"  v6_ad_disconnect       count = {len(id_basic)}")
    print(f"  v6_ad_disconnect_strict count = {len(id_strict)}")
    print(f"  Intersection = {len(overlap)}")
    if overlap:
        print(f"  Sample overlapping product_ids: {list(overlap)[:5]}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Build ad-disconnect consumer skepticism filters for ml_data_filters"
    )
    ap.add_argument(
        "--date-cutoff", type=str, required=True,
        help="Date cutoff for data loading (YYYY-MM-DD)"
    )
    ap.add_argument(
        "--modes", nargs="+", default=ALL_MODES,
        choices=ALL_MODES,
        help="Which filter versions to build (default: all)"
    )
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"\n{'='*70}")
    print(f"🚀 build_ad_disconnect_filters.py")
    print(f"   date_cutoff = {args.date_cutoff}")
    print(f"   modes       = {args.modes}")
    print(f"{'='*70}")

    # 1. Load data
    df = load_data(args.date_cutoff)

    # 2. Connect to DB
    conn = get_db_connection()

    id_basic  = set()
    id_strict = set()

    # 3. Run each mode
    if "v6_ad_disconnect_prod" in args.modes:
        result = build_ad_disconnect(df, conn)
        id_basic, _ = result

    if "v6_ad_disconnect_strict_prod" in args.modes:
        result = build_ad_disconnect_strict(df, conn)
        if len(result) == 4:
            id_strict, _ad_p90, _qdm_p30, _cas_p30 = result
        else:
            id_strict = set()

    conn.close()

    # 4. Overlap
    if len(args.modes) > 1:
        print_overlap(id_basic, id_strict)

    print(f"\n✅ Done!")
    print("""
Verification SQL:
  SELECT version_tag, COUNT(*) AS cnt
  FROM public.ml_data_filters
  WHERE version_tag IN (
    'v6_ad_disconnect_prod',
    'v6_ad_disconnect_strict_prod'
  )
  GROUP BY version_tag ORDER BY version_tag;
""")


if __name__ == "__main__":
    main()
