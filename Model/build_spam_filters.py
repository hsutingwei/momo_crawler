#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model/build_spam_filters.py
===========================
基於「評論異常訊號」的商品過濾條件產生器

產生 3 個 version_tag 並寫入 public.ml_data_filters：
  1. v5_spam_fixed_prod      - 固定閾值版
  2. v5_spam_quantile_prod   - 分位數版
  3. v5_spam_score_top5_prod - Spam Score Top-5% 版

Null / 極端值處理規則（確保結果可重現）：
  - feat_entropy_emb  null → 補 1.0（最不可疑，不進 filter）
  - feat_ncd_spam     null → 補 1.0（最不可疑，不進 filter）
  - kin_acc_abs       null → 補 0.0；負值裁切為 0 後再 normalize
  - max(kin_acc_abs) = 0  → normalized_kin_acc_abs = 0

Quantile 計算母體：
  三個關鍵欄位均非空且 comment_count_90d > 5 的全體商品。

Usage:
  python Model/build_spam_filters.py --date-cutoff 2025-06-25
  python Model/build_spam_filters.py --date-cutoff 2025-06-25 \\
      --modes v5_spam_fixed_prod v5_spam_quantile_prod
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from psycopg2.extras import execute_values

# --------------------------------------------------------------------------
# 路徑設定（與 build_filters.py 相同）
# --------------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseConfig
from Model.data_loader import load_product_level_training_set

ALL_MODES = [
    "v5_spam_fixed_prod",
    "v5_spam_quantile_prod",
    "v5_spam_score_top5_prod",
]

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


def insert_filters(conn, version_tag: str, rows: list[tuple]) -> int:
    """
    Batch-insert rows into ml_data_filters.

    rows: list of (filter_value, reason, score)  <- product_id as str
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
# 資料載入與 feature 準備
# --------------------------------------------------------------------------

def load_data(date_cutoff: str):
    """
    用 data_loader 載入含 spam feature 的商品 DataFrame。
    回傳欄位：product_id, feat_entropy_emb, feat_ncd_spam,
              kin_acc_abs, comment_count_90d
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

    # 合併 product_id (from meta)
    df = X_dense_df.copy()
    df["product_id"] = meta["product_id"].values
    df["y"] = y.values

    needed = ["product_id", "feat_entropy_emb", "feat_ncd_spam",
              "kin_acc_abs", "comment_count_90d"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in loaded data: {missing}")

    print(f"  ✅ Loaded {len(df)} products")
    print(f"  Null rates:")
    for col in ["feat_entropy_emb", "feat_ncd_spam", "kin_acc_abs", "comment_count_90d"]:
        null_n = df[col].isna().sum()
        print(f"    {col}: {null_n} nulls ({null_n/len(df)*100:.1f}%)")

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply null / extreme-value handling as per implementation plan.
    Adds cleaned columns:
        entropy_c    (feat_entropy_emb, null→1.0)
        ncd_c        (feat_ncd_spam,    null→1.0)
        kin_c        (kin_acc_abs,       null→0, clip ≥0)
        kin_norm     (kin_c / max(kin_c), or 0 if max==0)
        spam_score   composite score (higher = more suspicious)
    """
    df = df.copy()

    # --- null / clip ---
    df["entropy_c"] = df["feat_entropy_emb"].fillna(1.0)
    df["ncd_c"]     = df["feat_ncd_spam"].fillna(1.0)
    df["kin_c"]     = df["kin_acc_abs"].fillna(0.0).clip(lower=0.0)

    # --- normalize kin ---
    max_kin = df["kin_c"].max()
    df["kin_norm"] = df["kin_c"] / max_kin if max_kin > 0 else 0.0

    # --- spam_score: higher = more suspicious ---
    # entropy low → (1-entropy) high → spam-like
    # ncd    low → (1-ncd)    high → spam-like
    # acc    high → normalized high → spam-like
    df["spam_score"] = (
        (1.0 - df["entropy_c"]) * 0.4
        + (1.0 - df["ncd_c"]) * 0.4
        + df["kin_norm"] * 0.2
    )

    return df


# --------------------------------------------------------------------------
# 3 個過濾版本
# --------------------------------------------------------------------------

def build_fixed(df: pd.DataFrame, conn) -> int:
    """v5_spam_fixed_prod - 固定閾值版"""
    version_tag = "v5_spam_fixed_prod"
    print(f"\n{'='*70}")
    print(f"🔨 Building [{version_tag}]")
    print(f"  Rule: feat_entropy_emb < 0.5 AND feat_ncd_spam < 0.4"
          f" AND comment_count_90d > 5")

    mask = (
        (df["feat_entropy_emb"].fillna(1.0) < 0.5)
        & (df["feat_ncd_spam"].fillna(1.0) < 0.4)
        & (df["comment_count_90d"].fillna(0) > 5)
    )
    filtered = df[mask].copy()

    print(f"\n  [Summary]")
    print(f"  threshold: entropy < 0.5, ncd < 0.4, comment_count_90d > 5")
    print(f"  filtered_count = {len(filtered)}")

    delete_version(conn, version_tag)
    rows = [
        (str(int(row["product_id"])), "spam_fixed_threshold",
         float(row["kin_c"]))
        for _, row in filtered.iterrows()
    ]
    inserted = insert_filters(conn, version_tag, rows)
    print(f"  ✅ Inserted {inserted} records")
    return len(filtered)


def build_quantile(df: pd.DataFrame, conn) -> int:
    """v5_spam_quantile_prod - 分位數版"""
    version_tag = "v5_spam_quantile_prod"
    print(f"\n{'='*70}")
    print(f"🔨 Building [{version_tag}]")

    # Quantile 計算母體：三欄均非空 AND comment_count_90d > 5
    base_mask = (
        df["feat_entropy_emb"].notna()
        & df["feat_ncd_spam"].notna()
        & df["kin_acc_abs"].notna()
        & (df["comment_count_90d"].fillna(0) > 5)
    )
    base = df[base_mask]
    print(f"  Quantile population: {len(base)} products"
          f" (non-null + comment_count_90d > 5)")

    if len(base) < 20:
        print("  ⚠️  Population too small, skipping.")
        return 0, None, None, None

    # 計算分位數門檻
    entropy_p10 = base["feat_entropy_emb"].quantile(0.10)
    ncd_p10     = base["feat_ncd_spam"].quantile(0.10)
    acc_p80     = base["kin_acc_abs"].quantile(0.80)

    print(f"\n  [Summary]")
    print(f"  entropy_p10 = {entropy_p10:.4f}")
    print(f"  ncd_p10     = {ncd_p10:.4f}")
    print(f"  acc_p80     = {acc_p80:.4f}")

    mask = (
        (df["feat_entropy_emb"].fillna(1.0) <= entropy_p10)
        & (df["feat_ncd_spam"].fillna(1.0) <= ncd_p10)
        & (df["kin_c"] >= acc_p80)
        & (df["comment_count_90d"].fillna(0) > 5)
    )
    filtered = df[mask].copy()
    print(f"  filtered_count = {len(filtered)}")

    delete_version(conn, version_tag)
    rows = [
        (str(int(row["product_id"])), "spam_quantile_threshold",
         float(row["kin_c"]))
        for _, row in filtered.iterrows()
    ]
    inserted = insert_filters(conn, version_tag, rows)
    print(f"  ✅ Inserted {inserted} records")

    return len(filtered), entropy_p10, ncd_p10, acc_p80


def build_score(df: pd.DataFrame, conn) -> int:
    """v5_spam_score_top5_prod - Spam Score Top-5% 版"""
    version_tag = "v5_spam_score_top5_prod"
    print(f"\n{'='*70}")
    print(f"🔨 Building [{version_tag}]")
    print(f"  spam_score = (1-entropy)*0.4 + (1-ncd)*0.4 + kin_norm*0.2")
    print(f"  Higher score = more suspicious. Taking top 5% (P95+).")

    # P95 門檻
    p95 = df["spam_score"].quantile(0.95)

    mask = df["spam_score"] >= p95
    filtered = df[mask].copy()

    print(f"\n  [Summary]")
    print(f"  spam_score_p95 = {p95:.4f}")
    print(f"  filtered_count = {len(filtered)}")

    delete_version(conn, version_tag)
    rows = [
        (str(int(row["product_id"])), "spam_score_top5",
         float(row["spam_score"]))
        for _, row in filtered.iterrows()
    ]
    inserted = insert_filters(conn, version_tag, rows)
    print(f"  ✅ Inserted {inserted} records")
    return len(filtered), p95


# --------------------------------------------------------------------------
# Overlap summary
# --------------------------------------------------------------------------

def print_overlap(id_fixed: set, id_quantile: set, id_score: set):
    print(f"\n{'='*70}")
    print("📊 Overlap Summary")
    print(f"{'='*70}")
    overlap_fq  = id_fixed & id_quantile
    overlap_fs  = id_fixed & id_score
    overlap_qs  = id_quantile & id_score
    overlap_all = id_fixed & id_quantile & id_score
    print(f"  fixed ∩ quantile = {len(overlap_fq)}")
    print(f"  fixed ∩ score    = {len(overlap_fs)}")
    print(f"  quantile ∩ score = {len(overlap_qs)}")
    print(f"  All 3 intersection = {len(overlap_all)}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Build spam-signal-based product filters for ml_data_filters"
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
    print(f"🚀 build_spam_filters.py")
    print(f"   date_cutoff = {args.date_cutoff}")
    print(f"   modes       = {args.modes}")
    print(f"{'='*70}")

    # 1. Load data
    df_raw = load_data(args.date_cutoff)

    # 2. Prepare cleaned features
    df = prepare_features(df_raw)

    # 3. Connect to DB
    conn = get_db_connection()

    id_fixed    = set()
    id_quantile = set()
    id_score    = set()

    # 4. Run each mode
    if "v5_spam_fixed_prod" in args.modes:
        n = build_fixed(df, conn)
        mask = (
            (df["feat_entropy_emb"].fillna(1.0) < 0.5)
            & (df["feat_ncd_spam"].fillna(1.0) < 0.4)
            & (df["comment_count_90d"].fillna(0) > 5)
        )
        id_fixed = set(df.loc[mask, "product_id"].astype(int).tolist())

    if "v5_spam_quantile_prod" in args.modes:
        n, entropy_p10, ncd_p10, acc_p80 = build_quantile(df, conn)
        if entropy_p10 is not None:  # population was large enough
            mask = (
                (df["feat_entropy_emb"].fillna(1.0) <= entropy_p10)
                & (df["feat_ncd_spam"].fillna(1.0) <= ncd_p10)
                & (df["kin_c"] >= acc_p80)
                & (df["comment_count_90d"].fillna(0) > 5)
            )
            id_quantile = set(df.loc[mask, "product_id"].astype(int).tolist())

    if "v5_spam_score_top5_prod" in args.modes:
        n, p95 = build_score(df, conn)
        mask = df["spam_score"] >= p95
        id_score = set(df.loc[mask, "product_id"].astype(int).tolist())

    conn.close()

    # 5. Overlap
    if len(args.modes) > 1:
        print_overlap(id_fixed, id_quantile, id_score)

    print(f"\n✅ Done!")
    print(f"""
Verification SQL:
  SELECT version_tag, COUNT(*) AS cnt
  FROM public.ml_data_filters
  WHERE version_tag IN (
    'v5_spam_fixed_prod', 'v5_spam_quantile_prod', 'v5_spam_score_top5_prod'
  )
  GROUP BY version_tag ORDER BY version_tag;
""")


if __name__ == "__main__":
    main()
