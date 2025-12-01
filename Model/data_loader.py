# -*- coding: utf-8 -*-
"""
Data loader for model training.
- Connects DB via your DatabaseConfig (same as csv_to_db.py)
- Builds features and labels from your schema
"""

from __future__ import annotations
import os
from typing import Tuple, Optional, List, Dict

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.sparse import csr_matrix, coo_matrix, hstack

# use same project path style as csv_to_db.py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.database import DatabaseConfig  # <-- same as csv_to_db.py

# ====================== 銷售級距相關工具函數 ======================

# 銷售級距定義（離散 bucket）
SALES_BUCKETS = [0, 1, 5, 10, 20, 30, 50, 100, 500, 1000, 3000, 5000, 8000, 10000, 
                 30000, 50000, 100000, 150000, 200000, 300000, 400000, 500000, 1000000]

def get_bucket_gap(prev_bucket: int, next_bucket: int) -> int:
    """
    計算兩個級距之間的差距
    
    Args:
        prev_bucket: 前一個級距值
        next_bucket: 下一個級距值
    
    Returns:
        級距差距
    """
    return next_bucket - prev_bucket

def compute_adaptive_delta_threshold(prev_sales: int, 
                                     base_delta: float = 12.0,
                                     low_bucket_threshold: int = 30,
                                     low_bucket_delta: float = 5.0) -> float:
    """
    【未來可調整的接口】根據前一個銷售級距計算自適應的 delta 門檻
    
    設計意圖：
    - 低級距商品（<30）使用較小的 delta 門檻，因為相鄰級距差距小
    - 高級距商品（≥30）使用標準 delta 門檻，因為相鄰級距差距大
    
    目前實作：簡單的兩階段門檻
    未來可擴展為：
    - 階梯式門檻（多個級距區間對應不同門檻）
    - 基於「跳幾級」的定義（例如：至少跳 2 級才算成長）
    
    Args:
        prev_sales: 前一個銷售級距值
        base_delta: 標準 delta 門檻（預設 12）
        low_bucket_threshold: 低級距的閾值（預設 30）
        low_bucket_delta: 低級距使用的 delta 門檻（預設 5）
    
    Returns:
        計算後的 delta 門檻
    """
    if prev_sales < low_bucket_threshold:
        return low_bucket_delta
    else:
        return base_delta

def get_bucket_levels_jumped(prev_sales: int, next_sales: int) -> int:
    """
    【未來可調整的接口】計算跳了幾級
    
    可以用這個函數來定義「至少跳 N 級才算成長」，而不是用固定 delta
    
    Args:
        prev_sales: 前一個銷售級距值
        next_sales: 下一個銷售級距值
    
    Returns:
        跳躍的級數（例如：5→20 跳了 2 級，30→50 跳了 1 級）
    """
    if prev_sales >= next_sales:
        return 0
    
    prev_idx = None
    next_idx = None
    for i, bucket in enumerate(SALES_BUCKETS):
        if prev_sales >= bucket:
            prev_idx = i
        if next_sales >= bucket:
            next_idx = i
    
    if prev_idx is None or next_idx is None:
        return 0
    
    return next_idx - prev_idx


def get_connection():
    """Get a psycopg2 connection using your DatabaseConfig."""
    db_config = DatabaseConfig()
    return db_config.get_connection()


def fetch_top_terms(conn,
                    top_n: int = 100,
                    pipeline_version: Optional[str] = None,
                    product_id: Optional[int] = None,
                    min_len: int = 2,
                    max_len: int = 4) -> List[str]:
    """
    Return global (or per product) TF-IDF top-N tokens for building binary features.

    - If product_id is None => global Top-N across all comments
    - Else => Top-N under the specified product
    - If pipeline_version provided and nlp_stopwords exists, stopwords will be excluded.
    """
    sql = """
    WITH prod_comments AS (
      SELECT pc.comment_id
      FROM product_comments pc
      WHERE (%(product_id)s IS NULL OR pc.product_id = %(product_id)s)
    )
    SELECT
      ts.token,
      SUM(ts.tfidf) AS total_tfidf
    FROM tfidf_scores ts
    JOIN prod_comments pc
      ON pc.comment_id = ts.comment_id
    LEFT JOIN nlp_stopwords sw
      ON (%(pipeline_version)s IS NOT NULL
          AND sw.pipeline_version = %(pipeline_version)s
          AND sw.token = ts.token)
    WHERE (%(pipeline_version)s IS NULL OR sw.token IS NULL)
      AND char_length(ts.token) BETWEEN %(min_len)s AND %(max_len)s
      AND ts.token !~ '[[:space:][:punct:]]'
    GROUP BY ts.token
    ORDER BY total_tfidf DESC
    LIMIT %(top_n)s;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, dict(
            product_id=product_id,
            pipeline_version=pipeline_version,
            min_len=min_len,
            max_len=max_len,
            top_n=top_n
        ))
        rows = cur.fetchall()
    return [r["token"] for r in rows]


def fetch_comments_with_features_and_label(
    conn,
    pipeline_version: Optional[str] = None,
    date_cutoff: Optional[str] = "2025-06-25",
    drop_rows_without_next_batch: bool = True,
) -> pd.DataFrame:
    """
    Build a row-per-comment table with:
      - product/comment meta features
      - label y: whether NEXT batch sales_count > CURRENT batch (per product; batches keyed by comment capture_time)
    We align sales snapshots to nearest comment batch <= snapshot_time, then get per-product per-batch representative sales,
    then label each comment in that batch by comparing current vs next batch.

    NOTE: This returns a *wide* table with scalar features; TF tokens are *not* pivoted here.
          Use `fetch_comment_token_pairs_for_vocab` to build sparse TF features for chosen vocabulary.
    """
    # 改為呼叫新的資料集函數（SQL functions 已於資料庫中建立）
    sql = """
    SELECT
      comment_id,
      product_id,
      name,
      price,
      keyword,
      has_image_urls,
      image_urls_count,
      has_video_url,
      has_reply_content,
      score,
      like_count,
      comment_capture_time,
      batch_capture_time,
      cur_sales,
      next_sales,
      y_next_increase
    FROM ml_fn_build_comment_dataset_v2();
    """
    df = pd.read_sql(sql, conn, parse_dates=["comment_capture_time", "batch_capture_time"])
    # 可選：丟掉沒有下一批（label 為 NULL）的資料，避免混淆
    if drop_rows_without_next_batch:
        df = df.loc[df["y_next_increase"].notna()].copy()
    df["y_next_increase"] = df["y_next_increase"].astype(int)
    # 依 cutoff 做 train/test mask；統一使用 UTC-aware 比較，避免 tz 衝突
    if date_cutoff:
        cutoff_ts = pd.to_datetime(date_cutoff, utc=True)
        # 若 batch_capture_time 非 tz-aware，補齊為 UTC
        if df["batch_capture_time"].dt.tz is None:
            df["batch_capture_time"] = df["batch_capture_time"].dt.tz_localize("UTC")
        df["is_train"] = (df["batch_capture_time"] <= cutoff_ts)
    else:
        df["is_train"] = True
    return df


def fetch_comment_token_pairs_for_vocab(
    conn,
    vocab_tokens: List[str],
    pipeline_version: Optional[str] = None,
) -> pd.DataFrame:
    """
    For a chosen vocabulary (list of tokens), fetch (comment_id, token) pairs where token appears in the comment.
    Then you can pivot to a sparse 1/0 matrix.

    We read from tfidf_scores (or tfidf_term_freq) to leverage existed joins.
    """
    if not vocab_tokens:
        return pd.DataFrame(columns=["comment_id", "token"])

    # 用 UNNEST 傳 vocab
    sql = """
    WITH vocab AS (
      SELECT UNNEST(%(vocab_tokens)s::text[]) AS token
    )
    SELECT
      ts.comment_id,
      ts.token
    FROM tfidf_scores ts
    JOIN vocab v ON v.token = ts.token
    /* 如果要限定 pipeline_version 的 stopwords 已處理，這邊不需再限制 */
    GROUP BY ts.comment_id, ts.token;
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, dict(vocab_tokens=vocab_tokens))
        rows = cur.fetchall()
    return pd.DataFrame(rows)  # columns: comment_id, token


def build_sparse_binary_matrix(pairs: pd.DataFrame, vocab: List[str], id_index: pd.Series) -> csr_matrix:
    """
    Build a CSR sparse matrix of shape (n_comments, vocab_size) with 1/0 presence.
    - pairs: DataFrame[comment_id, token]
    - vocab: list of tokens (column order)
    - id_index: Series mapping comment_id -> row idx (0..n-1)
    """
    if pairs.empty or len(vocab) == 0 or id_index.empty:
        return csr_matrix((len(id_index), len(vocab)), dtype=np.int8)

    token_to_col = {t: j for j, t in enumerate(vocab)}
    # 只保留 vocab 內 token
    pairs = pairs[pairs["token"].isin(token_to_col.keys())].copy()
    pairs["row"] = pairs["comment_id"].map(id_index)
    pairs["col"] = pairs["token"].map(token_to_col)
    pairs = pairs.dropna(subset=["row", "col"])
    rows = pairs["row"].astype(int).to_numpy()
    cols = pairs["col"].astype(int).to_numpy()
    data = np.ones(len(pairs), dtype=np.int8)
    mat = coo_matrix((data, (rows, cols)), shape=(len(id_index), len(vocab)), dtype=np.int8).tocsr()
    return mat


def load_training_set(
    top_n: int = 100,
    pipeline_version: Optional[str] = None,
    date_cutoff: Optional[str] = "2025-06-25",
    vocab_scope: str = "global",  # 'global' or 'product' (留作擴充)
) -> Tuple[pd.DataFrame, csr_matrix, pd.Series, pd.DataFrame]:
    """
    High-level loader:
      - fetch base scalar features + label per comment
      - fetch/compute vocab (Top-N) and build sparse binary TF matrix
      - return (X_dense_df, X_tfidf_sparse, y_series, meta_df)

    X_dense_df columns:
      ['price', 'has_image_urls', 'image_urls_count', 'has_video_url',
       'has_reply_content', 'score', 'like_count']
    y_series:
      0/1 indicating next batch sales increase
    meta_df:
      ['comment_id', 'product_id', 'keyword', 'batch_capture_time', 'cur_sales', 'next_sales', 'is_train']
    """
    conn = get_connection()
    try:
        # 1) 取結構化 + 標籤
        base = fetch_comments_with_features_and_label(
            conn,
            pipeline_version=pipeline_version,
            date_cutoff=date_cutoff,
            drop_rows_without_next_batch=True,
        )

        # 2) 取 vocab（Top-N token）
        if vocab_scope == "product":
            # 這裡可以依 product 分別求 Top-N，再合併去重；先用 global 比較直覺
            vocab = fetch_top_terms(conn, top_n=top_n, pipeline_version=pipeline_version, product_id=None)
        else:
            vocab = fetch_top_terms(conn, top_n=top_n, pipeline_version=pipeline_version, product_id=None)

        # 3) 取 (comment_id, token) 對，再轉成 1/0 稀疏矩陣
        pairs = fetch_comment_token_pairs_for_vocab(conn, vocab_tokens=vocab, pipeline_version=pipeline_version)
        id_index = pd.Series(range(len(base)), index=base["comment_id"])  # map comment_id to row index
        X_tfidf = build_sparse_binary_matrix(pairs, vocab=vocab, id_index=id_index)

        # 4) Dense features
        dense_cols = ["price", "has_image_urls", "image_urls_count", "has_video_url",
                      "has_reply_content", "score", "like_count"]
        # 缺失處理
        for c in dense_cols:
            if c not in base.columns:
                base[c] = 0
        X_dense = base[dense_cols].fillna(0).astype(float)

        # 5) y 與 meta
        y = base["y_next_increase"].astype(int)
        meta = base[["comment_id", "product_id", "keyword", "batch_capture_time", "cur_sales", "next_sales", "is_train"]].copy()

        return X_dense, X_tfidf, y, meta, vocab

    finally:
        conn.close()

# ====================== PRODUCT-LEVEL LOADER (one row per product) ======================

def load_product_level_training_set(
    date_cutoff: str = "2025-06-25",
    top_n: int = 100,
    pipeline_version: Optional[str] = None,
    vocab_mode: str = "global",
    single_keyword: Optional[str] = None,
    exclude_products: Optional[List[int]] = None,
    label_delta_threshold: float = 1.0,
    label_ratio_threshold: Optional[float] = None,
    label_max_gap_days: Optional[float] = None,
    label_mode: str = "next_batch",
    label_window_days: float = 7.0,
    align_max_gap_days: Optional[float] = None,
    min_comments: int = 0,
    keyword_whitelist: Optional[List[str]] = None,
    keyword_blacklist: Optional[List[str]] = None,
    label_strategy: str = "absolute",
    label_params: Optional[Dict] = None,
    return_meta_details: bool = False,
) -> Tuple[pd.DataFrame, csr_matrix, pd.Series, pd.DataFrame, List[str]]:
    """
    ?? (X_dense_df, X_tfidf_sparse, y_series, meta_df, vocab)

    - ? product_id ???????
    - label_mode: next_batch / fixed_window
    """
    exclude_products = exclude_products or []
    conn = get_connection()
    try:
        # ====================== Label Strategy Resolution ======================
        # 'absolute' = 目前正式線的既有行為 (只看 delta)
        # 'hybrid'   = 方案 A (絕對 delta + 相對成長率 ratio)
        
        eff_ratio_threshold = label_ratio_threshold  # Default to argument value
        
        if label_strategy == "absolute":
            # Force ratio check to be disabled (None) for absolute strategy
            eff_ratio_threshold = None
        elif label_strategy == "hybrid":
            if not label_params or "ratio_threshold" not in label_params:
                raise ValueError("Strategy 'hybrid' requires 'ratio_threshold' in label_params")
            eff_ratio_threshold = float(label_params["ratio_threshold"])
        elif label_strategy == "multiclass":
             # For multiclass, we use SQL to filter candidates (delta >= 10)
             # and then refine classes in Python based on ratio_buckets.
             # So we disable ratio check in SQL (eff_ratio_threshold = None)
             # and force return_meta_details = True to get max_raw_ratio.
             eff_ratio_threshold = None
             return_meta_details = True
        elif label_strategy == "default":
             # Keep existing behavior (use arguments as is)
             pass
        else:
             # For Phase 1 & 2, we support absolute, hybrid, and multiclass
             raise ValueError(f"Unsupported label_strategy: {label_strategy}")

        align_gap_seconds = (align_max_gap_days * 86400.0) if align_max_gap_days else None
        params = {
            "cutoff": date_cutoff,
            "excluded": exclude_products if exclude_products else [-1],
            "single_kw": single_keyword,
            "delta_threshold": label_delta_threshold,
            "ratio_threshold": eff_ratio_threshold,
            "max_gap_seconds": (label_max_gap_days * 86400.0) if label_max_gap_days else None,
            "label_window_days": label_window_days,
            "align_gap_seconds": align_gap_seconds
        }
        label_mode = (label_mode or "next_batch").lower()
        if label_mode == "next_batch":
            sql_y = """
        WITH comment_batches AS (
          SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
          FROM product_comments pc
          JOIN products p ON p.id = pc.product_id
          WHERE (%(single_kw)s IS NULL OR p.keyword = %(single_kw)s)
          GROUP BY p.keyword, pc.product_id, pc.capture_time
        ),
        snap_mapped AS (
          SELECT
            s.product_id,
            p.keyword,
            s.capture_time AS snapshot_time,
            s.sales_count,
            cb_near.batch_time
          FROM sales_snapshots s
          JOIN products p ON p.id = s.product_id
          LEFT JOIN LATERAL (
            SELECT cb.batch_time
            FROM comment_batches cb
            WHERE cb.keyword = p.keyword
              AND cb.product_id = s.product_id
              AND cb.batch_time <= s.capture_time
              AND (%(align_gap_seconds)s IS NULL
                   OR s.capture_time >= cb.batch_time - INTERVAL '1 second' * %(align_gap_seconds)s)
            ORDER BY cb.batch_time DESC
            LIMIT 1
          ) AS cb_near ON TRUE
        ),
        batch_repr AS (
          SELECT *
          FROM (
            SELECT
              product_id,
              keyword,
              batch_time,
              sales_count,
              snapshot_time,
              ROW_NUMBER() OVER (
                PARTITION BY product_id, batch_time
                ORDER BY snapshot_time DESC
              ) AS rn
            FROM snap_mapped
            WHERE batch_time IS NOT NULL
          ) t
          WHERE rn = 1
        ),
        seq AS (
          SELECT
            product_id,
            keyword,
            batch_time,
            snapshot_time,
            sales_count,
            LAG(sales_count) OVER (PARTITION BY product_id ORDER BY batch_time) AS prev_sales
          FROM batch_repr
        ),
        y_post AS (
          SELECT
            product_id,
            MAX(
              CASE
                WHEN batch_time > %(cutoff)s::timestamp
                     AND prev_sales IS NOT NULL
                     -- 【級距差異說明】銷售級距是離散 bucket：0, 1, 5, 10, 20, 30, 50, 100, 500, 1000, ...
                     -- 相鄰級距差距：0→1(1), 1→5(4), 5→10(5), 10→20(10), 20→30(10), 30→50(20), 50→100(50), ...
                     -- 當 delta_threshold=12 時：
                     --   - 低級距商品（<30）：相鄰級距差距都 < 12，需要一次跳兩級以上才會滿足條件（例如 5→20, 10→30）
                     --   - 高級距商品（≥30）：相鄰級距差距都 ≥ 12，一次跳一級就會被視為「大成長」（例如 30→50, 50→100）
                     -- 這是一個相對「保守，只抓大成長」的設計，對低級距商品更嚴格
                     -- 【未來調整點】可以考慮階梯式門檻（見 data_loader.py 中的 compute_adaptive_delta_threshold 函數）
                     AND (sales_count - prev_sales) >= %(delta_threshold)s
                     AND (
                          %(ratio_threshold)s IS NULL
                          OR prev_sales = 0
                          OR (((sales_count::float / NULLIF(prev_sales, 0)) - 1) >= %(ratio_threshold)s)
                     )
                     AND (
                          %(max_gap_seconds)s IS NULL
                          OR EXTRACT(EPOCH FROM (snapshot_time - batch_time)) <= %(max_gap_seconds)s
                     )
                THEN 1 ELSE 0
              END
            ) AS y,
            MAX(
              CASE
                WHEN batch_time > %(cutoff)s::timestamp
                     AND prev_sales IS NOT NULL
                     AND (
                          %(max_gap_seconds)s IS NULL
                          OR EXTRACT(EPOCH FROM (snapshot_time - batch_time)) <= %(max_gap_seconds)s
                     )
                THEN (sales_count - prev_sales)
                ELSE NULL
              END
            ) AS max_raw_delta,
            MAX(
              CASE
                WHEN batch_time > %(cutoff)s::timestamp
                     AND prev_sales IS NOT NULL
                     AND prev_sales > 0
                     AND (
                          %(max_gap_seconds)s IS NULL
                          OR EXTRACT(EPOCH FROM (snapshot_time - batch_time)) <= %(max_gap_seconds)s
                     )
                THEN ((sales_count::float / prev_sales) - 1)
                ELSE NULL
              END
            ) AS max_raw_ratio
          FROM seq
          GROUP BY product_id
        )
        SELECT product_id, y, max_raw_delta, max_raw_ratio
        FROM y_post
        WHERE product_id <> ALL(%(excluded)s)
        """
        elif label_mode == "fixed_window":
            # fixed_window 模式：取消 cutoff 过滤，返回所有数据用于 time-based split
            sql_y = """
        WITH comment_batches AS (
          SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
          FROM product_comments pc
          JOIN products p ON p.id = pc.product_id
          WHERE (%(single_kw)s IS NULL OR p.keyword = %(single_kw)s)
          GROUP BY p.keyword, pc.product_id, pc.capture_time
        ),
        prev_snap AS (
          SELECT
            cb.product_id,
            cb.batch_time,
            ps.snapshot_time AS prev_snapshot_time,
            ps.sales_count AS prev_sales
          FROM comment_batches cb
          LEFT JOIN LATERAL (
            SELECT s.capture_time AS snapshot_time, s.sales_count
            FROM sales_snapshots s
            WHERE s.product_id = cb.product_id
              AND s.capture_time <= cb.batch_time
              AND (%(align_gap_seconds)s IS NULL
                   OR s.capture_time >= cb.batch_time - INTERVAL '1 second' * %(align_gap_seconds)s)
            ORDER BY s.capture_time DESC
            LIMIT 1
          ) ps ON TRUE
        ),
        future_snap AS (
          SELECT
            cb.product_id,
            cb.batch_time,
            cb.prev_sales,
            MAX(s.sales_count) AS future_max_sales
          FROM prev_snap cb
          LEFT JOIN sales_snapshots s
            ON s.product_id = cb.product_id
           AND s.capture_time > cb.batch_time
           AND s.capture_time <= cb.batch_time + INTERVAL '1 day' * %(label_window_days)s
          GROUP BY cb.product_id, cb.batch_time, cb.prev_sales
        ),
        y_post AS (
          SELECT
            cb.product_id,
            -- 【重要】使用 MIN(batch_time) 作為 representative_batch_time，用於 time-based split
            -- 設計理由：每個商品可能有多個 batch_time，選擇最早的可以確保該商品的所有歷史資訊都在訓練集中
            -- 避免同一個商品同時出現在訓練集和測試集，造成 data leakage
            -- 【潛在問題】如果某個商品的 batch_time 分布不均，可能會導致 time-based split 時某些時間區間沒有樣本
            MIN(cb.batch_time) AS representative_batch_time,
            MAX(
              CASE
                WHEN cb.prev_sales IS NULL THEN 0
                WHEN fs.future_max_sales IS NULL THEN 0
                -- 【級距差異說明】銷售級距是離散 bucket：0, 1, 5, 10, 20, 30, 50, 100, ...
                -- 相鄰級距差距：0→1(1), 1→5(4), 5→10(5), 10→20(10), 20→30(10), 30→50(20), 50→100(50), ...
                -- 當 delta_threshold=12 時：
                --   - 低級距商品（<30）：相鄰級距差距都 < 12，需要一次跳兩級以上才會滿足條件（例如 5→20, 10→30）
                --   - 高級距商品（≥30）：相鄰級距差距都 ≥ 12，一次跳一級就會被視為「大成長」（例如 30→50, 50→100）
                -- 這是一個相對「保守，只抓大成長」的設計，對低級距商品更嚴格
                -- 【未來調整點】可以考慮：
                --   1. 低級距用較小的 delta 門檻（例如 delta=5 for sales < 30）
                --   2. 或用「跳幾級」來定義成長，而不是死用固定 delta
                --   3. 在 label_delta_threshold 參數中預留階梯式門檻的接口
                WHEN (fs.future_max_sales - cb.prev_sales) >= %(delta_threshold)s
                     AND (
                          %(ratio_threshold)s IS NULL
                          OR cb.prev_sales = 0
                          OR ((fs.future_max_sales::float / NULLIF(cb.prev_sales, 0)) - 1) >= %(ratio_threshold)s
                     )
                THEN 1 ELSE 0
              END
            ) AS y,
            MAX(
                CASE
                    WHEN cb.prev_sales IS NULL THEN NULL
                    WHEN fs.future_max_sales IS NULL THEN NULL
                    ELSE (fs.future_max_sales - cb.prev_sales)
                END
            ) AS max_raw_delta,
            MAX(
                CASE
                    WHEN cb.prev_sales IS NULL THEN NULL
                    WHEN cb.prev_sales = 0 THEN NULL
                    WHEN fs.future_max_sales IS NULL THEN NULL
                    ELSE ((fs.future_max_sales::float / cb.prev_sales) - 1)
                END
            ) AS max_raw_ratio
          FROM prev_snap cb
          LEFT JOIN future_snap fs
            ON fs.product_id = cb.product_id
           AND fs.batch_time = cb.batch_time
          GROUP BY cb.product_id
        )
        SELECT product_id, y, representative_batch_time, max_raw_delta, max_raw_ratio
        FROM y_post
        WHERE product_id <> ALL(%(excluded)s)
        """
        else:
            raise ValueError(f"Unsupported label_mode: {label_mode}")

        y_df = pd.read_sql(sql_y, conn, params=params)
        if y_df.empty:
            if label_mode == "fixed_window":
                y_df = pd.DataFrame(columns=["product_id", "y", "representative_batch_time"])
            else:
                y_df = pd.DataFrame(columns=["product_id", "y"])
        # 确保 representative_batch_time 存在（fixed_window 模式）
        if label_mode == "fixed_window" and "representative_batch_time" not in y_df.columns:
            y_df["representative_batch_time"] = None

        # 在 fixed_window 模式下，取消 cutoff 过滤以支持 time-based split
        # Modified to use comment_date for cutoff filtering to ensure we include all comments posted before cutoff,
        # even if captured later (assuming they would have been visible).
        cutoff_filter = "" if label_mode == "fixed_window" else "WHERE pc.comment_date <= %(cutoff)s::date"
        sql_dense = f"""
        WITH pre_comments AS (
          SELECT pc.*, p.name, p.price::float AS price, p.keyword
          FROM product_comments pc
          JOIN products p ON p.id = pc.product_id
          {cutoff_filter}
        ),
        media_agg AS (
          SELECT
            product_id,
            MAX(
              CASE
                WHEN image_urls IS NULL THEN 0
                WHEN jsonb_typeof(image_urls) <> 'array' THEN 0
                WHEN jsonb_array_length(image_urls) > 0 THEN 1 ELSE 0
              END
            ) AS has_image_urls,
            SUM(
              CASE
                WHEN image_urls IS NULL THEN 0
                WHEN jsonb_typeof(image_urls) <> 'array' THEN 0
                ELSE jsonb_array_length(image_urls)
              END
            ) AS image_urls_count,
            MAX(CASE WHEN video_url IS NOT NULL AND video_url <> '' THEN 1 ELSE 0 END) AS has_video_url,
            MAX(CASE WHEN reply_content IS NOT NULL AND reply_content <> '' THEN 1 ELSE 0 END) AS has_reply_content,
            AVG(score::float) FILTER (WHERE score IS NOT NULL) AS score_mean,
            SUM(like_count::int) FILTER (WHERE like_count IS NOT NULL) AS like_count_sum,
            COUNT(*) AS comment_count_pre,
            -- Temporal Features (Modified to use comment_date instead of capture_time)
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '7 days') AS comment_count_7d,
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '30 days') AS comment_count_30d,
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days') AS comment_count_90d,
            EXTRACT(EPOCH FROM (%(cutoff)s::timestamp - MAX(comment_date))) / 86400.0 AS days_since_last_comment,
            -- Temporal Trend Features (90-day segmented windows)
            -- comment_3rd_30d: 0-30 days before cutoff (Recent)
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '30 days') AS comment_3rd_30d,
            -- comment_2nd_30d: 31-60 days before cutoff
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '60 days' 
                               AND comment_date < %(cutoff)s::date - INTERVAL '30 days') AS comment_2nd_30d,
            -- comment_1st_30d: 61-90 days before cutoff
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days' 
                               AND comment_date < %(cutoff)s::date - INTERVAL '60 days') AS comment_1st_30d,
            -- Content Features (Recent 90 Days)
            AVG(score::float) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days') AS sentiment_mean_recent,
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days' AND score >= 4) AS pos_count_recent,
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days' AND score <= 2) AS neg_count_recent,
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days' AND comment_text ~ '促銷|特價|打折|滿額|免運|團購') AS promo_count_recent,
            COUNT(*) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days' AND comment_text ~ '回購|囤貨|囤了|再買|買爆') AS repurchase_count_recent
          FROM pre_comments
          GROUP BY product_id
        ),
        pre_seq AS (
          WITH comment_batches AS (
            SELECT p.keyword, pc.product_id, pc.capture_time AS batch_time
            FROM product_comments pc
            JOIN products p ON p.id = pc.product_id
            GROUP BY p.keyword, pc.product_id, pc.capture_time
          ),
          snap_mapped AS (
            SELECT
              s.product_id,
              p.keyword,
              s.capture_time AS snapshot_time,
              s.sales_count,
              cb_near.batch_time
            FROM sales_snapshots s
            JOIN products p ON p.id = s.product_id
            LEFT JOIN LATERAL (
              SELECT cb.batch_time
              FROM comment_batches cb
              WHERE cb.keyword = p.keyword
                AND cb.product_id = s.product_id
                AND cb.batch_time <= s.capture_time
              ORDER BY cb.batch_time DESC
              LIMIT 1
            ) AS cb_near ON TRUE
          ),
          batch_repr AS (
            SELECT *
            FROM (
              SELECT
                product_id,
                keyword,
                batch_time,
                sales_count,
                snapshot_time,
                ROW_NUMBER() OVER (
                  PARTITION BY product_id, batch_time
                  ORDER BY snapshot_time DESC
                ) AS rn
              FROM snap_mapped
              WHERE batch_time IS NOT NULL
            ) t
            WHERE rn = 1
          ),
          seq AS (
            SELECT
              product_id,
              batch_time,
              sales_count,
              LAG(sales_count) OVER (PARTITION BY product_id ORDER BY batch_time) AS prev_sales
            FROM batch_repr
          )
          SELECT
            product_id,
            MAX(CASE WHEN """ + (("TRUE" if label_mode == "fixed_window" else "batch_time <= %(cutoff)s::timestamp")) + """
                        AND prev_sales IS NOT NULL
                        AND sales_count IS DISTINCT FROM prev_sales
                     THEN 1 ELSE 0 END) AS had_any_change_pre,
            COUNT(*) FILTER (WHERE """ + (("TRUE" if label_mode == "fixed_window" else "batch_time <= %(cutoff)s::timestamp")) + """
                               AND prev_sales IS NOT NULL
                               AND sales_count > prev_sales) AS num_increases_pre
          FROM seq
          GROUP BY product_id
        )
        SELECT
          p.id AS product_id,
          p.name,
          p.price::float AS price,
          p.keyword,
          COALESCE(m.has_image_urls,0) AS has_image_urls,
          COALESCE(m.image_urls_count,0) AS image_urls_count,
          COALESCE(m.has_video_url,0) AS has_video_url,
          COALESCE(m.has_reply_content,0) AS has_reply_content,
          COALESCE(m.score_mean,0) AS score_mean,
          COALESCE(m.like_count_sum,0) AS like_count_sum,
          COALESCE(m.comment_count_pre,0) AS comment_count_pre,
          COALESCE(m.comment_count_7d,0) AS comment_count_7d,
          COALESCE(m.comment_count_30d,0) AS comment_count_30d,
          COALESCE(m.comment_count_90d,0) AS comment_count_90d,
          COALESCE(m.comment_1st_30d,0) AS comment_1st_30d,
          COALESCE(m.comment_2nd_30d,0) AS comment_2nd_30d,
          COALESCE(m.comment_3rd_30d,0) AS comment_3rd_30d,
          COALESCE(m.days_since_last_comment, 365) AS days_since_last_comment,
          COALESCE(m.sentiment_mean_recent, 3.0) AS sentiment_mean_recent, -- Default to neutral 3.0 if no recent comments
          COALESCE(m.pos_count_recent,0) AS pos_count_recent,
          COALESCE(m.neg_count_recent,0) AS neg_count_recent,
          COALESCE(m.promo_count_recent,0) AS promo_count_recent,
          COALESCE(m.repurchase_count_recent,0) AS repurchase_count_recent,
          COALESCE(s.had_any_change_pre,0) AS had_any_change_pre,
          COALESCE(s.num_increases_pre,0) AS num_increases_pre
        FROM products p
        LEFT JOIN media_agg m ON m.product_id = p.id
        LEFT JOIN pre_seq s   ON s.product_id = p.id
        WHERE p.id <> ALL(%(excluded)s)
        """
        dense = pd.read_sql(sql_dense, conn, params=params)

        # Fillna for dense features
        dense_cols = [
            "price", "comment_count_pre", "score_mean", "like_count_sum",
            "comment_count_7d", "comment_count_30d", "comment_count_90d", "days_since_last_comment",
            "comment_1st_30d", "comment_2nd_30d", "comment_3rd_30d",
            "sentiment_mean_recent", "pos_count_recent", "neg_count_recent",
            "promo_count_recent", "repurchase_count_recent",
            "has_image_urls", "has_video_url", "has_reply_content",
            "had_any_change_pre", "num_increases_pre"
        ]
        for c in dense_cols:
            if c not in dense.columns:
                dense[c] = 0
        
        dense = dense.fillna(0)
        
        # Calculate Ratios
        # comment_7d_ratio = comment_count_7d / (comment_count_pre + 1)
        dense["comment_7d_ratio"] = dense["comment_count_7d"] / (dense["comment_count_pre"] + 1)
        
        # Temporal Trend Ratio: Recent 30d vs Previous 60d
        # ratio_recent30_to_prev60 = comment_3rd_30d / (comment_1st_30d + comment_2nd_30d + 1e-6)
        dense["ratio_recent30_to_prev60"] = dense["comment_3rd_30d"] / (dense["comment_1st_30d"] + dense["comment_2nd_30d"] + 1e-6)

        # Content Ratios
        # Use comment_count_90d as denominator for recent features
        # Add 1 to denominator to avoid division by zero
        dense["neg_ratio_recent"] = dense["neg_count_recent"] / (dense["comment_count_90d"] + 1)
        dense["promo_ratio_recent"] = dense["promo_count_recent"] / (dense["comment_count_90d"] + 1)
        dense["repurchase_ratio_recent"] = dense["repurchase_count_recent"] / (dense["comment_count_90d"] + 1)
        
        # Handle days_since_last_comment for items with no comments
        # If comment_count_pre == 0, days_since_last_comment will be 365 (from COALESCE) or 0 (if not matched)
        # We already handled COALESCE in SQL, but let's be safe
        dense.loc[dense["comment_count_pre"] == 0, "days_since_last_comment"] = 365.0

        df = dense.merge(y_df, on="product_id", how="left")
        
        # ====================== Multiclass Label Logic ======================
        if label_strategy == "multiclass":
            # y_df['y'] from SQL is 1 if delta >= threshold (and ratio check skipped), 0 otherwise.
            # We need to split y=1 into classes based on ratio_buckets.
            # Class 0: y=0 (delta < threshold)
            # Class 1: y=1 and ratio < bucket[0]
            # Class 2: y=1 and bucket[0] <= ratio < bucket[1]
            # ...
            # Class N: y=1 and ratio >= bucket[-1]
            
            buckets = label_params.get("ratio_buckets", [])
            if not buckets:
                raise ValueError("Strategy 'multiclass' requires 'ratio_buckets' in label_params")
            
            print(f"[DEBUG] Multiclass Buckets: {buckets}")
            # Check max_raw_ratio stats for y=1
            y1_ratios = df.loc[df["y"]==1, "max_raw_ratio"]
            print(f"[DEBUG] y=1 max_raw_ratio stats:\n{y1_ratios.describe()}")
            
            def assign_class(row):
                # If y is 0 or NaN (no data), it's Class 0
                if pd.isna(row["y"]) or row["y"] == 0:
                    return 0
                # y=1, check ratio
                ratio = row["max_raw_ratio"]
                # If ratio is NaN (e.g. prev_sales=0), it implies infinite growth -> Highest Class
                if pd.isna(ratio):
                    return len(buckets) + 1
                
                for i, b in enumerate(buckets):
                    if ratio < b:
                        return i + 1
                return len(buckets) + 1
            
            # Ensure max_raw_ratio is available (it should be because we forced return_meta_details=True)
            if "max_raw_ratio" not in df.columns:
                 # Try to merge from y_df if not already merged (dense merge might not include it if not in dense sql)
                 # Actually y_df has it.
                 pass

            df["y"] = df.apply(assign_class, axis=1)

        df["y"] = df["y"].fillna(0).astype(int)

        # ====================== Keyword 篩選邏輯 ======================
        # 【設計理由】排除特定 keyword 的原因：
        # - 「口罩」：在 2020-2021 年 COVID-19 期間，口罩商品的銷售模式極度異常
        #   （大量囤貨、政府配給、價格波動劇烈），與一般消費商品的評論-銷售關聯模式差異過大
        #   若納入訓練會導致模型學習到異常模式，影響對正常商品的預測能力
        # - 未來若要調整 blacklist，請修改此處的註解說明，並確保：
        #   1. DB 中 products.keyword 欄位的實際值（可用 SQL: SELECT DISTINCT keyword FROM products）
        #   2. CLI 參數 --keyword-blacklist 傳入的值
        #   3. 此處比對時使用的值
        #   三者編碼一致（建議統一使用 UTF-8）
        
        if keyword_whitelist:
            df = df[df["keyword"].isin(keyword_whitelist)]
        if keyword_blacklist:
            # 【編碼修正】確保 blacklist 中的關鍵字編碼正確
            # 將可能的亂碼修正為正確的 UTF-8 編碼
            # 常見問題：Windows 系統預設編碼（cp950/cp936）可能導致中文字符亂碼
            normalized_blacklist = []
            for kw in keyword_blacklist:
                # 修正已知的亂碼問題
                if kw == "??蔗" or kw == "??" or (isinstance(kw, bytes) and b'\xbf\xbf' in kw):
                    normalized_kw = "口罩"
                else:
                    normalized_kw = kw
                normalized_blacklist.append(normalized_kw)
            df = df[~df["keyword"].isin(normalized_blacklist)]
        
        if min_comments > 0 and "comment_count_pre" in df.columns:
            df = df[df["comment_count_pre"] >= min_comments]

        vocab = fetch_top_terms(conn,
                                top_n=top_n,
                                pipeline_version=pipeline_version,
                                product_id=None,
                                min_len=2, max_len=4)

        if vocab:
            # 在 fixed_window 模式下，取消 cutoff 过滤
            cutoff_where = "" if label_mode == "fixed_window" else "WHERE pc.capture_time <= %(cutoff)s::timestamp"
            sql_pairs = f"""
            WITH vocab AS (
              SELECT UNNEST(%(tokens)s::text[]) AS token
            )
            SELECT DISTINCT
              pc.product_id,
              ts.token
            FROM tfidf_scores ts
            JOIN product_comments pc ON pc.comment_id = ts.comment_id
            JOIN vocab v ON v.token = ts.token
            {cutoff_where}
              AND pc.product_id <> ALL(%(excluded)s)
            """
            pairs = pd.read_sql(sql_pairs, conn, params={"tokens": vocab, **params})
        else:
            pairs = pd.DataFrame(columns=["product_id", "token"])

        # 在 fixed_window 模式下，meta 需要包含 representative_batch_time 用于 time-based split
        if label_mode == "fixed_window" and "representative_batch_time" in y_df.columns:
            meta = df[["product_id", "name", "keyword"]].copy()
            # 合并 representative_batch_time
            meta = meta.merge(y_df[["product_id", "representative_batch_time"]], on="product_id", how="left")
            # 确保时间格式正确
            if "representative_batch_time" in meta.columns:
                meta["representative_batch_time"] = pd.to_datetime(meta["representative_batch_time"], utc=True)
        else:
            meta = df[["product_id", "name", "keyword"]].copy()
            
        if return_meta_details:
            # Merge max_raw_delta and max_raw_ratio from y_df
            if "max_raw_delta" in y_df.columns:
                meta = meta.merge(y_df[["product_id", "max_raw_delta", "max_raw_ratio"]], on="product_id", how="left")
        id_index = pd.Series(range(len(meta)), index=meta["product_id"])
        if pairs.empty or not vocab:
            X_tfidf = csr_matrix((len(meta), 0), dtype=np.int8)
        else:
            token_to_col = {t: j for j, t in enumerate(vocab)}
            pairs = pairs[pairs["token"].isin(token_to_col.keys())].copy()
            pairs["row"] = pairs["product_id"].map(id_index)
            pairs["col"] = pairs["token"].map(token_to_col)
            pairs = pairs.dropna(subset=["row", "col"])
            rows = pairs["row"].astype(int).to_numpy()
            cols = pairs["col"].astype(int).to_numpy()
            data = np.ones(len(pairs), dtype=np.int8)
            X_tfidf = coo_matrix((data, (rows, cols)),
                                 shape=(len(meta), len(vocab)), dtype=np.int8).tocsr()

        dense_cols = [
            "price",
            "has_image_urls", "image_urls_count", "has_video_url",
            "has_reply_content", "score_mean", "like_count_sum",
            "had_any_change_pre", "num_increases_pre", "comment_count_pre",
            "comment_count_7d", "comment_count_30d", "comment_count_90d", "days_since_last_comment", "comment_7d_ratio",
            "comment_1st_30d", "comment_2nd_30d", "comment_3rd_30d", "ratio_recent30_to_prev60",
            "sentiment_mean_recent", "neg_ratio_recent", "promo_ratio_recent", "repurchase_ratio_recent"
        ]
        for c in dense_cols:
            if c not in df.columns:
                df[c] = 0
        X_dense = df[dense_cols].fillna(0).astype(float)

        y = df["y"].astype(int)

        meta["is_train"] = True

        return X_dense, X_tfidf, y, meta, vocab

    finally:
        conn.close()
