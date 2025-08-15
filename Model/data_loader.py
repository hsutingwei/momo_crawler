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
    # SQL:
    # 1) derive per-product per-batch representative sales (align snapshots to comment batches; take last snapshot in batch)
    # 2) for each product, get current and next batch sales via LEAD
    # 3) join back to comments (by product_id and batch_time = pc.capture_time), produce label
    sql = """
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
    prod_batch_sales AS (
      SELECT
        product_id,
        batch_time,
        sales_count AS cur_sales,
        LEAD(sales_count) OVER (
          PARTITION BY product_id ORDER BY batch_time
        ) AS next_sales
      FROM batch_repr
    ),
    comments_base AS (
      SELECT
        pc.comment_id,
        pc.product_id,
        p.name,
        p.price::float AS price,
        p.keyword,
        pc.image_urls,
        pc.video_url,
        pc.reply_content,
        pc.score::float AS score,
        pc.like_count::int AS like_count,
        pc.capture_time AS comment_capture_time
      FROM product_comments pc
      JOIN products p ON p.id = pc.product_id
    ),
    comments_labeled AS (
      SELECT
        cb.comment_id,
        cb.product_id,
        cb.name,
        cb.price,
        cb.keyword,
        cb.image_urls,
        cb.video_url,
        cb.reply_content,
        cb.score,
        cb.like_count,
        cb.comment_capture_time,
        pbs.batch_time,
        pbs.cur_sales,
        pbs.next_sales,
        CASE
          WHEN pbs.next_sales IS NULL THEN 0
          WHEN pbs.next_sales > pbs.cur_sales THEN 1
          ELSE 0
        END AS y_next_increase
      FROM comments_base cb
      JOIN prod_batch_sales pbs
        ON pbs.product_id = cb.product_id
       AND pbs.batch_time = cb.comment_capture_time
    )
    SELECT
      cl.comment_id,
      cl.product_id,
      cl.name,
      cl.price,
      cl.keyword,
      /* image flags */
      CASE
        WHEN cl.image_urls IS NULL THEN 0
        WHEN jsonb_typeof(cl.image_urls) <> 'array' THEN 0
        WHEN jsonb_array_length(cl.image_urls) > 0 THEN 1 ELSE 0
      END AS has_image_urls,
      CASE
        WHEN cl.image_urls IS NULL THEN 0
        WHEN jsonb_typeof(cl.image_urls) <> 'array' THEN 0
        ELSE jsonb_array_length(cl.image_urls)
      END AS image_urls_count,
      /* video / reply flags */
      CASE WHEN cl.video_url IS NOT NULL AND cl.video_url <> '' THEN 1 ELSE 0 END AS has_video_url,
      CASE WHEN cl.reply_content IS NOT NULL AND cl.reply_content <> '' THEN 1 ELSE 0 END AS has_reply_content,
      /* scores */
      cl.score,
      cl.like_count,
      /* batch keys & label */
      cl.comment_capture_time,
      cl.batch_time AS batch_capture_time,
      cl.cur_sales,
      cl.next_sales,
      cl.y_next_increase
    FROM comments_labeled cl
    /* optional time-based filtering happens in Python after fetch */
    ;
    """
    df = pd.read_sql(sql, conn, parse_dates=["comment_capture_time", "batch_capture_time"])
    # 可選：丟掉沒有下一批（label 為 NULL）的資料，避免混淆
    if drop_rows_without_next_batch:
        df = df.loc[df["y_next_increase"].notna()].copy()
    df["y_next_increase"] = df["y_next_increase"].astype(int)
    # 依 cutoff 做 train/test mask（先留給上層使用）
    if date_cutoff:
        df["is_train"] = (df["batch_capture_time"] <= pd.to_datetime(date_cutoff))
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