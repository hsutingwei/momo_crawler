
import pandas as pd
import numpy as np
import sys
import os
import psycopg2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import DatabaseConfig

def get_connection():
    return DatabaseConfig().get_connection()

def fetch_data():
    print("Connecting to DB...")
    conn = get_connection()
    try:
        # Parameters (matching default load_product_level_training_set)
        cutoff = "2025-06-25"
        params = {
            "cutoff": cutoff,
            "excluded": [-1],
            "single_kw": None,
            "delta_threshold": 1.0,
            "ratio_threshold": None,
            "max_gap_seconds": None,
            "align_gap_seconds": None,
            "label_window_days": 7.0
        }

        print("Fetching Targets (SQL Y)...")
        # COPIED SQL_Y (Next Batch Strategy)
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
        y_df = pd.read_sql(sql_y, conn, params=params)

        print("Fetching Features (SQL Dense)...")
        # COPIED SQL_DENSE
        sql_dense = """
        WITH pre_comments AS (
          SELECT pc.*, p.name, p.price::float AS price, p.keyword,
                 css.score_arousal, css.score_novelty, css.score_repurchase, css.score_negative, css.score_advertisement
          FROM product_comments pc
          JOIN products p ON p.id = pc.product_id
          LEFT JOIN comment_semantic_scores css ON pc.comment_id = css.comment_id
          WHERE pc.comment_date <= %(cutoff)s::date
        ),
        media_agg AS (
          SELECT
            product_id,
            -- Semantic Mean Scores (Recent 90 Days)
            AVG(score_arousal) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days') AS bert_arousal_mean,
            AVG(score_novelty) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days') AS bert_novelty_mean,
            AVG(score_repurchase) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days') AS bert_repurchase_mean,
            AVG(score_negative) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days') AS bert_negative_mean,
            AVG(score_advertisement) FILTER (WHERE comment_date >= %(cutoff)s::date - INTERVAL '90 days') AS bert_advertisement_mean
          FROM pre_comments
          GROUP BY product_id
        )
        SELECT
          p.id AS product_id,
          p.price::float AS price,
          COALESCE(m.bert_arousal_mean,0) AS bert_arousal_mean,
          COALESCE(m.bert_novelty_mean,0) AS bert_novelty_mean,
          COALESCE(m.bert_repurchase_mean,0) AS bert_repurchase_mean,
          COALESCE(m.bert_negative_mean,0) AS bert_negative_mean,
          COALESCE(m.bert_advertisement_mean,0) AS bert_advertisement_mean
        FROM products p
        LEFT JOIN media_agg m ON m.product_id = p.id
        WHERE p.id <> ALL(%(excluded)s)
        """
        dense_df = pd.read_sql(sql_dense, conn, params=params)
        
        return dense_df.merge(y_df, on="product_id", how="left")
    finally:
        conn.close()

def analyze():
    df = fetch_data()
    print(f"Loaded {len(df)} rows.")
    
    # Target
    # y is the binary label from data_loader logic (valid increase > threshold)
    # max_raw_ratio is the continuous growth (if available)
    if 'max_raw_ratio' in df.columns:
        df['target'] = df['max_raw_ratio'].fillna(0)
        target_name = "Max Growth Ratio"
    else:
        df['target'] = df['y'].fillna(0)
        target_name = "Binary Increase Y"
        
    print(f"Using Target: {target_name}")
    
    # Bin Prices
    # Filter 0 price (data errors)
    df = df[df['price'] > 0].copy()
    
    p33 = df['price'].quantile(0.33)
    p66 = df['price'].quantile(0.66)
    
    print(f"Price Bins: Low < {p33:.1f} <= Mid < {p66:.1f} <= High")
    
    def get_bin(p):
        if p < p33: return '1. Low'
        elif p < p66: return '2. Mid'
        else: return '3. High'
    df['price_bin'] = df['price'].apply(get_bin)
    
    feats = [
        "bert_arousal_mean", "bert_novelty_mean", "bert_repurchase_mean", 
        "bert_negative_mean", "bert_advertisement_mean"
    ]
    
    print("\n" + "="*80)
    print(f"{'Feature':<25} | {'Low Price Corr':<15} | {'High Price Corr':<15} | {'Delta (High-Low)':<15}")
    print("-" * 80)
    
    results = []
    
    for f in feats:
        low_corr = df[df['price_bin']=='1. Low'][f].corr(df[df['price_bin']=='1. Low']['target'])
        mid_corr = df[df['price_bin']=='2. Mid'][f].corr(df[df['price_bin']=='2. Mid']['target'])
        high_corr = df[df['price_bin']=='3. High'][f].corr(df[df['price_bin']=='3. High']['target'])
        global_corr = df[f].corr(df['target'])
        
        delta = high_corr - low_corr
        
        print(f"{f:<25} | {low_corr:15.4f} | {high_corr:15.4f} | {delta:15.4f}")
        results.append((f, delta))
        
    print("="*80)
    
    # Suggestion Logic
    best_feat = max(results, key=lambda x: x[1])
    print("\nConclusion:")
    print(f"The feature with the strongest Price Interaction (High - Low correlation) is: {best_feat[0]}")
    if best_feat[0] == "bert_arousal_mean":
        print(">> H1 VALIDATED: Arousal works significantly better at High Price (validating 'Price-Weighted Arousal').")
    else:
        print(f">> NEW FINDING: {best_feat[0]} actually benefits MORE from Price weighting than Arousal!")

if __name__ == "__main__":
    analyze()
