
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import sys
import os
import psycopg2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import DatabaseConfig

def get_connection():
    return DatabaseConfig().get_connection()

def fetch_data_fast():
    print("Connecting to DB (Fast Mode)...")
    conn = get_connection()
    try:
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

        # 1. Fetch Y (Target)
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
            ) AS y
          FROM seq
          GROUP BY product_id
        )
        SELECT product_id, y
        FROM y_post
        WHERE product_id <> ALL(%(excluded)s)
        """
        y_df = pd.read_sql(sql_y, conn, params=params)

        # 2. Fetch Dense Features (SQL Only)
        # Note: We skip complex Python features (entropy, SBERT) and sparse TF-IDF to be fast
        # We rely on the core dense features + BERT scores which are already in DB
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
            MAX(
              CASE
                WHEN image_urls IS NULL THEN 0
                WHEN jsonb_typeof(image_urls) <> 'array' THEN 0
                WHEN jsonb_array_length(image_urls) > 0 THEN 1 ELSE 0
              END
            ) AS has_image_urls,
            AVG(score::float) FILTER (WHERE score IS NOT NULL) AS score_mean,
            SUM(like_count::int) FILTER (WHERE like_count IS NOT NULL) AS like_count_sum,
            COUNT(*) AS comment_count_pre,
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
          COALESCE(m.has_image_urls, 0) AS has_image_urls,
          COALESCE(m.score_mean, 0) AS score_mean,
          COALESCE(m.like_count_sum, 0) AS like_count_sum,
          COALESCE(m.comment_count_pre, 0) AS comment_count_pre,
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

def run_experiment_fast():
    df = fetch_data_fast()
    print(f"Data Loaded. Shape: {df.shape}")
    
    # Fill NA
    df = df.fillna(0)
    y = df['y']
    
    # ==========================================
    # 2. Feature Engineering
    # ==========================================
    
    # A. The "Clean" Logic (Previous Attempt)
    df["clean_arousal_score"] = df["bert_arousal_mean"] * (1 - df["bert_negative_mean"]) * (1 - df["bert_advertisement_mean"])
    df["price_weighted_arousal_clean"] = df["clean_arousal_score"] * np.log1p(df["price"])
    
    # B. The "Raw" Logic (New Hypothesis based on Controversy Analysis)
    # We do NOT penalize Negative/Ads because analysis showed they correlate with growth.
    df["price_weighted_arousal_raw"] = df["bert_arousal_mean"] * np.log1p(df["price"])
    df["price_weighted_novelty_raw"] = df["bert_novelty_mean"] * np.log1p(df["price"]) # Novelty also showed promise
    
    # Base Features
    base_feats = [
        "price", "has_image_urls", "score_mean", "like_count_sum", "comment_count_pre",
        "bert_arousal_mean", "bert_novelty_mean", "bert_repurchase_mean",
        "bert_negative_mean", "bert_advertisement_mean"
    ]
    
    # Define Sets
    X_baseline = df[base_feats]
    
    # Set 1: "Clean" (Original Hypothesis)
    X_exp_clean = df[base_feats + ["price_weighted_arousal_clean"]]
    
    # Set 2: "Raw" (Data-Driven Hypothesis)
    X_exp_raw = df[base_feats + ["price_weighted_arousal_raw", "price_weighted_novelty_raw"]]
    
    sets = {
        "Baseline": X_baseline,
        "Exp (Clean)": X_exp_clean,
        "Exp (Raw)": X_exp_raw
    }

    # ==========================================
    # 3. GPU/Resource Setup
    # ==========================================
    # Check for GPU
    use_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass
        
    xgb_params = {
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    if use_gpu:
        print(">> Using GPU for XGBoost training.")
        xgb_params['tree_method'] = 'hist'
        xgb_params['device'] = 'cuda'
    else:
        print(">> GPU not found/torch not installed. Using CPU.")
        xgb_params['n_jobs'] = -1  # Use all CPU cores

    # ==========================================
    # 4. Training Loop
    # ==========================================
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {name: {'precision': [], 'recall': [], 'f1': [], 'prauc': []} for name in sets}

    print("\nStarting 5-Fold CV...")
    
    fold = 1
    for train_index, test_index in kf.split(df, y):
        print(f"  Fold {fold}/5...", end="\r")
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        for name, X_data in sets.items():
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_data.iloc[train_index], y_train)
            preds = model.predict(X_data.iloc[test_index])
            probs = model.predict_proba(X_data.iloc[test_index])[:, 1]
            
            results[name]['precision'].append(precision_score(y_test, preds, zero_division=0))
            results[name]['recall'].append(recall_score(y_test, preds, zero_division=0))
            results[name]['f1'].append(f1_score(y_test, preds, zero_division=0))
            results[name]['prauc'].append(average_precision_score(y_test, probs))
        
        fold += 1

    # Analysis
    def avg(l): return np.mean(l)
    
    print("\n\n" + "="*80)
    print(f"{'Metric':<10} | {'Baseline':<10} | {'Exp (Clean)':<12} | {'Exp (Raw)':<12} | {'Lift (Raw)':<10}")
    print("-" * 80)
    
    final_report = {}
    for m in ['precision', 'recall', 'f1', 'prauc']:
        base = avg(results['Baseline'][m])
        clean = avg(results['Exp (Clean)'][m])
        raw = avg(results['Exp (Raw)'][m])
        
        lift = (raw - base) / base if base > 0 else 0
        final_report[m] = lift
        
        print(f"{m.upper():<10} | {base:.4f}     | {clean:.4f}       | {raw:.4f}       | {lift:+.2%}")
    print("="*80)
    
    if final_report['precision'] > 0:
        print("\nSUCCESS: Raw Price-Interactions improved the model!")
    else:
        print("\nNOTE: Price-Interactions might require tuning or non-linear models.")

if __name__ == "__main__":
    run_experiment_fast()
