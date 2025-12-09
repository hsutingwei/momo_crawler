import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
import psycopg2
import psycopg2.extras
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import DatabaseConfig
from Model.data_loader import load_product_level_training_set

# Suppress warnings
warnings.filterwarnings("ignore")

# ====================== 1. Feature Sets Definition ======================
base_cols = [
    "price", "has_image_urls", "has_video_url", "has_reply_content",
    "comment_count_pre", "score_mean", "like_count_sum",
    "comment_count_7d", "comment_count_30d", "comment_count_90d",
    "days_since_last_comment",
    "comment_1st_30d", "comment_2nd_30d", "comment_3rd_30d", "ratio_recent30_to_prev60",
    "sentiment_mean_recent", "neg_ratio_recent", "promo_ratio_recent",
    "had_any_change_pre", "num_increases_pre"
]

feats_price = [
    'price_weighted_arousal', 'price_weighted_novelty', 
    'bert_arousal_mean', 'bert_novelty_mean'
]

feats_kin = [
    'kin_acc_abs', 'kin_acc_rel', 'kin_jerk_abs', 
    'early_bird_momentum', 'validated_velocity'
]

feats_nov = [
    'category_fit_score', 'quality_driven_momentum', 
    'novelty_momentum', 'intensity_score', 'bert_novelty_mean'
]

def get_db_connection():
    db = DatabaseConfig()
    return db.get_connection()

def insert_filters(conn, version_tag, filter_level, items, reason, scores=None):
    """
    Batch insert filters. items can be list of strings (keywords) or ints (product_ids).
    scores: optional dict or list mapping item -> score.
    """
    if not items:
        return 0
        
    sql = """
    INSERT INTO ml_data_filters (version_tag, filter_level, filter_value, reason, score)
    VALUES %s
    ON CONFLICT (version_tag, filter_level, filter_value) DO NOTHING
    """
    
    values = []
    for i, item in enumerate(items):
        val_str = str(item)
        score_val = None
        if scores:
            if isinstance(scores, dict):
                score_val = scores.get(item)
            elif isinstance(scores, list) and len(scores) == len(items):
                score_val = scores[i]
        
        values.append((version_tag, filter_level, val_str, reason, score_val))
        
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, values)
    conn.commit()
    return len(values)

def main():
    print("Loading Data...")
    # Load dataset
    # Note: excluding nothing initially to calculate global metrics
    X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(
        label_strategy="absolute", 
        label_delta_threshold=10
    )
    
    # Ensure meta has necessary columns
    if "keyword" not in meta.columns:
        # data_loader might return 'keyword' in meta cols list
        print("Warning: keyword not found in meta columns.")
    
    print(f"Data Loaded: {len(X_dense_df)} samples.")

    conn = get_db_connection()

    # ====================== Version 1: v1_corr_kw ======================
    print("\n[Version 1] Generating 'v1_corr_kw' (Keyword Correlation)...")
    
    # Calculate correlation between 'comment_count_90d' and label 'y' per keyword
    # We can also use max_raw_ratio or other metrics, but 'y' is the direct target.
    # Actually, correlation with *growth* might be better represented by 'y' (0/1).
    # Since y is binary, point-biserial correlation is effectively Pearson.
    
    analysis_df = meta.copy()
    analysis_df['y'] = y
    analysis_df['volume'] = X_dense_df['comment_count_90d']
    
    keywords = analysis_df['keyword'].unique()
    low_corr_keywords = {}
    
    for kw in keywords:
        sub = analysis_df[analysis_df['keyword'] == kw]
        if len(sub) < 10: # Skip tiny categories
            continue
            
        # Avoid division by zero in correlation
        if sub['volume'].std() == 0 or sub['y'].std() == 0:
            corr = 0
        else:
            corr = sub['volume'].corr(sub['y'])
            
        if np.isnan(corr): corr = 0
        
        if corr < 0.05:
            low_corr_keywords[kw] = float(corr)
            
    # Insert
    count_v1 = insert_filters(conn, 'v1_corr_kw', 'keyword', list(low_corr_keywords.keys()), 'low_correlation', list(low_corr_keywords.values()))
    print(f" -> Inserted {count_v1} keywords (Correlation < 0.05).")


    # ====================== Version 2: v2_error_prod (Ensemble) ======================
    print("\n[Version 2] Generating 'v2_error_prod' (Ensemble Error Check)...")
    
    # Define Feature Combinations
    feature_sets = {
        "Base": base_cols,
        "Price": base_cols + feats_price,
        "Kin": base_cols + feats_kin,
        "Nov": base_cols + feats_nov
    }
    
    # Store predictions: {product_id: {model_name: is_correct}}
    # Or simpler: DataFrame with index=product_id, columns=[model_names...]
    
    results = pd.DataFrame(index=meta['product_id'])
    y_true = y.values
    
    # Calculate Class Weight (Imbalance Handling)
    n_pos = np.sum(y_true)
    n_neg = len(y_true) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f" -> Class Imbalance: Pos={n_pos}, Neg={n_neg}, scale_pos_weight={scale_pos_weight:.4f}")
    
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" -> Training with device: {device}")

    for name, cols in feature_sets.items():
        print(f"    Training Model: {name} ...")
        
        # Prepare X
        # Handle missing columns safely
        missing = [c for c in cols if c not in X_dense_df.columns]
        if missing:
            print(f"      Warning: Missing columns in X_dense: {missing}")
        
        valid_cols = [c for c in cols if c in X_dense_df.columns]
        X_subset = X_dense_df[valid_cols]
        from Model.run_experiments import std_scaler_to_sparse 
        X_subset_sparse = std_scaler_to_sparse(X_subset)
        X_final = hstack([X_subset_sparse, X_tfidf], format="csr")
        
        model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=6, 
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,  # ADDED: Imbalance Handling
            # GPU settings
            device=device,
            tree_method="hist" if device == "cuda" else "auto"
        )
        
        # Cross Val Predict
        y_pred = cross_val_predict(model, X_final, y_true, cv=kf, n_jobs=-1 if device=="cpu" else 1)
        
        # Determine strict correctness (TP or TN)
        # Actually misclassification means (y_pred != y_true)
        is_wrong = (y_pred != y_true).astype(int)
        results[name] = is_wrong

    # Count failures
    results['fail_count'] = results.sum(axis=1)
    
    # Filter Rule: Fail count >= 3
    bad_products = results[results['fail_count'] >= 3]
    
    # Insert
    count_v2 = insert_filters(conn, 'v2_error_prod', 'product_id', bad_products.index.tolist(), 'ensemble_error_count', bad_products['fail_count'].tolist())
    print(f" -> Inserted {count_v2} products (Misclassified by >= 3/4 models).")
    
    
    # ====================== Version 3: v3_error_kw ======================
    print("\n[Version 3] Generating 'v3_error_kw' (Keyword Error Rate)...")
    
    # Use Base model predictions (already calculated in results['Base'])
    # Join with meta to get keywords
    
    meta_with_err = meta.copy()
    meta_with_err = meta_with_err.set_index('product_id')
    meta_with_err['is_wrong'] = results['Base']
    
    # Group by keyword
    kw_stats = meta_with_err.groupby('keyword')['is_wrong'].agg(['mean', 'count'])
    # Filter: Error Rate > 0.5 (and meaningful count)
    high_error_kws = kw_stats[(kw_stats['mean'] > 0.5) & (kw_stats['count'] >= 5)]
    
    # Insert
    kw_list = high_error_kws.index.tolist()
    scores = high_error_kws['mean'].tolist()
    
    count_v3 = insert_filters(conn, 'v3_error_kw', 'keyword', kw_list, 'high_error_rate', scores)
    print(f" -> Inserted {count_v3} keywords (Error Rate > 0.5).")
    
    
    # ====================== Summary Report ======================
    print("\n" + "="*50)
    print("FILTER GENERATION SUMMARY")
    print("="*50)
    print(f"{'Version':<20} | {'Type':<10} | {'Deleted':<8} | {'Remaining (Est)':<10}")
    print("-" * 56)
    
    total_prods = len(meta)
    
    # v1 stats (approximate impact)
    v1_kws = set(low_corr_keywords.keys())
    v1_impact = meta[meta['keyword'].isin(v1_kws)].shape[0]
    print(f"{'v1_corr_kw':<20} | {'keyword':<10} | {v1_impact:<8} | {total_prods - v1_impact:<10}")
    
    # v2 stats
    v2_impact = len(bad_products)
    print(f"{'v2_error_prod':<20} | {'product':<10} | {v2_impact:<8} | {total_prods - v2_impact:<10}")
    
    # v3 stats
    v3_kws = set(high_error_kws.index)
    v3_impact = meta[meta['keyword'].isin(v3_kws)].shape[0]
    print(f"{'v3_error_kw':<20} | {'keyword':<10} | {v3_impact:<8} | {total_prods - v3_impact:<10}")
    
    conn.close()

if __name__ == "__main__":
    main()
