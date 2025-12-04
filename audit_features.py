import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import sys
import os

# Add Model directory to path to import data_loader
sys.path.append(os.path.join(os.getcwd(), 'Model'))
from data_loader import load_product_level_training_set

def audit_features():
    print("Loading data for audit...")
    # Load data
    X_dense, X_tfidf, y, meta, vocab = load_product_level_training_set()
    
    # Features to audit
    features_to_audit = [
        "feat_semantic_entropy",
        "feat_temporal_burstiness",
        "feat_lexical_diversity",
        "category_fit_score",
        "quality_driven_momentum"
    ]
    
    print(f"\n{'='*80}")
    print(f"{'FEATURE AUDIT REPORT':^80}")
    print(f"{'='*80}\n")
    
    print(f"{'Feature':<30} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Zeros(%)':<10} | {'Corr(y)':<10}")
    print("-" * 105)
    
    for feat in features_to_audit:
        if feat not in X_dense.columns:
            print(f"{feat:<30} | {'NOT FOUND':<70}")
            continue
            
        col_data = X_dense[feat]
        
        # 1. Distribution Stats
        mean_val = col_data.mean()
        std_val = col_data.std()
        min_val = col_data.min()
        max_val = col_data.max()
        
        # 2. Sparsity (Zeros or NaNs)
        # Note: For burstiness, -1 might be the fill value for missing, but let's check exact 0s too.
        # For others, 0 is the fill value.
        zeros_count = (col_data == 0).sum()
        zeros_pct = (zeros_count / len(col_data)) * 100
        
        # 3. Correlation with Target
        # Handle NaNs just in case (though data_loader should handle them)
        valid_mask = ~col_data.isna()
        if valid_mask.sum() > 1:
            corr, p_val = pointbiserialr(col_data[valid_mask], y[valid_mask])
        else:
            corr = 0.0
            
        print(f"{feat:<30} | {mean_val:<10.4f} | {std_val:<10.4f} | {min_val:<10.4f} | {max_val:<10.4f} | {zeros_pct:<9.1f}% | {corr:<10.4f}")

    print("-" * 105)
    
    # 4. Logic Sanity Checks
    print("\nLOGIC SANITY CHECKS:")
    
    # Category Fit Check
    if "category_fit_score" in X_dense.columns:
        neg_fit = (X_dense["category_fit_score"] < 0).sum()
        print(f"Category Fit < 0: {neg_fit} rows (Should be 0)")
        
    # Entropy Check
    if "feat_semantic_entropy" in X_dense.columns:
        neg_ent = (X_dense["feat_semantic_entropy"] < 0).sum()
        print(f"Entropy < 0: {neg_ent} rows (Should be 0)")
        
    # Momentum Check
    if "quality_driven_momentum" in X_dense.columns:
        high_mom = (X_dense["quality_driven_momentum"].abs() > 1.0).sum()
        print(f"Momentum > 1.0 (abs): {high_mom} rows (Check for outliers)")

if __name__ == "__main__":
    audit_features()
