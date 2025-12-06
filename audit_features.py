import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import sys
import os

# Add Model directory to path to import data_loader
sys.path.append(os.path.join(os.getcwd(), 'Model'))
from data_loader import load_product_level_training_set

def analyze_feature(df, col, target_col):
    col_data = df[col]
    
    # 1. Distribution Stats
    mean_val = col_data.mean()
    std_val = col_data.std()
    min_val = col_data.min()
    max_val = col_data.max()
    
    # 2. Sparsity (Zeros or NaNs)
    zeros_count = (col_data == 0).sum()
    zeros_pct = (zeros_count / len(col_data)) * 100
    
    # 3. Correlation with Target
    valid_mask = ~col_data.isna()
    if valid_mask.sum() > 1:
        corr, p_val = pointbiserialr(col_data[valid_mask], df[target_col][valid_mask])
    else:
        corr = 0.0
        
    print(f"{col:<30} | {mean_val:<10.4f} | {std_val:<10.4f} | {min_val:<10.4f} | {max_val:<10.4f} | {zeros_pct:<9.1f}% | {corr:<10.4f}")

def audit_features():
    print("Loading data for audit...")
    # Load data
    result = load_product_level_training_set()
    X_dense, X_tfidf, y, meta, vocab = result
    
    # Combine for analysis
    df = X_dense.copy()
    target_col = "target"
    df[target_col] = y
    
    print(f"\n{'='*80}")
    print(f"{'FEATURE AUDIT REPORT':^80}")
    print(f"{'='*80}\n")
    
    print(f"{'Feature':<30} | {'Mean':<10} | {'Std':<10} | {'Min':<10} | {'Max':<10} | {'Zeros(%)':<10} | {'Corr(y)':<10}")
    print("-" * 105)
    
    # 1. Kinematics
    kin_cols = ["kin_v_1", "kin_v_2", "kin_v_3", "kin_acc_abs", "kin_acc_rel", "kin_jerk_abs"]
    print(f"\n{'='*20} Kinematics Features {'='*20}")
    for col in kin_cols:
        if col in df.columns:
            analyze_feature(df, col, target_col)
        else:
            print(f"Feature {col} not found!")

    # 2. Algorithm Features
    algo_cols = ["category_fit_score", "quality_driven_momentum"]
    print(f"\n{'='*20} Algorithm Features {'='*20}")
    for col in algo_cols:
        if col in df.columns:
            analyze_feature(df, col, target_col)
        else:
            print(f"Feature {col} not found!")

    # 3. Diversity & Organic Features
    div_cols = [
        "feat_entropy_tfidf", "feat_entropy_emb", 
        "feat_temporal_burstiness", "feat_lexical_diversity",
        "feat_compression_ratio", "feat_ncd_spam"
    ]
    print(f"\n{'='*20} Diversity & Organic Features {'='*20}")
    for col in div_cols:
        if col in df.columns:
            analyze_feature(df, col, target_col)
            
            # Check correlation with Entropy (to see if NCD adds new info)
            if "feat_entropy_emb" in df.columns and col != "feat_entropy_emb":
                corr = df[col].corr(df["feat_entropy_emb"])
                print(f"  Correlation with Entropy (SBERT): {corr:.4f}")
        else:
            print(f"Feature {col} not found!")

    print("-" * 105)
    
    # 4. Logic Sanity Checks
    print("\nLOGIC SANITY CHECKS:")
    
    if "category_fit_score" in df.columns:
        neg_fit = (df["category_fit_score"] < 0).sum()
        print(f"Category Fit < 0: {neg_fit} rows (Should be 0)")
        
    if "feat_entropy_emb" in df.columns:
        neg_ent = (df["feat_entropy_emb"] < 0).sum()
        print(f"Entropy < 0: {neg_ent} rows (Should be 0)")

if __name__ == "__main__":
    audit_features()
