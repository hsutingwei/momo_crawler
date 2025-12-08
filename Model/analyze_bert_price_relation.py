
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.data_loader import load_product_level_training_set

def analyze():
    print("Loading Training Set (this may take a moment)...")
    # Load data
    # We use load_product_level_training_set to get the full feature set
    try:
        X_dense, _, y, meta, _ = load_product_level_training_set(top_n=1)
    except Exception as e:
        print(f"Error loading training set: {e}")
        return

    print(f"Data Loaded. Rows: {len(X_dense)}")

    # Combine relevant columns
    df = X_dense.copy()
    
    # We prefer continuous growth rate if available
    # meta contains 'cur_sales', 'next_sales'
    if 'next_sales' in meta.columns and 'cur_sales' in meta.columns:
        # Avoid division by zero
        # Growth = (Next - Cur) / (Cur + 1)
        # We might want to filter for items that actually have some activity or stay to all?
        # Let's keep all.
        df['target'] = (meta['next_sales'] - meta['cur_sales']) / (meta['cur_sales'] + 1.0)
        target_name = "Sales Growth Rate"
    else:
        df['target'] = y
        target_name = "Binary Increase Label"

    print(f"Target Variable: {target_name}")

    # Features to analyze
    # Note: 'clean_arousal_score' might strict the arousal too much for this raw analysis, 
    # but 'bert_arousal_mean' is the raw BERT output.
    bert_features = [
        "bert_arousal_mean", 
        "bert_novelty_mean", 
        "bert_repurchase_mean", 
        "bert_negative_mean", 
        "bert_advertisement_mean"
    ]
    
    # Price Binning
    # We simply use percentiles to divide into 3 roughly equal bins
    p33 = df['price'].quantile(0.33)
    p66 = df['price'].quantile(0.66)
    
    print(f"Price Bins Cutoffs: Low < {p33:.1f} <= Mid < {p66:.1f} <= High")
    
    def get_bin(p):
        if p < p33: return '1. Low'
        elif p < p66: return '2. Mid'
        else: return '3. High'
        
    df['price_bin'] = df['price'].apply(get_bin)
    
    print("\n" + "="*60)
    print(f"{'Dimension':<25} | {'Low Price':<10} | {'Mid Price':<10} | {'High Price':<10} | {'Global':<10}")
    print("-" * 60)
    
    results = []

    for feat in bert_features:
        if feat not in df.columns:
            print(f"Warning: {feat} not found in columns")
            continue
            
        # Calculate Global Correlation
        global_corr = df[feat].corr(df['target'])
        
        # Calculate Per-Bin Correlation
        row_str = f"{feat:<25} | "
        corrs = []
        for bin_name in ['1. Low', '2. Mid', '3. High']:
            sub = df[df['price_bin'] == bin_name]
            if len(sub) > 10:
                corr = sub[feat].corr(sub['target'])
            else:
                corr = np.nan
            corrs.append(corr)
            row_str += f"{corr:10.4f} | "
        
        row_str += f"{global_corr:10.4f}"
        print(row_str)
        results.append((feat, corrs, global_corr))

    print("="*60)
    print("\nInterpretation Guide:")
    print("- Positive Correlation (>0): Feature predicts GROWTH.")
    print("- Negative Correlation (<0): Feature predicts DECLINE.")
    print("- If 'Low Price' correlation is LOW/NEG and 'High Price' is HIGH, hypothesis H1 (Cheap Hype) is confirmed.")

if __name__ == "__main__":
    analyze()
