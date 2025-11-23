# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.data_loader import load_product_level_training_set

def main():
    print("=== Simulating Phase 2 Label Schemas ===")
    
    # 1. Load Data (Baseline settings to get raw delta/ratio)
    print("Loading data...")
    _, _, y_baseline, meta, _ = load_product_level_training_set(
        date_cutoff="2025-06-25",
        top_n=100,
        vocab_mode="global",
        label_strategy="absolute",
        label_params={"delta_threshold": 10},
        return_meta_details=True
    )
    
    # Ensure we have the necessary columns
    df = meta.copy()
    df["y_baseline"] = y_baseline
    
    # Fill NaNs for calculation
    df["max_raw_delta"] = df["max_raw_delta"].fillna(0)
    df["max_raw_ratio"] = df["max_raw_ratio"].fillna(0)
    
    total_n = len(df)
    print(f"Total Samples: {total_n}")
    
    # ==========================================
    # Schema A: Strict Ratio Levels
    # 0: No Growth (delta < 10 OR ratio < 0.1)
    # 1: Moderate Growth (delta >= 10 AND 0.1 <= ratio < 0.3)
    # 2: High Growth (delta >= 10 AND ratio >= 0.3)
    # ==========================================
    print("\n--- Schema A: Strict Ratio Levels ---")
    
    def apply_schema_a(row):
        if row["max_raw_delta"] < 10: return 0
        if row["max_raw_ratio"] < 0.1: return 0
        if row["max_raw_ratio"] < 0.3: return 1
        return 2
        
    df["schema_A"] = df.apply(apply_schema_a, axis=1)
    dist_a = df["schema_A"].value_counts().sort_index()
    print(dist_a)
    print(f"Class 0: {dist_a.get(0,0)} ({dist_a.get(0,0)/total_n:.2%})")
    print(f"Class 1: {dist_a.get(1,0)} ({dist_a.get(1,0)/total_n:.2%}) - Moderate (0.1 <= r < 0.3)")
    print(f"Class 2: {dist_a.get(2,0)} ({dist_a.get(2,0)/total_n:.2%}) - High (r >= 0.3)")
    
    # ==========================================
    # Schema B: Growth Quality (Volume vs Efficiency)
    # 0: No Growth (delta < 10)
    # 1: Volume Growth (delta >= 10 BUT ratio < 0.2) -> "Stable Giants"
    # 2: Efficiency Growth (delta >= 10 AND ratio >= 0.2) -> "True Growth"
    # ==========================================
    print("\n--- Schema B: Growth Quality (Volume vs Efficiency) ---")
    
    def apply_schema_b(row):
        if row["max_raw_delta"] < 10: return 0
        if row["max_raw_ratio"] < 0.2: return 1
        return 2
        
    df["schema_B"] = df.apply(apply_schema_b, axis=1)
    dist_b = df["schema_B"].value_counts().sort_index()
    print(dist_b)
    print(f"Class 0: {dist_b.get(0,0)} ({dist_b.get(0,0)/total_n:.2%})")
    print(f"Class 1: {dist_b.get(1,0)} ({dist_b.get(1,0)/total_n:.2%}) - Volume (d>=10, r<0.2)")
    print(f"Class 2: {dist_b.get(2,0)} ({dist_b.get(2,0)/total_n:.2%}) - Efficiency (d>=10, r>=0.2)")

    # ==========================================
    # Schema C: Percentile Based
    # Base: delta >= 10
    # Levels: 0 (Not base), 1 (Low tier), 2 (Mid tier), 3 (High tier)
    # ==========================================
    print("\n--- Schema C: Percentile Based ---")
    
    # Filter base population
    base_mask = df["max_raw_delta"] >= 10
    base_df = df[base_mask]
    
    if not base_df.empty:
        # Calculate percentiles of ratio
        p33 = np.percentile(base_df["max_raw_ratio"], 33)
        p66 = np.percentile(base_df["max_raw_ratio"], 66)
        print(f"Percentiles (of items with delta>=10): P33={p33:.4f}, P66={p66:.4f}")
        
        def apply_schema_c(row):
            if row["max_raw_delta"] < 10: return 0
            r = row["max_raw_ratio"]
            if r < p33: return 1
            if r < p66: return 2
            return 3
            
        df["schema_C"] = df.apply(apply_schema_c, axis=1)
        dist_c = df["schema_C"].value_counts().sort_index()
        print(dist_c)
        print(f"Class 0: {dist_c.get(0,0)} ({dist_c.get(0,0)/total_n:.2%}) - Non-growth")
        print(f"Class 1: {dist_c.get(1,0)} ({dist_c.get(1,0)/total_n:.2%}) - Low Tier (r < {p33:.2f})")
        print(f"Class 2: {dist_c.get(2,0)} ({dist_c.get(2,0)/total_n:.2%}) - Mid Tier ({p33:.2f} <= r < {p66:.2f})")
        print(f"Class 3: {dist_c.get(3,0)} ({dist_c.get(3,0)/total_n:.2%}) - Top Tier (r >= {p66:.2f})")
    else:
        print("No samples met baseline delta criteria.")

    # ==========================================
    # Sample Examples
    # ==========================================
    print("\n--- Sample Examples ---")
    cols = ["product_id", "max_raw_delta", "max_raw_ratio", "y_baseline", "schema_A", "schema_B", "schema_C"]
    
    # Pick 2 examples from each class of Schema C (if available)
    examples = []
    if "schema_C" in df.columns:
        for cls in sorted(df["schema_C"].unique()):
            sample = df[df["schema_C"] == cls].head(2)
            examples.append(sample[cols])
            
    if examples:
        final_sample = pd.concat(examples)
        print(final_sample.to_string(index=False))
    
if __name__ == "__main__":
    main()
