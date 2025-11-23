# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.data_loader import load_product_level_training_set

def main():
    print("=== Inspecting Label Strategies ===")
    
    # Define output directory
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model", "outputs_label_experiments")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data for each strategy
    # We use fixed parameters for data loading to ensure we get the same set of samples (products)
    # Key alignment: product_id (and representative_batch_time if fixed_window)
    
    common_params = {
        "date_cutoff": "2025-06-25",
        "top_n": 100,
        "vocab_mode": "global",
        # Ensure we get the same products
        "exclude_products": None, 
        "single_keyword": None
    }
    
    print("\nLoading Baseline (Absolute)...")
    _, _, y_baseline, meta_baseline, _ = load_product_level_training_set(
        **common_params,
        label_strategy="absolute",
        label_params={"delta_threshold": 10},
        return_meta_details=True # Get max_raw_delta/ratio here
    )
    
    print("Loading Hybrid Relaxed (Ratio=0.1)...")
    _, _, y_relaxed, meta_relaxed, _ = load_product_level_training_set(
        **common_params,
        label_strategy="hybrid",
        label_params={"delta_threshold": 10, "ratio_threshold": 0.1}
    )
    
    print("Loading Hybrid Strict (Ratio=0.3)...")
    _, _, y_strict, meta_strict, _ = load_product_level_training_set(
        **common_params,
        label_strategy="hybrid",
        label_params={"delta_threshold": 10, "ratio_threshold": 0.3}
    )
    
    # 2. Align and Merge
    # Assuming product_level loader returns one row per product (unique product_id)
    # We verify this assumption
    
    print("\nVerifying alignment...")
    
    # Check uniqueness
    if not meta_baseline["product_id"].is_unique:
        print("WARNING: product_id is not unique in baseline meta!")
    
    # Rename series for merging
    y_baseline.name = "y_baseline"
    y_relaxed.name = "y_relaxed"
    y_strict.name = "y_strict"
    
    # Create a master dataframe aligned by product_id
    # We use meta_baseline as the base
    df_master = meta_baseline.copy()
    df_master["y_baseline"] = y_baseline
    
    # Merge other ys
    # We assume the order is same if SQL is deterministic, but safe to merge on product_id
    # However, y series index is 0..N, not product_id. 
    # We need to map y back to product_id using meta
    
    # Helper to map y to product_id
    def map_y_to_pid(meta_df, y_series):
        temp = meta_df[["product_id"]].copy()
        temp["y"] = y_series
        return temp.set_index("product_id")["y"]

    y_relaxed_mapped = map_y_to_pid(meta_relaxed, y_relaxed)
    y_strict_mapped = map_y_to_pid(meta_strict, y_strict)
    
    df_master = df_master.merge(y_relaxed_mapped.rename("y_relaxed"), on="product_id", how="left")
    df_master = df_master.merge(y_strict_mapped.rename("y_strict"), on="product_id", how="left")
    
    # Fill NaNs if any (shouldn't be if sample sets are identical)
    df_master["y_relaxed"] = df_master["y_relaxed"].fillna(0).astype(int)
    df_master["y_strict"] = df_master["y_strict"].fillna(0).astype(int)
    
    N = len(df_master)
    print(f"Total Samples (N): {N}")
    
    # 3. Calculate Distribution Summary
    summary_rows = []
    
    strategies = [
        ("baseline_v1", "absolute", "delta=10", "y_baseline"),
        ("hybrid_relaxed", "hybrid", "delta=10, ratio=0.1", "y_relaxed"),
        ("hybrid_strict", "hybrid", "delta=10, ratio=0.3", "y_strict"),
    ]
    
    for name, strat, params, col in strategies:
        pos = df_master[col].sum()
        neg = N - pos
        rate = pos / N if N > 0 else 0
        summary_rows.append({
            "Experiment": name,
            "Strategy": strat,
            "Params": params,
            "N": N,
            "Pos": pos,
            "Neg": neg,
            "Pos Rate": f"{rate:.4f}"
        })
        
    df_summary = pd.DataFrame(summary_rows)
    
    # Output Summary
    print("\nDistribution Summary:")
    print(df_summary.to_string(index=False))
    
    df_summary.to_csv(os.path.join(output_dir, "label_distribution_summary.csv"), index=False)
    
    with open(os.path.join(output_dir, "label_distribution_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Label Distribution Summary\n\n")
        f.write(f"**Total Samples (N):** {N}\n")
        f.write("**Alignment Key:** `product_id` (One sample per product)\n\n")
        
        # Manual markdown table
        cols = df_summary.columns.tolist()
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df_summary.iterrows():
            row_str = [str(row[c]) for c in cols]
            f.write("| " + " | ".join(row_str) + " |\n")
            
    # 4. Analyze Dropped Samples (Baseline=1, Strict=0)
    print("\nAnalyzing Dropped Samples...")
    
    mask_dropped = (df_master["y_baseline"] == 1) & (df_master["y_strict"] == 0)
    df_dropped = df_master[mask_dropped].copy()
    
    mask_baseline_pos = (df_master["y_baseline"] == 1)
    df_baseline_pos = df_master[mask_baseline_pos].copy()
    
    print(f"Dropped Count: {len(df_dropped)}")
    
    # Save dropped samples details
    # Columns: product_id, name, max_raw_delta, max_raw_ratio, y_baseline, y_strict
    out_cols = ["product_id", "name", "max_raw_delta", "max_raw_ratio", "y_baseline", "y_strict"]
    # Ensure columns exist (max_raw_delta/ratio come from return_meta_details=True)
    available_cols = [c for c in out_cols if c in df_dropped.columns]
    
    df_dropped[available_cols].to_csv(os.path.join(output_dir, "hybrid_strict_dropped_samples.csv"), index=False)
    
    # Generate Analysis Report
    with open(os.path.join(output_dir, "hybrid_strict_analysis.md"), "w", encoding="utf-8") as f:
        f.write("# Hybrid Strict Dropped Samples Analysis\n\n")
        
        f.write("## Overview\n")
        f.write(f"- **Baseline Positives:** {len(df_baseline_pos)}\n")
        f.write(f"- **Dropped by Strict (Ratio<0.3):** {len(df_dropped)}\n")
        f.write(f"- **Drop Rate:** {len(df_dropped)/len(df_baseline_pos):.2%}\n\n")
        
        f.write("## Statistics Comparison\n\n")
        
        stats_cols = ["max_raw_delta", "max_raw_ratio"]
        
        f.write("### All Baseline Positives (y=1)\n")
        if all(c in df_baseline_pos.columns for c in stats_cols):
            desc = df_baseline_pos[stats_cols].describe().T
            # Manual table
            f.write("| Metric | Count | Mean | Std | Min | 25% | 50% | 75% | Max |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for idx, row in desc.iterrows():
                row_vals = [f"{x:.4f}" for x in row]
                f.write(f"| {idx} | {' | '.join(row_vals)} |\n")
        else:
            f.write("Missing meta details (max_raw_delta/ratio).\n")
            
        f.write("\n### Dropped Samples (y_base=1, y_strict=0)\n")
        if not df_dropped.empty and all(c in df_dropped.columns for c in stats_cols):
            desc = df_dropped[stats_cols].describe().T
            f.write("| Metric | Count | Mean | Std | Min | 25% | 50% | 75% | Max |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for idx, row in desc.iterrows():
                row_vals = [f"{x:.4f}" for x in row]
                f.write(f"| {idx} | {' | '.join(row_vals)} |\n")
        elif df_dropped.empty:
            f.write("No samples dropped.\n")
        else:
            f.write("Missing meta details.\n")
            
        f.write("\n## Conclusion\n")
        if not df_dropped.empty:
            avg_ratio = df_dropped["max_raw_ratio"].mean()
            avg_delta = df_dropped["max_raw_delta"].mean()
            f.write(f"The dropped samples have an average max growth ratio of **{avg_ratio:.4f}** and average delta of **{avg_delta:.2f}**.\n")
            f.write("These represent products that met the absolute delta threshold (10) but failed to meet the 30% relative growth requirement.\n")
            if avg_delta > 10 and avg_ratio < 0.3:
                 f.write("This indicates they likely had a high base sales volume, making a delta of 10+ insignificant in relative terms.\n")
        else:
            f.write("No samples were dropped, indicating all baseline positives also met the strict ratio criteria.\n")

    print(f"\nAnalysis complete. Reports saved to {output_dir}")

if __name__ == "__main__":
    main()
