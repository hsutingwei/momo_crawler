# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import sys

def main():
    print("=== Analyzing Binary Explosive Errors ===")
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "Model", "outputs_label_experiments", "binary_explosive_error_analysis.csv")
    out_md_path = os.path.join(base_dir, "Model", "outputs_label_experiments", "binary_explosive_error_analysis.md")
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples.")
    
    # Define categories based on y_true and y_pred_best
    # y_pred_best is already in the CSV
    
    conditions = [
        (df["y_true"] == 1) & (df["y_pred_best"] == 1), # TP
        (df["y_true"] == 0) & (df["y_pred_best"] == 1), # FP
        (df["y_true"] == 1) & (df["y_pred_best"] == 0), # FN
        (df["y_true"] == 0) & (df["y_pred_best"] == 0)  # TN
    ]
    choices = ["TP", "FP", "FN", "TN"]
    df["category"] = np.select(conditions, choices, default="Unknown")
    
    print("Category Distribution:")
    print(df["category"].value_counts())
    
    # Features to analyze
    features = [
        "price", "comment_count_pre", "score_mean", "like_count_sum", 
        "had_any_change_pre", "num_increases_pre", 
        "has_image_urls", "has_video_url", "has_reply_content",
        "comment_count_7d", "comment_count_30d", "days_since_last_comment", "comment_7d_ratio"
    ]
    
    # Calculate stats per category
    stats = df.groupby("category")[features].agg(["mean", "median", "std"])
    
    # Generate Markdown Report
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# Binary Explosive Error Analysis\n\n")
        f.write(f"**Total Samples:** {len(df)}\n")
        f.write(f"**Best Threshold Used:** {df['best_threshold_used'].iloc[0]:.4f}\n\n")
        
        f.write("## 1. Category Distribution\n\n")
        dist = df["category"].value_counts()
        f.write("| Category | Count | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        for cat in ["TP", "FP", "FN", "TN"]:
            count = dist.get(cat, 0)
            pct = count / len(df) * 100
            f.write(f"| {cat} | {count} | {pct:.2f}% |\n")
        f.write("\n")
        
        f.write("## 2. Feature Comparison (Mean / Median)\n\n")
        
        for feat in features:
            f.write(f"### {feat}\n\n")
            f.write("| Category | Mean | Median | Std |\n")
            f.write("| --- | --- | --- | --- |\n")
            for cat in ["TP", "FP", "FN", "TN"]:
                if cat in stats.index:
                    m = stats.loc[cat, (feat, "mean")]
                    med = stats.loc[cat, (feat, "median")]
                    s = stats.loc[cat, (feat, "std")]
                    f.write(f"| {cat} | {m:.4f} | {med:.4f} | {s:.4f} |\n")
                else:
                    f.write(f"| {cat} | - | - | - |\n")
            f.write("\n")
            
        f.write("## 3. Key Observations\n\n")
        
        # Automated observations
        f.write("### TP vs FN (Missed Opportunities)\n")
        try:
            tp_price = stats.loc["TP", ("price", "median")]
            fn_price = stats.loc["FN", ("price", "median")]
            f.write(f"- **Price**: TP median ({tp_price}) vs FN median ({fn_price}).\n")
            
            tp_comments = stats.loc["TP", ("comment_count_pre", "median")]
            fn_comments = stats.loc["FN", ("comment_count_pre", "median")]
            f.write(f"- **Comments**: TP median ({tp_comments}) vs FN median ({fn_comments}).\n")
        except KeyError:
            f.write("- Insufficient data for comparison.\n")
            
        f.write("\n### FP vs TP (False Alarms)\n")
        try:
            fp_price = stats.loc["FP", ("price", "median")]
            f.write(f"- **Price**: FP median ({fp_price}) vs TP median ({tp_price}).\n")
            
            fp_comments = stats.loc["FP", ("comment_count_pre", "median")]
            f.write(f"- **Comments**: FP median ({fp_comments}) vs TP median ({tp_comments}).\n")
        except KeyError:
            f.write("- Insufficient data for comparison.\n")

    print(f"Report generated at {out_md_path}")

if __name__ == "__main__":
    main()
