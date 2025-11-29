# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_errors(pred_csv_path, output_md_path, experiment_name, compare_csv_path=None, compare_name=None):
    print(f"Analyzing errors for {experiment_name}...")
    
    if not os.path.exists(pred_csv_path):
        print(f"Error: Prediction CSV not found at {pred_csv_path}")
        return

    df = pd.read_csv(pred_csv_path)
    
    # Load comparison data if provided
    df_compare = None
    if compare_csv_path and os.path.exists(compare_csv_path):
        df_compare = pd.read_csv(compare_csv_path)
        # Ensure alignment by product_id
        df = df.merge(df_compare[["product_id", "y_true", "y_pred_best"]], on="product_id", how="left", suffixes=("", f"_{compare_name}"))
        # Rename y_true from merge to avoid confusion if needed, but they should be identical if same dataset
        # Actually, y_true might be different if label definitions differ!
        # hybrid_strict (ratio>=0.3) vs binary_explosive (ratio>=1.0) -> y_true WILL be different.
        # So we should keep both y_true and y_true_explosive
        
    
    # Basic Metrics
    y_true = df["y_true"]
    y_pred = df["y_pred_best"]
    
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)
    
    cats = np.full(len(df), "TN", dtype=object)
    cats[tp_mask] = "TP"
    cats[fp_mask] = "FP"
    cats[fn_mask] = "FN"
    df["category"] = cats
    
    # Generate Report
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(f"# {experiment_name} Error Analysis\n\n")
        
        f.write("## 檔案說明\n")
        f.write(f"- 本報告針對實驗 `{experiment_name}`（delta>=10, ratio>=0.3 的成長任務），分析 TP/FP/FN/TN 在各種特徵上的分布差異。\n")
        if compare_name:
            f.write(f"- 特別包含與 `{compare_name}`（爆品任務）的交集比較。\n")
        f.write("\n")
        
        f.write(f"**Total Samples:** {len(df)}\n")
        if "best_threshold_used" in df.columns:
            f.write(f"**Best Threshold Used:** {df['best_threshold_used'].iloc[0]:.4f}\n\n")
            
        # 1. Category Distribution
        f.write("## 1. Category Distribution\n\n")
        counts = df["category"].value_counts()
        total = len(df)
        f.write("| Category | Count | Percentage |\n")
        f.write("| --- | --- | --- |\n")
        for cat in ["TP", "FP", "FN", "TN"]:
            c = counts.get(cat, 0)
            p = c / total * 100
            f.write(f"| {cat} | {c} | {p:.2f}% |\n")
        f.write("\n")
        
        # 2. Feature Comparison
        f.write("## 2. Feature Comparison (Mean / Median)\n\n")
        features = [
            "price", "comment_count_pre", "score_mean", "like_count_sum",
            "had_any_change_pre", "num_increases_pre",
            "has_image_urls", "has_video_url", "has_reply_content",
            "comment_count_7d", "comment_count_30d", "comment_count_90d",
            "days_since_last_comment", "comment_7d_ratio",
            "sentiment_mean_recent", "neg_ratio_recent", 
            "promo_ratio_recent", "repurchase_ratio_recent",
            "max_raw_delta", "max_raw_ratio"
        ]
        
        for feat in features:
            if feat not in df.columns:
                continue
                
            f.write(f"### {feat}\n\n")
            f.write("| Category | Mean | Median | Std |\n")
            f.write("| --- | --- | --- | --- |\n")
            for cat in ["TP", "FP", "FN", "TN"]:
                sub = df[df["category"] == cat]
                if len(sub) > 0:
                    mean_val = sub[feat].mean()
                    med_val = sub[feat].median()
                    std_val = sub[feat].std()
                    f.write(f"| {cat} | {mean_val:.4f} | {med_val:.4f} | {std_val:.4f} |\n")
                else:
                    f.write(f"| {cat} | - | - | - |\n")
            f.write("\n")

        # 3. Intersection Analysis
        if compare_name and f"y_true_{compare_name}" in df.columns:
            f.write(f"## 3. 與 {compare_name} 的交集分析\n\n")
            
            y_strict = df["y_true"]
            y_explosive = df[f"y_true_{compare_name}"]
            pred_strict = df["y_pred_best"]
            pred_explosive = df[f"y_pred_best_{compare_name}"]
            
            # 3.1 任務定義差異 (Label Overlap)
            f.write("### 3.1 真實標籤 (Ground Truth) 重疊度\n")
            f.write("- **Strict (Ratio>=0.3)**: 包含較廣泛的成長商品。\n")
            f.write("- **Explosive (Ratio>=1.0)**: 嚴格的爆品子集。\n\n")
            
            overlap_matrix = pd.crosstab(y_strict, y_explosive, rownames=['Strict'], colnames=['Explosive'])
            f.write("#### Label Confusion Matrix (Strict vs Explosive)\n")
            f.write(overlap_matrix.to_markdown() + "\n\n")
            
            # 3.2 預測重疊度 (Prediction Overlap)
            f.write("### 3.2 模型預測 (Prediction) 重疊度\n")
            pred_overlap = pd.crosstab(pred_strict, pred_explosive, rownames=['Pred_Strict'], colnames=['Pred_Explosive'])
            f.write(pred_overlap.to_markdown() + "\n\n")
            
            # 3.3 針對「真爆品 (True Explosive, y_exp=1)」的捕捉能力比較
            f.write("### 3.3 針對「真爆品 (y_explosive=1)」的捕捉能力\n")
            mask_true_exp = (y_explosive == 1)
            df_exp = df[mask_true_exp]
            
            caught_by_strict = df_exp[df_exp["y_pred_best"] == 1]
            caught_by_exp = df_exp[df_exp[f"y_pred_best_{compare_name}"] == 1]
            
            f.write(f"- **Total True Explosives**: {len(df_exp)}\n")
            f.write(f"- **Caught by Strict**: {len(caught_by_strict)} ({len(caught_by_strict)/len(df_exp):.2%})\n")
            f.write(f"- **Caught by Explosive**: {len(caught_by_exp)} ({len(caught_by_exp)/len(df_exp):.2%})\n\n")
            
            # Analyze those caught by strict but missed by explosive (if any)
            strict_wins = df_exp[(df_exp["y_pred_best"] == 1) & (df_exp[f"y_pred_best_{compare_name}"] == 0)]
            if len(strict_wins) > 0:
                f.write("#### Strict 抓到但 Explosive 漏掉的爆品 (Strict Wins)\n")
                f.write(f"Count: {len(strict_wins)}\n")
                f.write("特徵平均值：\n")
                        # Filter features to only those present in strict_wins
            available_features = [f for f in features if f in strict_wins.columns]
            if available_features:
                f.write(strict_wins[available_features].mean().to_frame().T.to_markdown() + "\n\n")
            else:
                f.write("No feature columns available for comparison.\n\n")
            
            # Analyze those caught by explosive but missed by strict (if any)
            exp_wins = df_exp[(df_exp["y_pred_best"] == 0) & (df_exp[f"y_pred_best_{compare_name}"] == 1)]
            if len(exp_wins) > 0:
                f.write("#### Explosive 抓到但 Strict 漏掉的爆品 (Explosive Wins)\n")
                f.write(f"Count: {len(exp_wins)}\n")
                f.write("特徵平均值：\n")
                        # Filter features to only those present in exp_wins
            available_features = [f for f in features if f in exp_wins.columns]
            if available_features:
                f.write(exp_wins[available_features].mean().to_frame().T.to_markdown() + "\n\n")
            else:
                f.write("No feature columns available for comparison.\n\n")

    print(f"Analysis saved to {output_md_path}")

def main():
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model", "outputs_label_experiments")
    
    # Analyze hybrid_strict_balanced
    analyze_errors(
        pred_csv_path=os.path.join(output_dir, "preds_hybrid_strict_balanced.csv"),
        output_md_path=os.path.join(output_dir, "hybrid_strict_balanced_error_analysis.md"),
        experiment_name="hybrid_strict_balanced",
        compare_csv_path=os.path.join(output_dir, "preds_binary_explosive.csv"),
        compare_name="binary_explosive"
    )

if __name__ == "__main__":
    main()
