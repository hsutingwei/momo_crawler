import pandas as pd
import os

def generate_temporal_analysis(csv_path, output_md_path, experiment_name):
    print(f"Processing {experiment_name}...")
    df = pd.read_csv(csv_path)
    
    # Define groups
    # TP: y_true=1, y_pred_best=1
    # FN: y_true=1, y_pred_best=0
    # FP: y_true=0, y_pred_best=1
    # TN: y_true=0, y_pred_best=0
    
    df["group"] = "TN"
    df.loc[(df["y_true"]==1) & (df["y_pred_best"]==1), "group"] = "TP"
    df.loc[(df["y_true"]==1) & (df["y_pred_best"]==0), "group"] = "FN"
    df.loc[(df["y_true"]==0) & (df["y_pred_best"]==1), "group"] = "FP"
    
    features = [
        "comment_1st_30d", "comment_2nd_30d", "comment_3rd_30d", 
        "ratio_recent30_to_prev60"
    ]
    
    # Calculate mean for each group
    stats = df.groupby("group")[features].mean()
    
    # Reorder index if possible
    desired_order = ["TP", "FN", "FP", "TN"]
    stats = stats.reindex([g for g in desired_order if g in stats.index])
    
    md_content = f"""
## Temporal Trend Features (90-day window)

在此次實驗中，我們加入了 90 天內的三段式評論數統計與加速比率特徵，以捕捉爆發前的趨勢：

- **comment_3rd_30d**: 最近 30 天 (0-30 days before cutoff)
- **comment_2nd_30d**: 中間 30 天 (31-60 days before cutoff)
- **comment_1st_30d**: 最早 30 天 (61-90 days before cutoff)
- **ratio_recent30_to_prev60**: 加速比率 = `comment_3rd_30d / (comment_1st_30d + comment_2nd_30d + 1e-6)`

### 特徵平均值比較 (Mean Values by Group)

{stats.to_markdown()}

### 初步觀察結論

"""
    
    # Add some automated insights
    try:
        tp_ratio = stats.loc["TP", "ratio_recent30_to_prev60"] if "TP" in stats.index else 0
        tn_ratio = stats.loc["TN", "ratio_recent30_to_prev60"] if "TN" in stats.index else 0
        fn_ratio = stats.loc["FN", "ratio_recent30_to_prev60"] if "FN" in stats.index else 0
        
        if tp_ratio > tn_ratio * 1.5:
            md_content += f"- **正向訊號明顯**：TP 的加速比率 ({tp_ratio:.2f}) 顯著高於 TN ({tn_ratio:.2f})，顯示爆品在 cutoff 前確實有加速跡象。\n"
        elif tp_ratio > tn_ratio:
            md_content += f"- **正向訊號微弱**：TP 的加速比率 ({tp_ratio:.2f}) 略高於 TN ({tn_ratio:.2f})，區別度可能不足。\n"
        else:
            md_content += f"- **無明顯加速特徵**：TP 的加速比率 ({tp_ratio:.2f}) 並未高於 TN ({tn_ratio:.2f})。\n"
            
        if fn_ratio < tp_ratio:
            md_content += f"- **漏抓原因**：FN 的加速比率 ({fn_ratio:.2f}) 低於 TP，可能因為這些爆品是「突然爆發」或「穩定成長型」，在 cutoff 前尚未展現明顯加速。\n"
            
    except Exception as e:
        md_content += f"- (無法自動生成結論: {e})\n"

    # Append to file
    with open(output_md_path, "a", encoding="utf-8") as f:
        f.write("\n" + md_content)
    print(f"Updated {output_md_path}")

def main():
    base_dir = r"c:\YvesProject\中央\線上評論\momo_crawler-main\Model\outputs_label_experiments"
    
    # 1. Hybrid Strict Balanced
    generate_temporal_analysis(
        os.path.join(base_dir, "preds_hybrid_strict_balanced.csv"),
        os.path.join(base_dir, "hybrid_strict_balanced_error_analysis.md"),
        "hybrid_strict_balanced"
    )
    
    # 2. Binary Explosive
    generate_temporal_analysis(
        os.path.join(base_dir, "preds_binary_explosive.csv"),
        os.path.join(base_dir, "binary_explosive_error_analysis.md"),
        "binary_explosive"
    )

if __name__ == "__main__":
    main()
