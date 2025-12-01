import pandas as pd
import os

def generate_arousal_analysis(csv_path, output_md_path, experiment_name):
    print(f"Processing {experiment_name}...")
    df = pd.read_csv(csv_path)
    
    # Define groups
    df["group"] = "TN"
    df.loc[(df["y_true"]==1) & (df["y_pred_best"]==1), "group"] = "TP"
    df.loc[(df["y_true"]==1) & (df["y_pred_best"]==0), "group"] = "FN"
    df.loc[(df["y_true"]==0) & (df["y_pred_best"]==1), "group"] = "FP"
    
    features = [
        "arousal_ratio", "novelty_ratio", "intensity_score", "repurchase_ratio_recent"
    ]
    
    # Calculate mean for each group
    stats = df.groupby("group")[features].mean()
    
    # Reorder index
    desired_order = ["TP", "FN", "FP", "TN"]
    stats = stats.reindex([g for g in desired_order if g in stats.index])
    
    md_content = f"""
## Arousal & Novelty Analysis (Emotional Intensity)

為了區分「穩定熱銷 (Sustained Hits)」與「爆發新品 (Viral Hits)」，我們引入了情緒強度特徵：

- **arousal_ratio**: 驚嘆、誇張、強烈情緒關鍵字比例 (e.g. 驚豔, 太神, 扯)。
- **novelty_ratio**: 初次體驗、觀望很久關鍵字比例 (e.g. 第一次買, 終於下手)。
- **intensity_score**: 複合指標 = `(arousal + novelty) / (repurchase + 0.1)`。設計用來獎勵高情緒/新奇，懲罰無聊的回購。

### 特徵平均值比較 (Mean Values by Group)

{stats.to_markdown()}

### 初步觀察結論

"""
    
    # Automated insights
    try:
        tp_intensity = stats.loc["TP", "intensity_score"] if "TP" in stats.index else 0
        tn_intensity = stats.loc["TN", "intensity_score"] if "TN" in stats.index else 0
        fn_intensity = stats.loc["FN", "intensity_score"] if "FN" in stats.index else 0
        fp_intensity = stats.loc["FP", "intensity_score"] if "FP" in stats.index else 0
        
        if tp_intensity > tn_intensity:
            md_content += f"- **強度區分有效**：TP 的 Intensity Score ({tp_intensity:.4f}) 高於 TN ({tn_intensity:.4f})，顯示爆品通常帶有較強烈的情緒或新奇感。\n"
        else:
            md_content += f"- **強度區分不明顯**：TP ({tp_intensity:.4f}) 與 TN ({tn_intensity:.4f}) 差異不大。\n"
            
        if fn_intensity < tp_intensity:
            md_content += f"- **漏抓特徵**：FN 的 Intensity Score ({fn_intensity:.4f}) 低於 TP，可能因為這些被漏掉的爆品屬於「低調熱賣」或「回購型」，導致 Intensity 被 repurchase 分母稀釋。\n"
            
        if fp_intensity > tp_intensity:
            md_content += f"- **誤判來源**：FP 的 Intensity Score ({fp_intensity:.4f}) 甚至高於 TP，顯示有些非爆品雖然情緒激動 (Arousal 高)，但可能缺乏其他支撐 (如銷量轉化)，導致模型誤判。\n"

    except Exception as e:
        md_content += f"- (無法自動生成結論: {e})\n"

    # Append to file
    with open(output_md_path, "a", encoding="utf-8") as f:
        f.write("\n" + md_content)
    print(f"Updated {output_md_path}")

def main():
    base_dir = r"c:\YvesProject\中央\線上評論\momo_crawler-main\Model\outputs_label_experiments"
    
    # 1. Hybrid Strict Balanced
    generate_arousal_analysis(
        os.path.join(base_dir, "preds_hybrid_strict_balanced.csv"),
        os.path.join(base_dir, "hybrid_strict_balanced_error_analysis.md"),
        "hybrid_strict_balanced"
    )
    
    # 2. Binary Explosive
    generate_arousal_analysis(
        os.path.join(base_dir, "preds_binary_explosive.csv"),
        os.path.join(base_dir, "binary_explosive_error_analysis.md"),
        "binary_explosive"
    )

if __name__ == "__main__":
    main()
