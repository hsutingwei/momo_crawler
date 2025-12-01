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
        "arousal_ratio", "novelty_ratio", "intensity_score", "repurchase_ratio_recent",
        "validated_velocity", "price_weighted_arousal", "novelty_momentum", "is_mature_product"
    ]
    
    # Calculate mean for each group
    stats = df.groupby("group")[features].mean()
    
    # Reorder index
    desired_order = ["TP", "FN", "FP", "TN"]
    stats = stats.reindex([g for g in desired_order if g in stats.index])
    
    md_content = f"""
## Interaction Features Analysis (Cross Features)

為了進一步區分雜訊與真實爆發，我們引入了交互特徵：

- **validated_velocity**: `Acceleration * log1p(Volume)`。確認高成長是否由足夠的評論量支撐。
- **price_weighted_arousal**: `Arousal * log1p(Price)`。區分「廉價炒作 (FP)」與「高價驚豔 (TP)」。
- **novelty_momentum**: `Acceleration * (1 - Repurchase)`。確認成長動力來自「新客」而非「回購」。
- **is_mature_product**: 標記是否為成熟商品 (高累積評論或高回購)。

### 特徵平均值比較 (Mean Values by Group)

{stats.to_markdown()}

### 深入觀察結論

"""
    
    # Automated insights
    try:
        tp_vv = stats.loc["TP", "validated_velocity"] if "TP" in stats.index else 0
        fp_vv = stats.loc["FP", "validated_velocity"] if "FP" in stats.index else 0
        
        tp_pwa = stats.loc["TP", "price_weighted_arousal"] if "TP" in stats.index else 0
        fp_pwa = stats.loc["FP", "price_weighted_arousal"] if "FP" in stats.index else 0
        
        tp_nm = stats.loc["TP", "novelty_momentum"] if "TP" in stats.index else 0
        fn_nm = stats.loc["FN", "novelty_momentum"] if "FN" in stats.index else 0

        if tp_vv > fp_vv:
            md_content += f"- **速度驗證有效**：TP 的 Validated Velocity ({tp_vv:.4f}) 高於 FP ({fp_vv:.4f})，顯示真爆品通常伴隨更紮實的評論量成長。\n"
        
        if tp_pwa > fp_pwa:
            md_content += f"- **價格過濾有效**：TP 的 Price Weighted Arousal ({tp_pwa:.4f}) 高於 FP ({fp_pwa:.4f})，成功區分出高價值的驚豔商品，壓制了廉價品的雜訊。\n"
        else:
            md_content += f"- **價格過濾不明顯**：TP ({tp_pwa:.4f}) 與 FP ({fp_pwa:.4f}) 差異不大，可能因為部分爆品本身也是低價品。\n"
            
        if tp_nm > fn_nm:
            md_content += f"- **新客動力區分**：TP 的 Novelty Momentum ({tp_nm:.4f}) 顯著高於 FN ({fn_nm:.4f})，證實了爆品成長主要來自新客，而 FN 多為回購驅動。\n"

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
