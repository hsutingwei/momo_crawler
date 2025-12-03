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
        "clean_arousal_score", "bert_arousal_mean", "bert_novelty_mean", 
        "bert_repurchase_mean", "bert_negative_mean", "bert_advertisement_mean",
        "intensity_score", "validated_velocity", "is_mature_product",
        "kin_v_1", "kin_v_2", "kin_v_3", "kin_acc_abs", "kin_acc_rel", "kin_jerk_abs",
        "early_bird_momentum", "category_fit_score", "quality_driven_momentum"
    ]
    
    # Calculate mean for each group
    stats = df.groupby("group")[features].mean()
    
    # Reorder index
    desired_order = ["TP", "FN", "FP", "TN"]
    stats = stats.reindex([g for g in desired_order if g in stats.index])
    
    md_content = f"""
## BERT Semantic Feature Analysis

我們已導入 BERT Zero-Shot Classification 取代舊的 Regex 關鍵字。以下分析新特徵的效果：

- **clean_arousal_score**: `Arousal * (1 - Negative) * (1 - Advertisement)`。排除負評與廣告後的純淨驚豔分數。
- **bert_negative_mean**: BERT 判定的負面抱怨機率。
- **bert_advertisement_mean**: BERT 判定的廣告業配機率。
- **bert_novelty_mean**: BERT 判定的新奇感機率。

## Review Kinematics Analysis (New)

引入「評論動力學」特徵，捕捉評論流量的物理變化：

- **Velocity ($v$)**: `kin_v_1` (近7天), `kin_v_2` (前7-14天)。
- **Acceleration ($a$)**: `kin_acc_abs` ($v_1 - v_2$)。正值代表評論量加速增長。
- **Jerk ($j$)**: `kin_jerk_abs` ($v_1 - 2v_2 + v_3$)。加速度的變化率，預期能捕捉爆發瞬間。

### 特徵平均值比較 (Mean Values by Group)

{stats.to_markdown()}

### 深入觀察結論

"""
    
    # Automated insights
    try:
        tp_clean = stats.loc["TP", "clean_arousal_score"] if "TP" in stats.index else 0
        fp_clean = stats.loc["FP", "clean_arousal_score"] if "FP" in stats.index else 0
        
        tp_neg = stats.loc["TP", "bert_negative_mean"] if "TP" in stats.index else 0
        fp_neg = stats.loc["FP", "bert_negative_mean"] if "FP" in stats.index else 0
        
        tp_ad = stats.loc["TP", "bert_advertisement_mean"] if "TP" in stats.index else 0
        fp_ad = stats.loc["FP", "bert_advertisement_mean"] if "FP" in stats.index else 0

        tp_acc = stats.loc["TP", "kin_acc_abs"] if "TP" in stats.index else 0
        fn_acc = stats.loc["FN", "kin_acc_abs"] if "FN" in stats.index else 0
        
        tp_jerk = stats.loc["TP", "kin_jerk_abs"] if "TP" in stats.index else 0
        fn_jerk = stats.loc["FN", "kin_jerk_abs"] if "FN" in stats.index else 0

        if tp_clean > fp_clean:
            md_content += f"- **Clean Arousal 有效**：TP 的 Clean Arousal ({tp_clean:.4f}) 高於 FP ({fp_clean:.4f})，顯示過濾負評與廣告後，驚豔感更能代表真實爆品。\n"
        else:
            md_content += f"- **Clean Arousal 區分力不足**：TP ({tp_clean:.4f}) 與 FP ({fp_clean:.4f}) 差異不大，需檢查 FP 是否為高品質的非爆品。\n"
            
        if fp_neg > tp_neg:
            md_content += f"- **負評過濾驗證**：FP 的負評分數 ({fp_neg:.4f}) 高於 TP ({tp_neg:.4f})，證實部分 FP 來自於負面情緒的高 Arousal。\n"
            
        if fp_ad > tp_ad:
            md_content += f"- **廣告偵測驗證**：FP 的廣告分數 ({fp_ad:.4f}) 高於 TP ({tp_ad:.4f})，證實部分 FP 來自於業配或廣告文案。\n"

        if tp_acc > fn_acc:
            md_content += f"- **加速度 (Acceleration) 有效**：TP 的加速度 ({tp_acc:.4f}) 顯著高於 FN ({fn_acc:.4f})，顯示爆品具有更強的評論增長動能。\n"
        
        if tp_jerk > fn_jerk:
            md_content += f"- **急動度 (Jerk) 有效**：TP 的 Jerk ({tp_jerk:.4f}) 高於 FN ({fn_jerk:.4f})，成功捕捉到爆發瞬間的非線性增長。\n"

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
