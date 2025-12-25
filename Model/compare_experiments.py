import pandas as pd
import glob
import os

def load_summary(path):
    files = glob.glob(os.path.join(path, "*ALL_summary.csv"))
    if not files:
        return None
    return pd.read_csv(files[0])

def main():
    base_path = "Model/outputs"
    exp1 = "exp_baseline_no_f"
    exp2 = "exp_v2_filter"
    
    df1 = load_summary(os.path.join(base_path, exp1))
    df2 = load_summary(os.path.join(base_path, exp2))
    
    if df1 is None or df2 is None:
        print("Error: Could not find summary CSVs.")
        return

    # Select best run (e.g. SVM + lgbm_fs is usually good, or just take average)
    # Let's take the row with highest f1_1 for fairness
    row1 = df1.sort_values("f1_1_mean", ascending=False).iloc[0]
    row2 = df2.sort_values("f1_1_mean", ascending=False).iloc[0]
    
    print(f"{'Metric':<25} | {'Baseline':<12} | {'v2 Filter':<12} | {'Diff'}")
    print("-" * 75)
    
    metrics = {
        'auc_mean': 'AUC',
        'f1_1_mean': 'F1 (Class 1)',
        'precision_1_mean': 'Precision (1)',
        'recall_1_mean': 'Recall (1)',
        'accuracy_mean': 'Accuracy'
    }
    
    for col, label in metrics.items():
        v1 = row1.get(col, 0)
        v2 = row2.get(col, 0)
        diff = v2 - v1
        print(f"{label:<25} | {v1:.4f}{'':<6} | {v2:.4f}{'':<6} | {diff:+.4f}")

    print("\n[Details]")
    print(f"Baseline Algo: {row1['algorithm']} + {row1['fs_method']}")
    print(f"v2 Filter Algo: {row2['algorithm']} + {row2['fs_method']}")

if __name__ == "__main__":
    main()
