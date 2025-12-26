import pandas as pd
import glob
import os
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

def analyze_predictions(folder):
    files = glob.glob(os.path.join(folder, "*predictions.csv"))
    if not files:
        print(f"No predictions found in {folder}")
        return

    df = pd.read_csv(files[0])
    y_true = df["y_true"]
    y_score = df["y_score"]
    
    # 1. Check Stats at default 0.5
    y_pred_05 = (y_score >= 0.5).astype(int)
    f1_05 = f1_score(y_true, y_pred_05)
    print(f"--- {folder} ---")
    print(f"Default (0.5) F1: {f1_05:.4f}")
    print(f"Mean Score: {y_score.mean():.4f}")
    print(f"Max Score: {y_score.max():.4f}")
    
    # 2. Find Best Threshold
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s)
    best_f1 = f1s[best_idx]
    best_th = thresholds[best_idx]
    
    print(f"Best F1: {best_f1:.4f} at Threshold: {best_th:.4f}")
    print("-" * 20)

def main():
    analyze_predictions("Model/outputs/exp_baseline_hybrid")
    analyze_predictions("Model/outputs/exp_baseline_hybrid_fix")
    analyze_predictions("Model/outputs/exp_v2_filter_hybrid")

if __name__ == "__main__":
    main()
