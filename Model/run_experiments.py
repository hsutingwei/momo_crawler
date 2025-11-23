# -*- coding: utf-8 -*-
import os
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    average_precision_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import hstack

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.data_loader import load_product_level_training_set
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

def std_scaler_to_sparse(Xdf: pd.DataFrame) -> csr_matrix:
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(Xdf.values)
    return csr_matrix(X_scaled)

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_experiment(exp_config, output_dir):
    print(f"\n=== Running Experiment: {exp_config['name']} ===")
    print(f"Strategy: {exp_config['strategy']}")
    print(f"Params: {exp_config['params']}")
    
    # 1. Load Data
    # Use fixed parameters for other args to ensure fair comparison
    # Assuming these match the current production settings or reasonable defaults
    X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(
        date_cutoff="2025-06-25",
        top_n=100,
        vocab_mode="global",
        label_delta_threshold=exp_config['params'].get('delta_threshold', 10),
        label_strategy=exp_config['strategy'],
        label_params=exp_config['params']
    )
    
    # 2. Check Distribution
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    pos_rate = n_pos / len(y) if len(y) > 0 else 0.0
    print(f"Data Loaded: {len(y)} samples")
    print(f"Class Distribution: Pos={n_pos}, Neg={n_neg}, Rate={pos_rate:.4f}")
    
    if n_pos < 10:
        print("WARNING: Too few positive samples. Skipping training.")
        return {
            "experiment": exp_config['name'],
            "n_samples": len(y),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "pos_rate": pos_rate,
            "status": "skipped_low_pos"
        }

    # 3. Prepare Features
    X_dense = std_scaler_to_sparse(X_dense_df)
    X_all = hstack([X_dense, X_tfidf], format="csr")
    
    # 4. Train & Evaluate (5-Fold CV with fixed random_state)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics_list = []
    
    # Detect Multiclass
    num_classes = y.nunique()
    is_multiclass = num_classes > 2
    
    y_encoded = y
    le = None
    explosive_class_idx = None
    
    if is_multiclass:
        le = LabelEncoder()
        y_encoded = pd.Series(le.fit_transform(y), index=y.index)
        num_classes = len(le.classes_)
        print(f"Multiclass detected: {num_classes} classes")
        print(f"Classes mapping: {dict(zip(le.classes_, range(num_classes)))}")
        print("Class Distribution:\n", y.value_counts().sort_index())
        
        if 3 in le.classes_:
            explosive_class_idx = le.transform([3])[0]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_encoded)), y_encoded)):
        X_train, X_val = X_all[tr_idx], X_all[va_idx]
        y_train, y_val = y_encoded.iloc[tr_idx], y_encoded.iloc[va_idx]
        
        # Use XGBoost with fixed random_state
        if is_multiclass:
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                objective="multi:softprob",
                num_class=num_classes,
                eval_metric="mlogloss",
                use_label_encoder=False
            )
        else:
            model = XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                use_label_encoder=False
            )
        
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_val)
        
        if is_multiclass:
            # Multiclass Metrics
            m = {
                "accuracy": accuracy_score(y_val, y_pred),
                "f1_macro": f1_score(y_val, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_val, y_pred, average="weighted", zero_division=0),
                # Class 3 (Explosive) Metrics if exists
                "f1_class3": f1_score(y_val, y_pred, labels=[explosive_class_idx], average=None, zero_division=0)[0] if explosive_class_idx is not None else 0.0,
                "precision_class3": precision_score(y_val, y_pred, labels=[explosive_class_idx], average=None, zero_division=0)[0] if explosive_class_idx is not None else 0.0,
                "recall_class3": recall_score(y_val, y_pred, labels=[explosive_class_idx], average=None, zero_division=0)[0] if explosive_class_idx is not None else 0.0
            }
        else:
            # Binary Metrics
            y_prob = model.predict_proba(X_val)[:, 1]
            m = {
                "accuracy": accuracy_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred, zero_division=0),
                "recall": recall_score(y_val, y_pred, zero_division=0),
                "f1": f1_score(y_val, y_pred, zero_division=0),
                "pr_auc": average_precision_score(y_val, y_prob),
                "roc_auc": roc_auc_score(y_val, y_prob)
            }
        metrics_list.append(m)
    
    # Average Metrics
    avg_metrics = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0].keys()}
    
    result = {
        "experiment": exp_config['name'],
        "n_samples": len(y),
        "n_pos": n_pos if not is_multiclass else int((y==3).sum()), # For multiclass, treat Class 3 as "Pos" for summary
        "n_neg": n_neg if not is_multiclass else int((y!=3).sum()),
        "pos_rate": pos_rate if not is_multiclass else (y==3).mean(),
        "status": "completed",
        **avg_metrics
    }
    
    print("Average Metrics:")
    for k, v in avg_metrics.items():
        print(f"  {k}: {v:.4f}")
        
    return result

def main():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "label_experiments.yaml")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Model", "outputs_label_experiments")
    os.makedirs(output_dir, exist_ok=True)
    
    config = load_config(config_path)
    results = []
    
    for exp in config['experiments']:
        try:
            res = run_experiment(exp, output_dir)
            results.append(res)
        except Exception as e:
            print(f"Error running experiment {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "experiment": exp['name'],
                "status": "failed",
                "error": str(e)
            })

    # Save Results
    df_res = pd.DataFrame(results)
    
    # CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Markdown
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Label Experiment Results\n\n")
        # Manual markdown table
        cols = df_res.columns.tolist()
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in df_res.iterrows():
            row_str = [str(row[c]) for c in cols]
            f.write("| " + " | ".join(row_str) + " |\n")
    print(f"Results saved to {md_path}")

if __name__ == "__main__":
    main()
