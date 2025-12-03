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
    
    # Get Train Settings
    train_settings = exp_config.get("train_settings", {})
    use_class_weight = train_settings.get("use_class_weight", False)
    do_threshold_search = train_settings.get("do_threshold_search", False)
    
    # Backward compatibility for binary_explosive if train_settings not present but params has it
    if "use_pos_weight" in exp_config["params"]:
         use_class_weight = exp_config["params"]["use_pos_weight"]

    print(f"Settings: use_class_weight={use_class_weight}, do_threshold_search={do_threshold_search}")

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
    error_analysis_rows = [] # Collection for error analysis
    
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

    if not y.index.is_unique:
        print("[WARNING] y index is not unique! Resetting index.")
        y = y.reset_index(drop=True)
        y_encoded = y_encoded.reset_index(drop=True)
        # Also reset X_all index? X_all is sparse matrix, indexed by integer 0..N.
        # If y index was not 0..N, splitting might be weird if using loc?
        # skf.split uses indices (integers), so it slices X_all by row number.
        # y_encoded.iloc[tr_idx] uses integer position.
        # So y index shouldn't matter for iloc.
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y_encoded)), y_encoded)):
        X_train, X_val = X_all[tr_idx], X_all[va_idx]
        y_train, y_val = y_encoded.iloc[tr_idx], y_encoded.iloc[va_idx]
        
        # Use XGBoost with fixed random_state
        scale_pos_weight = None
        if not is_multiclass and use_class_weight:
             # Calculate scale_pos_weight = n_neg / n_pos
             # Note: n_pos/n_neg are total counts, but for CV we should ideally use training set counts.
             # However, since we use StratifiedKFold, the ratio is preserved.
             # We can approximate using the fold's y_train.
             n_neg_tr = (y_train == 0).sum()
             n_pos_tr = (y_train == 1).sum()
             if n_pos_tr > 0:
                 scale_pos_weight = n_neg_tr / n_pos_tr
                 print(f"  [Fold {fold}] Applied scale_pos_weight: {scale_pos_weight:.4f}")

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
                use_label_encoder=False,
                scale_pos_weight=scale_pos_weight
            )
        
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"[ERROR Fold {fold}] model.fit failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Predict
        y_pred = model.predict(X_val)
        
        # Convert y_val to numpy to avoid pandas alignment issues
        y_val_np = y_val.values
        
        if is_multiclass:
            # Multiclass Metrics
            m = {
                "accuracy": accuracy_score(y_val_np, y_pred),
                "f1_macro": f1_score(y_val_np, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_val_np, y_pred, average="weighted", zero_division=0),
                # Class 3 (Explosive) Metrics if exists
                "f1_class3": f1_score(y_val_np, y_pred, labels=[explosive_class_idx], average=None, zero_division=0)[0] if explosive_class_idx is not None else 0.0,
                "precision_class3": precision_score(y_val_np, y_pred, labels=[explosive_class_idx], average=None, zero_division=0)[0] if explosive_class_idx is not None else 0.0,
                "recall_class3": recall_score(y_val_np, y_pred, labels=[explosive_class_idx], average=None, zero_division=0)[0] if explosive_class_idx is not None else 0.0
            }
        else:
            # Binary Metrics
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # 1. Default Threshold (0.5) Metrics
            cm = confusion_matrix(y_val_np, y_pred)
            print(f"  [Fold {fold}] Confusion Matrix (Th=0.5): TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
            
            m = {
                "accuracy": accuracy_score(y_val_np, y_pred),
                "precision": precision_score(y_val_np, y_pred, zero_division=0),
                "recall": recall_score(y_val_np, y_pred, zero_division=0),
                "f1": f1_score(y_val_np, y_pred, zero_division=0),
                "pr_auc": average_precision_score(y_val_np, y_prob),
                "roc_auc": roc_auc_score(y_val_np, y_prob)
            }
            
            # 2. Threshold Scanning (if enabled)
            best_metrics = {}
            if do_threshold_search:
                thresholds = np.linspace(0.05, 0.95, 19)
                best_th = 0.5
                best_f1 = -1.0
                
                for th in thresholds:
                    y_pred_th = (y_prob >= th).astype(int)
                    f1_th = f1_score(y_val_np, y_pred_th, zero_division=0)
                    if f1_th > best_f1:
                        best_f1 = f1_th
                        best_th = th
                        best_metrics = {
                            "f1_best_th": f1_th,
                            "precision_best_th": precision_score(y_val_np, y_pred_th, zero_division=0),
                            "recall_best_th": recall_score(y_val_np, y_pred_th, zero_division=0),
                            "best_threshold": best_th
                        }
                # Merge best metrics
                m.update(best_metrics)
            
            # Collect data for error analysis if threshold search is enabled (implies we care about detailed analysis)
            # OR if it's explicitly one of our target experiments
            if do_threshold_search:
                try:
                    # Get validation meta and dense features
                    meta_val = meta.iloc[va_idx].reset_index(drop=True)
                    X_dense_val = X_dense_df.iloc[va_idx].reset_index(drop=True)
                    
                    # Create a DataFrame for this fold's predictions
                    fold_df = pd.DataFrame({
                        "product_id": meta_val["product_id"],
                        "y_true": y_val.values,
                        "y_prob": y_prob
                    })
                    
                    # Add dense features
                    cols_to_add = [
                        "price", "comment_count_pre", "score_mean", "like_count_sum", 
                        "had_any_change_pre", "num_increases_pre", 
                        "has_image_urls", "has_video_url", "has_reply_content",
                        "comment_count_7d", "comment_count_30d", "comment_count_90d", "days_since_last_comment", "comment_7d_ratio",
                        "comment_1st_30d", "comment_2nd_30d", "comment_3rd_30d", "ratio_recent30_to_prev60",
                        "sentiment_mean_recent", "neg_ratio_recent", "promo_ratio_recent", "repurchase_ratio_recent",
                        "arousal_ratio", "novelty_ratio", "intensity_score", "clean_arousal_score",
                        "bert_arousal_mean", "bert_novelty_mean", "bert_repurchase_mean", "bert_negative_mean", "bert_advertisement_mean",
                        "kin_v_1", "kin_v_2", "kin_v_3", "kin_acc_abs", "kin_acc_rel", "kin_jerk_abs",
                        "validated_velocity", "price_weighted_arousal", "novelty_momentum", "is_mature_product"
                    ]
                    
                    # Also add max_raw_delta / max_raw_ratio if available in meta
                    if "max_raw_delta" in meta_val.columns:
                        fold_df["max_raw_delta"] = meta_val["max_raw_delta"]
                    if "max_raw_ratio" in meta_val.columns:
                        fold_df["max_raw_ratio"] = meta_val["max_raw_ratio"]

                    # Filter cols that exist in X_dense_val
                    existing_cols = [c for c in cols_to_add if c in X_dense_val.columns]
                    
                    # Use concat to add columns, handling potential duplicates by dropping them first from X_dense_val selection if needed
                    # But X_dense_val should be clean now.
                    # We align on index (both are 0..N)
                    fold_df = pd.concat([fold_df, X_dense_val[existing_cols]], axis=1)
                    
                    error_analysis_rows.append(fold_df)
                except Exception as e:
                    print(f"[ERROR Fold {fold}] Error analysis collection failed: {e}")
                    import traceback
                    traceback.print_exc()

        metrics_list.append(m)
    
    # Save Prediction CSV if applicable (for experiments with threshold search)
    if do_threshold_search and error_analysis_rows:
        full_error_df = pd.concat(error_analysis_rows, ignore_index=True)
        
        # Calculate best threshold on the full dataset
        thresholds = np.linspace(0.05, 0.95, 19)
        best_th_full = 0.5
        best_f1_full = -1.0
        
        for th in thresholds:
            preds = (full_error_df["y_prob"] >= th).astype(int)
            f1 = f1_score(full_error_df["y_true"], preds, zero_division=0)
            if f1 > best_f1_full:
                best_f1_full = f1
                best_th_full = th
        
        full_error_df["y_pred_best"] = (full_error_df["y_prob"] >= best_th_full).astype(int)
        full_error_df["best_threshold_used"] = best_th_full
        
        # Save CSV with experiment name
        out_csv = os.path.join(output_dir, f"preds_{exp_config['name']}.csv")
        full_error_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Saved predictions to {out_csv} (Best Th: {best_th_full:.4f})")
        
        # Also save as binary_explosive_error_analysis.csv for backward compatibility if needed
        if exp_config['name'] == 'binary_explosive':
             legacy_csv = os.path.join(output_dir, "binary_explosive_error_analysis.csv")
             full_error_df.to_csv(legacy_csv, index=False, encoding="utf-8-sig")

    # Average Metrics
    avg_metrics = {k: float(np.mean([m.get(k, np.nan) for m in metrics_list])) for k in metrics_list[0].keys()}
    
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
        if not np.isnan(v):
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
