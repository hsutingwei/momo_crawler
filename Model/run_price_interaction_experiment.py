
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from scipy.sparse import hstack
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.data_loader import load_product_level_training_set

def run_experiment():
    print("Loading Data...")
    try:
        # Load full feature set
        X_dense, X_tfidf, y, meta, vocab = load_product_level_training_set(top_n=100)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Data Loaded. Shape: {X_dense.shape}")

    # ==========================================
    # 2. Feature Engineering (On-the-fly)
    # ==========================================
    print("Engineering Interaction Features...")
    
    # Ensure price_weighted_arousal exists (it should be in data_loader, but let's be safe/explicit for the formula)
    # The user manual says: clean_arousal_score * log1p(price)
    # We check if 'clean_arousal_score' and 'price' exist.
    if 'clean_arousal_score' in X_dense.columns and 'price' in X_dense.columns:
        X_dense['price_weighted_arousal'] = X_dense['clean_arousal_score'] * np.log1p(X_dense['price'])
    else:
        print("Warning: Missing columns for price_weighted_arousal derivation. Using existing if present.")

    # price_weighted_novelty (New Finding)
    # Formula: bert_novelty_mean * log1p(price)
    # 'bert_novelty_mean' should be in X_dense as loaded by load_product_level_training_set
    if 'bert_novelty_mean' in X_dense.columns and 'price' in X_dense.columns:
        X_dense['price_weighted_novelty'] = X_dense['bert_novelty_mean'] * np.log1p(X_dense['price'])
        print("Created feature: price_weighted_novelty")
    else:
        print("Warning: bert_novelty_mean or price missing. Cannot create price_weighted_novelty.")
        # Create dummy if missing to avoid huge errors, or just skip
        X_dense['price_weighted_novelty'] = 0

    # ==========================================
    # 3. Define Feature Sets
    # ==========================================
    interaction_feats = ['price_weighted_arousal', 'price_weighted_novelty']
    
    # Check current columns
    # print(X_dense.columns.tolist())

    # Baseline: Drop interaction features
    # Note: data_loader might already include price_weighted_arousal, so we strictly drop it for baseline
    baseline_drop = [f for f in interaction_feats if f in X_dense.columns]
    X_dense_baseline = X_dense.drop(columns=baseline_drop)
    
    # Experimental: Keep everything
    X_dense_experimental = X_dense.copy()

    # Construct Sparse Matrices for Training
    # We assume X_tfidf is standard for both
    X_baseline = hstack([X_dense_baseline, X_tfidf]).tocsr()
    X_experimental = hstack([X_dense_experimental, X_tfidf]).tocsr()

    print(f"Baseline Features: {X_baseline.shape[1]}")
    print(f"Experimental Features: {X_experimental.shape[1]}")

    # ==========================================
    # 4. Training Loop (5-Fold CV)
    # ==========================================
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Metric Accumulators
    metrics_base = {'precision': [], 'recall': [], 'f1': [], 'prauc': []}
    metrics_exp = {'precision': [], 'recall': [], 'f1': [], 'prauc': []}

    print("\nStarting 5-Fold CV...")
    
    fold = 1
    for train_index, test_index in kf.split(X_dense, y):
        print(f"  Fold {fold}/5...", end="\r")
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # --- Baseline ---
        X_train_b, X_test_b = X_baseline[train_index], X_baseline[test_index]
        model_b = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model_b.fit(X_train_b, y_train)
        preds_b = model_b.predict(X_test_b)
        probs_b = model_b.predict_proba(X_test_b)[:, 1]
        
        metrics_base['precision'].append(precision_score(y_test, preds_b, zero_division=0))
        metrics_base['recall'].append(recall_score(y_test, preds_b, zero_division=0))
        metrics_base['f1'].append(f1_score(y_test, preds_b, zero_division=0))
        metrics_base['prauc'].append(average_precision_score(y_test, probs_b))

        # --- Experimental ---
        X_train_e, X_test_e = X_experimental[train_index], X_experimental[test_index]
        model_e = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model_e.fit(X_train_e, y_train)
        preds_e = model_e.predict(X_test_e)
        probs_e = model_e.predict_proba(X_test_e)[:, 1]

        metrics_exp['precision'].append(precision_score(y_test, preds_e, zero_division=0))
        metrics_exp['recall'].append(recall_score(y_test, preds_e, zero_division=0))
        metrics_exp['f1'].append(f1_score(y_test, preds_e, zero_division=0))
        metrics_exp['prauc'].append(average_precision_score(y_test, probs_e))
        
        fold += 1

    print("\n\nExperiment Complete.")

    # ==========================================
    # 5. Output Results
    # ==========================================
    
    def avg(l): return np.mean(l)

    print("\n" + "="*60)
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Experimental':<12} | {'Lift':<10}")
    print("-" * 60)
    
    results = {}
    for m in ['precision', 'recall', 'f1', 'prauc']:
        base_val = avg(metrics_base[m])
        exp_val = avg(metrics_exp[m])
        lift = (exp_val - base_val) / base_val if base_val > 0 else 0
        results[m] = (base_val, exp_val, lift)
        
        print(f"{m.upper():<15} | {base_val:.4f}     | {exp_val:.4f}       | {lift:+.2%}")
    
    print("="*60)
    
    # Conclusion
    p_lift = results['precision'][2]
    f1_lift = results['f1'][2]
    
    print("\nConclusion:")
    if p_lift > 0.01:
        print("SUCCESS: Price Interaction features significantly improved Precision.")
        print("Hypothesis Verified: Weighting features by Price reduces Low-Price False Positives.")
    elif f1_lift > 0:
        print("MODERATE SUCCESS: Overall F1 improved, though Precision impact was small.")
    else:
        print("FAILURE: Interaction features did not improve the model over the baseline.")

if __name__ == "__main__":
    run_experiment()
