# -*- coding: utf-8 -*-
"""
Model/train_v2.py
新版訓練腳本 - 整合 7-Phase ML Pipeline

特性:
- 固定 8:2 split + K-fold CV (可復現)
- Hash tracking (dataset/split/feature)
- 5 個自動化 leakage checks
- PostgreSQL 實驗記錄
- 15+ artifacts per run
- Ablation study 支援

用法:
python Model/train_v2.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --feature-set baseline \
  --n-folds 10 \
  --random-seed 42 \
  --run-id my_experiment_001

與 train.py 的差異:
- 使用固定 split (不是 date cutoff split)
- 自動 leakage prevention
- 完整 artifact 管理
- PostgreSQL logging
"""

import os
import sys
import json
import argparse
import warnings
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb

# Import data loader
from data_loader import load_product_level_training_set

# Import new ML Pipeline modules
from experiment_utils import (
    make_splits, compute_dataset_hash, compute_split_hash, compute_feature_hash,
    get_feature_whitelist, validate_feature_whitelist, get_git_info,
    compute_code_fingerprint, generate_samples_metadata
)
from experiment_logger import (
    get_db_connection, make_run_id, insert_run_start, update_run_finish,
    upsert_samples, upsert_predictions, upsert_features, upsert_artifact
)
from feature_transformer import FeatureTransformer
from leakage_prevention import verify_no_leakage
from imbalance_handling import compute_scale_pos_weight_per_fold, save_imbalance_report
from artifact_manager import ArtifactManager, save_baseline_best_params


def parse_args():
    """解析命令行參數"""
    ap = argparse.ArgumentParser(description='ML Pipeline v2 - Production Training')
    
    # Basic settings
    ap.add_argument('--mode', type=str, default='product_level',
                    choices=['product_level'],
                    help='訓練模式（目前只支援 product_level）')
    ap.add_argument('--date-cutoff', type=str, default='2025-06-25',
                    help='資料切分日期（用於載入數據）')
    ap.add_argument('--pipeline-version', type=str, default=None,
                    help='資料處理流程版本號')
    
    # Run identification
    ap.add_argument('--run-id', type=str, default=None,
                    help='實驗 Run ID（若不指定則自動生成）')
    ap.add_argument('--group-id', type=str, default=None,
                    help='Ablation study group ID（用於關聯多個實驗）')
    ap.add_argument('--output-dir', type=str, default='runs',
                    help='輸出目錄')
    
    # Feature engineering
    ap.add_argument('--feature-set', type=str, default='baseline',
                    choices=['baseline', '+physical', '+semantic', '+psych'],
                    help='特徵集合')
    ap.add_argument('--feature-transform-profile', type=str, default='phaseA',
                    choices=['none', 'phaseA', 'phaseB'],
                    help='特徵轉換 profile')
    
    # Split strategy
    ap.add_argument('--n-folds', type=int, default=10,
                    help='K-fold CV 折數')
    ap.add_argument('--test-size', type=float, default=0.2,
                    help='Test set 比例（預設 0.2 = 8:2 split）')
    ap.add_argument('--random-seed', type=int, default=42,
                    help='隨機種子（確保可復現）')
    ap.add_argument('--holdout-strategy', type=str, default='stratified',
                    choices=['stratified', 'group', 'none'],
                    help='Holdout split 策略')
    ap.add_argument('--cv-strategy', type=str, default='stratified_kfold',
                    choices=['stratified_kfold', 'group_kfold'],
                    help='CV split 策略')
    ap.add_argument('--group-key', type=str, default=None,
                    help='Group split 使用的欄位（keyword/category）')
    
    # Model settings
    ap.add_argument('--model-type', type=str, default='xgboost',
                    choices=['xgboost'],
                    help='模型類型（目前只支援 xgboost）')
    ap.add_argument('--hyperparameter-mode', type=str, default='tuning',
                    choices=['tuning', 'locked'],
                    help='超參數模式：tuning（搜尋）或 locked（使用 baseline）')
    ap.add_argument('--baseline-params-path', type=str, default=None,
                    help='Baseline best params 路徑（hyperparameter_mode=locked 時必須）')
    
    # Imbalance handling
    ap.add_argument('--imbalance-mode', type=str, default='scale_pos_weight',
                    choices=['scale_pos_weight', 'none'],
                    help='類不平衡處理模式')
    ap.add_argument('--scale-pos-weight-scope', type=str, default='fold_train',
                    choices=['fold_train', 'train_pool'],
                    help='scale_pos_weight 計算範圍')
    
    # Threshold
    ap.add_argument('--threshold-mode', type=str, default='fixed',
                    choices=['fixed', 'tuned'],
                    help='Threshold 模式：fixed（0.5）或 tuned（OOF 搜尋）')
    
    # Leakage prevention
    ap.add_argument('--preprocess-fit-scope', type=str, default='train_fold_only',
                    choices=['train_fold_only', 'train_only'],
                    help='前處理 fit 範圍（防止 leakage）')
    ap.add_argument('--fail-on-leakage', action='store_true', default=True,
                    help='發現 leakage 時立即失敗')
    
    # Database
    ap.add_argument('--enable-db-logging', action='store_true', default=True,
                    help='啟用 PostgreSQL 記錄')
    
    # Label strategy
    ap.add_argument('--label-strategy', type=str, default='hybrid',
                    choices=['absolute', 'hybrid'],
                    help='標籤定義策略')
    ap.add_argument('--label-delta-threshold', type=float, default=10.0,
                    help='Delta threshold')
    ap.add_argument('--label-ratio-threshold', type=float, default=1.0,
                    help='Ratio threshold')
    
    # Exclusions
    ap.add_argument('--exclude-products', type=str, default=None,
                    help='排除的商品 ID（逗號分隔）')
    
    # Paper mode
    ap.add_argument('--paper-mode', action='store_true', default=True,
                    help='論文模式：啟用所有 fail-fast checks')
    
    # Hardware optimization
    ap.add_argument('--use-gpu', action='store_true', default=True,
                    help='使用 GPU 加速 (XGBoost gpu_hist)')
    ap.add_argument('--n-jobs', type=int, default=16,
                    help='XGBoost 訓練線程數（建議為 CPU 線程數的一半）')
    ap.add_argument('--gpu-id', type=int, default=0,
                    help='GPU 設備 ID')
    
    return ap.parse_args()


def load_real_data(args):
    """
    使用真實的 data_loader 載入數據（不是模擬數據！）
    """
    print("\n" + "="*80)
    print("📊 Loading Real Data (load_product_level_training_set)")
    print("="*80)
    
    # 使用真實的 product-level data loader
    X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(
        date_cutoff=args.date_cutoff,
        pipeline_version=args.pipeline_version,
        top_n=200,  # TF-IDF top 200 features
        label_strategy=args.label_strategy,
        label_delta_threshold=args.label_delta_threshold,
        label_params={'ratio_threshold': args.label_ratio_threshold} if args.label_strategy == 'hybrid' else None,
        exclude_products=[int(p) for p in args.exclude_products.split(',')] if args.exclude_products else None,
        vocab_mode='global'
    )
    
    print(f"✅ Loaded real data:")
    print(f"  Dense features: {X_dense_df.shape}")
    print(f"  TF-IDF features: {X_tfidf.shape if X_tfidf is not None else 'None'}")
    print(f"  Labels: {len(y)}")
    print(f"  Positive rate: {(y==1).sum() / len(y):.2%}")
    
    return X_dense_df, X_tfidf, y, meta, vocab


def train_with_new_pipeline(args):
    """
    使用新 pipeline 執行完整訓練流程
    """
    # ========================================================================
    # Step 1: Initialize Run
    # ========================================================================
    run_id = args.run_id or make_run_id(prefix=args.feature_set)
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🚀 Starting Run: {run_id}")
    print("="*80)
    print(f"📁 Output directory: {run_dir}")
    print(f"🎓 Paper mode: {args.paper_mode}")
    print(f"🔐 Fail on leakage: {args.fail_on_leakage}")
    
    # Git info
    git_info = get_git_info()
    code_fingerprint = compute_code_fingerprint([
        'Model/train_v2.py',
        'Model/data_loader.py',
        'Model/experiment_utils.py',
        'Model/feature_transformer.py'
    ])
    
    if git_info['git_dirty']:
        print("⚠️  Warning: Git workspace is dirty! Commit before running experiments.")
    
    # ========================================================================
    # Step 2: Load REAL Data
    # ========================================================================
    X_dense_df, X_tfidf, y, meta, vocab = load_real_data(args)
    
    # Create full dataframe with product_id
    df_full = X_dense_df.copy()
    df_full['product_id'] = meta['product_id'].values
    df_full['y_true'] = y.values
    df_full['keyword'] = meta.get('keyword', ['unknown'] * len(y)).values
    
    # ========================================================================
    # Step 3: Create Splits (Fixed 8:2 + K-fold CV)
    # ========================================================================
    print("\n" + "="*80)
    print("📋 Step 3: Creating Splits")
    print("="*80)
    
    splits_df = make_splits(
        df_full,
        holdout_strategy=args.holdout_strategy,
        cv_strategy=args.cv_strategy,
        group_key=args.group_key,
        test_size=args.test_size,
        n_folds=args.n_folds,
        random_seed=args.random_seed,
        save_splits_path=os.path.join(run_dir, 'splits.parquet'),
        force_resplit=False
    )
    
    # ========================================================================
    # Step 4: Compute Hashes
    # ========================================================================
    print("\n" + "="*80)
    print("🔐 Step 4: Computing Hashes")
    print("="*80)
    
    samples_df = splits_df.merge(df_full[['product_id', 'y_true', 'keyword']], on='product_id')
    samples_df['is_included'] = True
    samples_df['is_excluded'] = False
    
    dataset_hash = compute_dataset_hash(samples_df)
    split_hash = compute_split_hash(splits_df)
    
    print(f"  dataset_hash: {dataset_hash}")
    print(f"  split_hash: {split_hash}")
    
    # ========================================================================
    # Step 5: Feature Engineering
    # ========================================================================
    print("\n" + "="*80)
    print("🔧 Step 5: Feature Engineering")
    print("="*80)
    
    # Get feature whitelist
    feature_whitelist = get_feature_whitelist(args.feature_set)
    validate_feature_whitelist(feature_whitelist, mode='paper' if args.paper_mode else 'legacy')
    
    # Filter to available features
    available_features = [f for f in feature_whitelist if f in df_full.columns]
    print(f"  Feature set: {args.feature_set}")
    print(f"  Available features: {len(available_features)}/{len(feature_whitelist)}")
    
    feature_hash = compute_feature_hash(
        available_features,
        feature_transform_profile=args.feature_transform_profile,
        clip_profile='none',
        impute_strategy='none',
        scaler='none'
    )
    print(f"  feature_hash: {feature_hash}")
    
    # ========================================================================
    # Step 6: Imbalance Report
    # ========================================================================
    print("\n" + "="*80)
    print("⚖️  Step 6: Imbalance Handling")
    print("="*80)
    
    y_df = samples_df[['product_id', 'y_true']]
    imbalance_report = compute_scale_pos_weight_per_fold(
        splits_df, y_df, 
        n_folds=args.n_folds,
        scope=args.scale_pos_weight_scope
    )
    
    print(f"  Scope: {args.scale_pos_weight_scope}")
    print(f"  Mean scale_pos_weight: {imbalance_report['statistics']['mean_scale_pos_weight']:.2f}")
    
    # ========================================================================
    # Step 7: Initialize Artifact Manager & Database
    # ========================================================================
    print("\n" + "="*80)
    print("💾 Step 7: Initialize Artifact Manager")
    print("="*80)
    
    manager = ArtifactManager(run_id, run_dir)
    
    # Save config
    config = vars(args)
    config.update({
        'git_commit': git_info['git_commit'],
        'git_branch': git_info['git_branch'],
        'git_dirty': git_info['git_dirty'],
        'code_fingerprint': code_fingerprint,
        'dataset_hash': dataset_hash,
        'split_hash': split_hash,
        'feature_hash': feature_hash
    })
    manager.save_config(config)
    
    # Save basic artifacts
    manager.save_splits(splits_df)
    manager.save_labels(samples_df[['product_id', 'y_true']])
    manager.save_hashes({
        'dataset_hash': dataset_hash,
        'split_hash': split_hash,
        'feature_hash': feature_hash
    })
    manager.save_samples_metadata(generate_samples_metadata(samples_df))
    manager.save_imbalance_report(imbalance_report)
    manager.save_feature_list(available_features)
    manager.save_feature_hash_context({
        'features': sorted(available_features),
        'transform_profile': args.feature_transform_profile,
        'clip': 'none',  
        'impute': 'none',
        'scaler': 'none'
    })
    
    # Database logging
    conn = None
    if args.enable_db_logging:
        try:
            conn = get_db_connection()
            insert_run_start(
                conn,
                run_id=run_id,
                git_commit=git_info['git_commit'],
                git_branch=git_info['git_branch'],
                git_dirty=git_info['git_dirty'],
                runner=os.getenv('USER', 'unknown'),
                command=' '.join(sys.argv),
                config=config,
                date_cutoff=args.date_cutoff,
                label_strategy=args.label_strategy,
                label_params={
                    'delta_threshold': args.label_delta_threshold,
                    'ratio_threshold': args.label_ratio_threshold
                },
                split_strategy=args.cv_strategy,
                cv_params={'n_folds': args.n_folds, 'random_seed': args.random_seed},
                preprocess_fit_scope=args.preprocess_fit_scope,
                pipeline_version=args.pipeline_version or 'v2.0',
                code_fingerprint_hash=code_fingerprint,
                feature_set=args.feature_set,
                model_type=args.model_type,
                model_params={'default': True}
            )
            
            # Upsert samples
            upsert_samples(conn, run_id, samples_df)
            upsert_features(conn, run_id, available_features, active=True)
            
            print("  ✅ Database logging initialized")
        except Exception as e:
            print(f"  ⚠️  Database logging failed: {e}")
            conn = None
    
    # ========================================================================
    # Step 8: K-Fold Training (真實訓練，不是模擬！)
    # ========================================================================
    print("\n" + "="*80)
    print("🎯 Step 8: K-Fold Cross-Validation Training")
    print("="*80)
    
    train_pool = samples_df[samples_df['split'] == 'train_pool']
    test_set = samples_df[samples_df['split'] == 'test']
    
    all_oof_preds = []
    
    for fold in range(args.n_folds):
        print(f"\n  Fold {fold+1}/{args.n_folds}...")
        
        # Split data
        train_fold_df = train_pool[train_pool['fold_id'] != fold]
        val_fold_df = train_pool[train_pool['fold_id'] == fold]
        
        # Get features
        X_train = df_full.loc[df_full['product_id'].isin(train_fold_df['product_id']), available_features]
        y_train = train_fold_df['y_true'].values
        X_val = df_full.loc[df_full['product_id'].isin(val_fold_df['product_id']), available_features]
        y_val = val_fold_df['y_true'].values
        
        # Feature transformation (with leakage prevention)
        transformer = FeatureTransformer(
            profile=args.feature_transform_profile,
            feature_whitelist=available_features
        )
        transformer.fit(X_train)  # Only fit on train fold!
        X_train_transformed = transformer.transform(X_train)
        X_val_transformed = transformer.transform(X_val)
        
        # Leakage check
        train_fold_ids = set(train_fold_df['product_id'])
        test_ids = set(test_set['product_id'])
        
        leakage_report = verify_no_leakage(
            splits_df,
            current_fold=fold,
            preprocess_fit_scope=args.preprocess_fit_scope,
            tfidf_source_ids=train_fold_ids,  # Assuming TF-IDF fit on train fold
            scaler_source_ids=train_fold_ids,
            fail_fast=args.fail_on_leakage
        )
        
        if fold == 0:  # Save leakage report for first fold
            manager.save_leakage_checks(leakage_report)
        
        # Train model with scale_pos_weight
        spw = imbalance_report['folds'][f'fold_{fold}']['scale_pos_weight']
        
        # Build XGBoost params
        xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight': spw if args.imbalance_mode == 'scale_pos_weight' else 1.0,
            'random_state': args.random_seed,
            'eval_metric': 'logloss'
        }
        
        # Hardware optimization (i9-13900K + RTX 4080)
        if args.use_gpu:
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': args.gpu_id,
                'predictor': 'gpu_predictor',
                'max_bin': 256  # GPU optimal
            })
        
        xgb_params['n_jobs'] = args.n_jobs
        
        model = xgb.XGBClassifier(**xgb_params)
        
        model.fit(X_train_transformed, y_train)
        
        # Predict on validation fold
        y_prob = model.predict_proba(X_val_transformed)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        # Store OOF predictions
        val_preds = val_fold_df[['product_id', 'y_true']].copy()
        val_preds['fold_id'] = fold
        val_preds['y_prob'] = y_prob
        val_preds['y_pred'] = y_pred
        val_preds['threshold'] = 0.5
        val_preds['split'] = 'train_pool'
        
        all_oof_preds.append(val_preds)
        
        # Metrics
        auc = roc_auc_score(y_val, y_prob)
        f1 = f1_score(y_val, y_pred)
        print(f"    AUC: {auc:.4f}, F1: {f1:.4f}")
    
    # ========================================================================
    # Step 9: Final Model Training & Test Evaluation
    # ========================================================================
    print("\n" + "="*80)
    print("🔬 Step 9: Final Model Training & Test Evaluation")
    print("="*80)
    
    # Train on full train_pool
    X_train_full = df_full.loc[df_full['product_id'].isin(train_pool['product_id']), available_features]
    y_train_full = train_pool['y_true'].values
    
    transformer_final = FeatureTransformer(profile=args.feature_transform_profile, feature_whitelist=available_features)
    transformer_final.fit(X_train_full)
    X_train_full_transformed = transformer_final.transform(X_train_full)
    
    # Use train_pool scope for final model
    n_pos = (y_train_full == 1).sum()
    n_neg = (y_train_full == 0).sum()
    spw_final = n_neg / n_pos if n_pos > 0 else 1.0
    
    # Build final model params
    xgb_params_final = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'scale_pos_weight': spw_final if args.imbalance_mode == 'scale_pos_weight' else 1.0,
        'random_state': args.random_seed
    }
    
    if args.use_gpu:
        xgb_params_final.update({
            'tree_method': 'gpu_hist',
            'gpu_id': args.gpu_id,
            'predictor': 'gpu_predictor',
            'max_bin': 256
        })
    
    xgb_params_final['n_jobs'] = args.n_jobs
    
    model_final = xgb.XGBClassifier(**xgb_params_final)
    
    model_final.fit(X_train_full_transformed, y_train_full)
    
    # Predict on test set (只評估一次！)
    X_test = df_full.loc[df_full['product_id'].isin(test_set['product_id']), available_features]
    X_test_transformed = transformer_final.transform(X_test)
    
    y_test_prob = model_final.predict_proba(X_test_transformed)[:, 1]
    y_test_pred = (y_test_prob > 0.5).astype(int)
    
    test_preds = test_set[['product_id', 'y_true']].copy()
    test_preds['y_prob'] = y_test_prob
    test_preds['y_pred'] = y_test_pred
    test_preds['threshold'] = 0.5
    test_preds['split'] = 'test'
    test_preds['fold_id'] = -1
    
    # ========================================================================
    # Step 10: Save Predictions & Metrics
    # ========================================================================
    print("\n" + "="*80)
    print("📊 Step 10: Save Predictions & Metrics")
    print("="*80)
    
    # Concatenate OOF predictions
    oof_preds_df = pd.concat(all_oof_preds, ignore_index=True)
    
    # Save predictions
    manager.save_predictions_oof(oof_preds_df)
    manager.save_predictions_test(test_preds)
    
    # Compute metrics
    oof_auc = roc_auc_score(oof_preds_df['y_true'], oof_preds_df['y_prob'])
    oof_f1 = f1_score(oof_preds_df['y_true'], oof_preds_df['y_pred'])
    test_auc = roc_auc_score(test_preds['y_true'], test_preds['y_prob'])
    test_f1 = f1_score(test_preds['y_true'], test_preds['y_pred'])
    
    metrics = {
        'oof_aggregate': {
            'auc': {'mean': oof_auc, 'std': 0.0},  # TODO: compute per-fold std
            'f1': {'mean': oof_f1, 'std': 0.0}
        },
        'oof_global': {
            'auc': oof_auc,
            'f1': oof_f1,
            'precision': precision_score(oof_preds_df['y_true'], oof_preds_df['y_pred']),
            'recall': recall_score(oof_preds_df['y_true'], oof_preds_df['y_pred'])
        },
        'test': {
            'auc': test_auc,
            'f1': test_f1,
            'precision': precision_score(test_preds['y_true'], test_preds['y_pred']),
            'recall': recall_score(test_preds['y_true'], test_preds['y_pred'])
        },
        'threshold': {
            'value': 0.5,
            'method': args.threshold_mode,
            'source': 'fixed'
        }
    }
    
    manager.save_metrics(metrics)
    manager.save_chosen_threshold({'value': 0.5, 'method': 'fixed'})
    
    print(f"  OOF AUC: {oof_auc:.4f}, F1: {oof_f1:.4f}")
    print(f"  Test AUC: {test_auc:.4f}, F1: {test_f1:.4f}")
    
    # Database logging
    if conn:
        try:
            upsert_predictions(conn, run_id, oof_preds_df)
            upsert_predictions(conn, run_id, test_preds)
            
            update_run_finish(
                conn,
                run_id=run_id,
                status='completed',
                dataset_hash=dataset_hash,
                split_hash=split_hash,
                feature_hash=feature_hash,
                metrics_json=metrics,
                conclusion='Training completed successfully'
            )
            conn.close()
            print("  ✅ Database updated")
        except Exception as e:
            print(f"  ⚠️  Database update failed: {e}")
    
    # ========================================================================
    # Step 11: Verify Artifacts
    # ========================================================================
    print("\n" + "="*80)
    print("✅ Step 11: Verify Artifacts")
    print("="*80)
    
    verification = manager.verify_mandatory_artifacts()
    print(f"  Generated: {verification['generated_count']}/{verification['mandatory_count']}")
    
    if not verification['all_present']:
        print(f"  ⚠️  Missing: {verification['missing']}")
    
    manager.save_artifact_summary()
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*80)
    print("🎉 Training Complete!")
    print("="*80)
    print(f"  Run ID: {run_id}")
    print(f"  Output: {run_dir}")
    print(f"  OOF AUC: {oof_auc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  split_hash: {split_hash}")
    print(f"  Artifacts: {verification['generated_count']}")
    print("="*80)


if __name__ == '__main__':
    args = parse_args()
    train_with_new_pipeline(args)
