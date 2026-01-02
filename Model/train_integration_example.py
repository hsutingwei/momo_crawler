# -*- coding: utf-8 -*-
"""
train_integration_example.py
展示如何將 Phase 1 modules 整合到訓練流程中的最小可運行示例

使用這個範本來更新 train.py
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# Import Phase 1 modules
from experiment_utils import (
    make_splits,
    compute_dataset_hash,
    compute_split_hash,
    compute_feature_hash,
    get_feature_whitelist,
    validate_feature_whitelist,
    get_git_info,
    compute_code_fingerprint,
    generate_samples_metadata
)

from experiment_logger import (
    get_db_connection,
    make_run_id,
    insert_run_start,
    update_run_finish,
    upsert_samples,
    upsert_predictions,
    upsert_features,
    upsert_artifact
)


def train_with_experiment_tracking_example(
    date_cutoff: str = '2025-06-25',
    feature_set: str = 'baseline',
    holdout_strategy: str = 'stratified',
    cv_strategy: str = 'stratified_kfold',
    n_folds: int = 10,
    random_seed: int = 42,
    group_id: Optional[str] = None,
    mode: str = 'paper'
):
    """
    整合實驗追蹤的訓練流程範例
    
    這個函數展示如何：
    1. 使用 make_splits() 創建固定的 splits
    2. 計算所有必要的 hashes
    3. 記錄到資料庫
    4. 產生所有 artifacts
    """
    
    # =================================================================
    # Step 1: 初始化 Run
    # =================================================================
    run_id = make_run_id(prefix=feature_set)
    run_dir = f'runs/{run_id}'
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Run: {run_id}")
    print(f"{'='*60}\n")
    
    # Get git info & code fingerprint
    git_info = get_git_info()
    code_fingerprint = compute_code_fingerprint([
        'Model/train.py',
        'Model/data_loader.py',
        'Model/experiment_utils.py'
    ])
    
    # =================================================================
    # Step 2: 載入數據 & 創建 Splits
    # =================================================================
    print("📊 Step 2: Loading data and creating splits...")
    
    # TODO: 用實際的 data_loader 替換
    # from data_loader import load_product_level_training_set
    # X_dense, X_tfidf, y, meta, vocab = load_product_level_training_set(...)
    
    # 示例：模擬數據
    np.random.seed(random_seed)
    n_samples = 1000
    df = pd.DataFrame({
        'product_id': range(1, n_samples + 1),
        'y_true': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'keyword': np.random.choice(['口罩', '手機', '筆電'], n_samples)
    })
    
    # 創建 splits（固定 8:2 + K-fold）
    splits_path = f'runs/{group_id or run_id}/splits.parquet'
    splits_df = make_splits(
        df,
        holdout_strategy=holdout_strategy,
        cv_strategy=cv_strategy,
        test_size=0.2,
        n_folds=n_folds,
        random_seed=random_seed,
        save_splits_path=splits_path,
        force_resplit=False  # 若已存在則直接讀取
    )
    
    # Merge splits with data
    samples_df = splits_df.merge(df, on='product_id', how='left')
    samples_df['is_included'] = True
    samples_df['is_excluded'] = False
    samples_df['exclusion_reason'] = None
    
    # =================================================================
    # Step 3: 計算 Hashes
    # =================================================================
    print("🔐 Step 3: Computing hashes...")
    
    dataset_hash = compute_dataset_hash(samples_df)
    split_hash = compute_split_hash(splits_df)
    
    # Get feature whitelist
    active_features = get_feature_whitelist(feature_set)
    validate_feature_whitelist(active_features, mode=mode)
    
    feature_hash = compute_feature_hash(
        active_features,
        feature_transform_profile='phaseA',
        clip_profile='none',
        impute_strategy='none',
        scaler='none'
    )
    
    print(f"  dataset_hash: {dataset_hash}")
    print(f"  split_hash: {split_hash}")
    print(f"  feature_hash: {feature_hash}")
    
    # =================================================================
    # Step 4: 記錄 Run Start 到資料庫
    # =================================================================
    print("💾 Step 4: Logging run start to database...")
    
    conn = get_db_connection()
    
    config = {
        'date_cutoff': date_cutoff,
        'feature_set': feature_set,
        'holdout_strategy': holdout_strategy,
        'cv_strategy': cv_strategy,
        'n_folds': n_folds,
        'random_seed': random_seed,
        'mode': mode
    }
    
    insert_run_start(
        conn,
        run_id=run_id,
        git_commit=git_info['git_commit'],
        git_branch=git_info['git_branch'],
        git_dirty=git_info['git_dirty'],
        runner=os.getenv('USER', 'unknown'),
        command=' '.join(sys.argv),
        config=config,
        date_cutoff=date_cutoff,
        label_strategy='hybrid',
        label_params={'delta_threshold': 10, 'ratio_threshold': 1.0},
        split_strategy='stratified_kfold',  # Must match CHECK constraint
        cv_params={'n_folds': n_folds, 'random_seed': random_seed},
        preprocess_fit_scope='train_fold_only',
        pipeline_version='v2.0',
        code_fingerprint_hash=code_fingerprint,
        feature_set=feature_set,
        model_type='xgboost',
        model_params={'max_depth': 6, 'learning_rate': 0.1}
    )
    
    # 記錄 samples
    upsert_samples(conn, run_id, samples_df)
    
    # 記錄 features
    upsert_features(conn, run_id, active_features, active=True)
    
    # =================================================================
    # Step 5: 訓練 & 評估（K-Fold CV）
    # =================================================================
    print("🎯 Step 5: Training with K-fold CV...")
    
    train_pool = samples_df[samples_df['split'] == 'train_pool']
    test_set = samples_df[samples_df['split'] == 'test']
    
    # 模擬 K-fold CV 訓練
    all_oof_preds = []
    
    for fold in range(n_folds):
        print(f"  Fold {fold+1}/{n_folds}...")
        
        train_idx = train_pool[train_pool['fold_id'] != fold]
        val_idx = train_pool[train_pool['fold_id'] == fold]
        
        # TODO: 實際訓練模型
        # model = train_model(train_idx)
        # y_prob = model.predict_proba(val_idx)[:, 1]
        
        # 模擬預測
        val_preds = val_idx.copy()
        val_preds['y_prob'] = np.random.uniform(0, 1, len(val_idx))
        val_preds['y_pred'] = (val_preds['y_prob'] > 0.5).astype(int)
        val_preds['threshold'] = 0.5
        val_preds['is_oof'] = 1
        
        all_oof_preds.append(val_preds)
    
    # 合併 OOF predictions
    oof_preds_df = pd.concat(all_oof_preds, ignore_index=True)
    
    # =================================================================
    # Step 6: Test Set 評估（只一次！）
    # =================================================================
    print("🔬 Step 6: Final test evaluation...")
    
    # TODO: 在完整 train_pool 上訓練最終模型
    # final_model = train_model(train_pool)
    # test_probs = final_model.predict_proba(test_set)[:, 1]
    
    # 模擬測試預測
    test_preds = test_set.copy()
    test_preds['y_prob'] = np.random.uniform(0, 1, len(test_set))
    test_preds['y_pred'] = (test_preds['y_prob'] > 0.5).astype(int)
    test_preds['threshold'] = 0.5
    test_preds['is_oof'] = 0
    
    # =================================================================
    # Step 7: 記錄 Predictions 到資料庫
    # =================================================================
    print("💾 Step 7: Saving predictions...")
    
    upsert_predictions(conn, run_id, oof_preds_df)
    upsert_predictions(conn, run_id, test_preds)
    
    # =================================================================
    # Step 8: 計算 Metrics & 更新 Run Finish
    # =================================================================
    print("📈 Step 8: Computing metrics and finishing run...")
    
    # 計算 metrics（簡化版）
    from sklearn.metrics import roc_auc_score, f1_score
    
    oof_auc = roc_auc_score(oof_preds_df['y_true'], oof_preds_df['y_prob'])
    oof_f1 = f1_score(oof_preds_df['y_true'], oof_preds_df['y_pred'])
    test_auc = roc_auc_score(test_preds['y_true'], test_preds['y_prob'])
    test_f1 = f1_score(test_preds['y_true'], test_preds['y_pred'])
    
    metrics_json = {
        'oof_aggregate': {
            'auc': {'mean': oof_auc, 'std': 0.02},  # TODO: 實際計算 std
            'f1': {'mean': oof_f1, 'std': 0.01}
        },
        'oof_global': {
            'auc': oof_auc,
            'f1': oof_f1
        },
        'test': {
            'auc': test_auc,
            'f1': test_f1
        }
    }
    
    update_run_finish(
        conn,
        run_id=run_id,
        status='completed',
        dataset_hash=dataset_hash,
        split_hash=split_hash,
        feature_hash=feature_hash,
        metrics_json=metrics_json,
        conclusion='Integration example completed successfully'
    )
    
    # =================================================================
    # Step 9: 保存 Artifacts
    # =================================================================
    print("💾 Step 9: Saving artifacts...")
    
    # Save splits
    splits_df.to_parquet(f'{run_dir}/splits.parquet', index=False)
    
    # Save predictions
    oof_preds_df.to_parquet(f'{run_dir}/predictions_oof.parquet', index=False)
    test_preds.to_parquet(f'{run_dir}/predictions_test.parquet', index=False)
    
    # Save metadata
    metadata = generate_samples_metadata(samples_df)
    with open(f'{run_dir}/samples_metadata.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    # Save hashes
    hashes = {
        'dataset_hash': dataset_hash,
        'split_hash': split_hash,
        'feature_hash': feature_hash
    }
    with open(f'{run_dir}/hashes.json', 'w') as f:
        json.dump(hashes, f, indent=2)
    
    # Save config
    full_config = {
        **config,
        'run_id': run_id,
        'git_commit': git_info['git_commit'],
        'git_branch': git_info['git_branch'],
        'code_fingerprint': code_fingerprint
    }
    with open(f'{run_dir}/run_config.json', 'w', encoding='utf-8') as f:
        json.dump(full_config, f, ensure_ascii=False, indent=2)
    
    # Save metrics
    with open(f'{run_dir}/metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Record artifacts in database
    for artifact_type, filepath in [
        ('splits', f'{run_dir}/splits.parquet'),
        ('run_config', f'{run_dir}/run_config.json'),
        ('metrics', f'{run_dir}/metrics.json'),
        ('predictions_oof', f'{run_dir}/predictions_oof.parquet'),
        ('predictions_test', f'{run_dir}/predictions_test.parquet'),
    ]:
        upsert_artifact(conn, run_id, artifact_type, filepath)
    
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"✅ Run Completed: {run_id}")
    print(f"{'='*60}")
    print(f"📁 Artifacts saved to: {run_dir}")
    print(f"📊 OOF AUC: {oof_auc:.4f}, Test AUC: {test_auc:.4f}")
    print(f"🔐 split_hash: {split_hash}")
    
    return run_id, metrics_json


if __name__ == "__main__":
    # 範例執行
    import argparse
    from typing import Optional
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-set', default='baseline', choices=['baseline', '+physical', '+semantic', '+psych'])
    parser.add_argument('--group-id', default=None)
    parser.add_argument('--mode', default='paper', choices=['paper', 'legacy'])
    args = parser.parse_args()
    
    run_id, metrics = train_with_experiment_tracking_example(
        feature_set=args.feature_set,
        group_id=args.group_id,
        mode=args.mode
    )
    
    print(f"\n🎉 Example run completed! Run ID: {run_id}")
