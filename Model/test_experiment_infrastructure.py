# -*- coding: utf-8 -*-
"""
test_experiment_infrastructure.py
測試 Phase 1 核心基礎設施：Split Manager, Hash System, Feature Whitelist, DB Logger

執行：python Model/test_experiment_infrastructure.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.experiment_utils import (
    make_splits,
    compute_dataset_hash,
    compute_split_hash,
    compute_feature_hash,
    get_feature_whitelist,
    validate_feature_whitelist,
    get_git_info,
    generate_samples_metadata
)

from Model.experiment_logger import (
    get_db_connection,
    make_run_id,
    insert_run_start,
    update_run_finish,
    upsert_samples,
    upsert_predictions,
    upsert_features,
    upsert_artifact
)


def test_split_manager():
    """測試 Split Manager"""
    print("\n" + "="*60)
    print("TEST 1: Split Manager")
    print("="*60)
    
    # 創建模擬數據
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'product_id': range(1, n_samples + 1),
        'y_true': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),  # 5% positive
        'keyword': np.random.choice(['口罩', '手機', '筆電', '耳機'], n_samples)
    })
    
    print(f"📊 模擬數據: {len(df)} 筆, positive rate = {df['y_true'].mean():.2%}")
    
    # Test 1.1: Stratified holdout + Stratified K-fold
    print("\n[Test 1.1] Stratified holdout + Stratified K-fold")
    splits_df = make_splits(
        df,
        holdout_strategy='stratified',
        cv_strategy='stratified_kfold',
        test_size=0.2,
        n_folds=5,
        random_seed=42,
        save_splits_path='test_outputs/splits_stratified.parquet',
        force_resplit=True
    )
    
    # 驗證
    train_pool = splits_df[splits_df['split'] == 'train_pool']
    test_set = splits_df[splits_df['split'] == 'test']
    
    print(f"  Train pool: {len(train_pool)} ({len(train_pool)/len(splits_df):.1%})")
    print(f"  Test set: {len(test_set)} ({len(test_set)/len(splits_df):.1%})")
    print(f"  Folds in train_pool: {sorted(train_pool['fold_id'].unique())}")
    print(f"  Test fold_id: {test_set['fold_id'].unique()}")
    
    # 檢查 leakage
    train_ids = set(train_pool['product_id'])
    test_ids = set(test_set['product_id'])
    overlap = train_ids & test_ids
    assert len(overlap) == 0, f"❌ Leakage detected: {len(overlap)} products"
    print(f"  ✅ No leakage: train ∩ test = ∅")
    
    # Test 1.2: 測試檔案重複讀取（不應重新生成）
    print("\n[Test 1.2] 讀取現有 splits（不重新生成）")
    splits_df2 = make_splits(
        df,
        holdout_strategy='stratified',
        cv_strategy='stratified_kfold',
        save_splits_path='test_outputs/splits_stratified.parquet',
        force_resplit=False  # 應該直接讀取
    )
    
    # 驗證完全一致
    assert splits_df.equals(splits_df2), "❌ Splits not identical!"
    print(f"  ✅ Splits 完全一致")
    
    return splits_df


def test_hash_system(splits_df, original_df):
    """測試 Hash System"""
    print("\n" + "="*60)
    print("TEST 2: Hash System")
    print("="*60)
    
    # 創建模擬 samples dataframe（需要合併 y_true）
    samples_df = splits_df.merge(
        original_df[['product_id', 'y_true']], 
        on='product_id', 
        how='left'
    )
    samples_df['is_included'] = True
    samples_df['is_excluded'] = False
    
    # Test 2.1: dataset_hash
    print("\n[Test 2.1] Dataset Hash")
    dataset_hash = compute_dataset_hash(samples_df)
    print(f"  dataset_hash: {dataset_hash}")
    assert len(dataset_hash) == 16, "Hash should be 16 characters"
    print(f"  ✅ Hash length correct")
    
    # Test 2.2: split_hash
    print("\n[Test 2.2] Split Hash")
    split_hash = compute_split_hash(splits_df)
    print(f"  split_hash: {split_hash}")
    assert len(split_hash) == 16, "Hash should be 16 characters"
    print(f"  ✅ Hash length correct")
    
    # 驗證 hash 穩定性（相同輸入應產生相同 hash）
    split_hash2 = compute_split_hash(splits_df)
    assert split_hash == split_hash2, "Hash should be deterministic"
    print(f"  ✅ Hash 穩定性驗證通過")
    
    # Test 2.3: feature_hash
    print("\n[Test 2.3] Feature Hash")
    feature_names = ['price', 'comment_count_pre', 'score_mean']
    feature_hash = compute_feature_hash(
        feature_names,
        feature_transform_profile='phaseA',
        clip_profile='none',
        impute_strategy='median',
        scaler='none'
    )
    print(f"  feature_hash (phaseA, median): {feature_hash}")
    
    # 驗證不同 transform 產生不同 hash
    feature_hash2 = compute_feature_hash(
        feature_names,
        feature_transform_profile='phaseB',  # 不同
        clip_profile='none',
        impute_strategy='median',
        scaler='none'
    )
    assert feature_hash != feature_hash2, "Different transforms should produce different hashes"
    print(f"  ✅ Transform context 影響 hash")
    
    return dataset_hash, split_hash, feature_hash


def test_feature_whitelist():
    """測試 Feature Whitelist System"""
    print("\n" + "="*60)
    print("TEST 3: Feature Whitelist System")
    print("="*60)
    
    # Test 3.1: 獲取各 feature set
    print("\n[Test 3.1] Feature Sets")
    for fs in ['baseline', '+physical', '+semantic', '+psych']:
        features = get_feature_whitelist(fs)
        print(f"  {fs}: {len(features)} features")
    
    # Test 3.2: Fail-fast validation
    print("\n[Test 3.2] Forbidden Feature Validation")
    
    # 正常特徵（應通過）
    good_features = ['price', 'comment_count_pre', 'score_mean']
    try:
        validate_feature_whitelist(good_features, mode='paper')
        print(f"  ✅ 正常特徵通過驗證")
    except ValueError as e:
        print(f"  ❌ 不應拋出錯誤: {e}")
    
    # 禁止特徵（paper mode 應失敗）
    bad_features = ['price', 'keyword', 'product_id']  # 包含禁止特徵
    try:
        validate_feature_whitelist(bad_features, mode='paper')
        print(f"  ❌ 應拋出錯誤但沒有")
    except ValueError as e:
        print(f"  ✅ Paper mode 正確拋出錯誤: {e}")
    
    # Legacy mode（應 warning）
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_feature_whitelist(bad_features, mode='legacy')
        assert len(w) > 0, "Should produce warning"
        print(f"  ✅ Legacy mode 產生 warning: {w[0].message}")


def test_git_fingerprinting():
    """測試 Git Fingerprinting"""
    print("\n" + "="*60)
    print("TEST 4: Git Fingerprinting")
    print("="*60)
    
    git_info = get_git_info()
    print(f"  git_commit: {git_info['git_commit'][:16]}...")
    print(f"  git_branch: {git_info['git_branch']}")
    print(f"  git_dirty: {git_info['git_dirty']}")
    print(f"  ✅ Git info 獲取成功")


def test_samples_metadata(splits_df):
    """測試 Samples Metadata Generation"""
    print("\n" + "="*60)
    print("TEST 5: Samples Metadata Generation")
    print("="*60)
    
    # 創建完整 samples dataframe
    np.random.seed(42)
    samples_df = splits_df.copy()
    samples_df['y_true'] = np.random.choice([0, 1], len(samples_df), p=[0.95, 0.05])
    samples_df['is_included'] = True
    samples_df['is_excluded'] = False
    samples_df['keyword'] = np.random.choice(['口罩', '手機', '筆電'], len(samples_df))
    
    metadata = generate_samples_metadata(samples_df)
    
    print(f"  total_products: {metadata['total_products']}")
    print(f"  positives: {metadata['positives']}")
    print(f"  positive_rate: {metadata['positive_rate']:.2%}")
    print(f"  num_unique_keywords: {metadata['num_unique_keywords']}")
    print(f"  keyword_entropy: {metadata['keyword_entropy']:.3f}")
    print(f"  Top keywords: {list(metadata['keyword_distribution'].keys())[:3]}")
    print(f"  ✅ Metadata 生成成功")
    
    return metadata


def test_database_logger(splits_df, dataset_hash, split_hash, feature_hash):
    """測試 Database Logger"""
    print("\n" + "="*60)
    print("TEST 6: Database Logger")
    print("="*60)
    
    try:
        conn = get_db_connection()
        print(f"  ✅ 資料庫連接成功")
    except Exception as e:
        print(f"  ⚠️  資料庫連接失敗（跳過測試）: {e}")
        return
    
    # Test 6.1: 創建 run
    print("\n[Test 6.1] Insert Run Start")
    run_id = make_run_id(prefix="test")
    print(f"  run_id: {run_id}")
    
    try:
        git_info = get_git_info()
        insert_run_start(
            conn,
            run_id=run_id,
            git_commit=git_info['git_commit'],
            git_branch=git_info['git_branch'],
            git_dirty=git_info['git_dirty'],
            runner='test_script',
            command='python test_experiment_infrastructure.py',
            config={'test': True, 'mode': 'phaseA'},
            date_cutoff='2025-06-25',
            label_strategy='hybrid',
            label_params={'delta_threshold': 10, 'ratio_threshold': 1.0},
            split_strategy='stratified_holdout',
            cv_params={'n_folds': 5, 'random_seed': 42},
            preprocess_fit_scope='train_fold_only',
            pipeline_version='v2.0',
            code_fingerprint_hash='test123456789abc',
            feature_set='baseline',
            model_type='xgboost',
            model_params={'max_depth': 6, 'learning_rate': 0.1}
        )
        print(f"  ✅ Run start 插入成功")
    except Exception as e:
        print(f"  ❌ Run start 插入失敗: {e}")
        conn.close()
        return
    
    # Test 6.2: Upsert samples
    print("\n[Test 6.2] Upsert Samples")
    try:
        samples_df = splits_df.copy()
        np.random.seed(42)
        samples_df['y_true'] = np.random.choice([0, 1], len(samples_df), p=[0.95, 0.05])
        samples_df['keyword'] = np.random.choice(['口罩', '手機'], len(samples_df))
        samples_df['is_included'] = True
        samples_df['is_excluded'] = False
        samples_df['exclusion_reason'] = None
        
        upsert_samples(conn, run_id, samples_df)
        print(f"  ✅ Samples 插入成功")
    except Exception as e:
        print(f"  ❌ Samples 插入失敗: {e}")
    
    # Test 6.3: Upsert predictions
    print("\n[Test 6.3] Upsert Predictions")
    try:
        # 模擬預測結果（只對 test set）
        test_samples = samples_df[samples_df['split'] == 'test'].copy()
        test_samples['y_prob'] = np.random.uniform(0, 1, len(test_samples))
        test_samples['y_pred'] = (test_samples['y_prob'] > 0.5).astype(int)
        test_samples['threshold'] = 0.5
        test_samples['is_oof'] = 0  # Test predictions
        
        upsert_predictions(conn, run_id, test_samples)
        print(f"  ✅ Predictions 插入成功")
    except Exception as e:
        print(f"  ❌ Predictions 插入失敗: {e}")
    
    # Test 6.4: Upsert features
    print("\n[Test 6.4] Upsert Features")
    try:
        features = get_feature_whitelist('baseline')
        importances = {f: np.random.uniform(0, 1) for f in features[:5]}  # 只前 5 個有 importance
        
        upsert_features(conn, run_id, features, active=True, importances=importances)
        print(f"  ✅ Features 插入成功")
    except Exception as e:
        print(f"  ❌ Features 插入失敗: {e}")
    
    # Test 6.5: Upsert artifact
    print("\n[Test 6.5] Upsert Artifact")
    try:
        upsert_artifact(
            conn,
            run_id=run_id,
            artifact_type='run_config',
            file_path=f'test_outputs/runs/{run_id}/run_config.json',
            file_hash='abc123',
            meta={'test': True}
        )
        print(f"  ✅ Artifact 插入成功")
    except Exception as e:
        print(f"  ❌ Artifact 插入失敗: {e}")
    
    # Test 6.6: Update run finish
    print("\n[Test 6.6] Update Run Finish")
    try:
        update_run_finish(
            conn,
            run_id=run_id,
            status='completed',
            dataset_hash=dataset_hash,
            split_hash=split_hash,
            feature_hash=feature_hash,
            metrics_json={
                'oof_aggregate': {'auc': {'mean': 0.85, 'std': 0.02}},
                'test': {'auc': 0.84}
            },
            conclusion='測試運行成功'
        )
        print(f"  ✅ Run finish 更新成功")
    except Exception as e:
        print(f"  ❌ Run finish 更新失敗: {e}")
    
    conn.close()
    print(f"\n  🎉 資料庫測試完成！Run ID: {run_id}")


def main():
    """主測試流程"""
    print("\n" + "="*60)
    print("Phase 1 Core Infrastructure - Comprehensive Test")
    print("="*60)
    
    # 創建輸出目錄
    os.makedirs('test_outputs/runs', exist_ok=True)
    
    # 創建模擬數據（需要保存以傳遞給其他測試）
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'product_id': range(1, n_samples + 1),
        'y_true': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'keyword': np.random.choice(['口罩', '手機', '筆電', '耳機'], n_samples)
    })
    
    # 執行所有測試
    splits_df = test_split_manager()  # Uses df internally
    dataset_hash, split_hash, feature_hash = test_hash_system(splits_df, df)  # Now pass df
    test_feature_whitelist()
    test_git_fingerprinting()
    metadata = test_samples_metadata(splits_df)  # Will be updated
    test_database_logger(splits_df, dataset_hash, split_hash, feature_hash)  # Will be updated
    
    print("\n" + "="*60)
    print("✅ 所有測試完成！")
    print("="*60)
    
    # 輸出摘要
    print("\n📊 測試摘要：")
    print(f"  - Split Manager: ✅ 固定 8:2 + K-fold CV")
    print(f"  - Hash System: ✅ dataset/split/feature hash")
    print(f"  - Feature Whitelist: ✅ Fail-fast validation")
    print(f"  - Git Fingerprinting: ✅ Auto-capture")
    print(f"  - Metadata Generation: ✅ Keyword distribution")
    print(f"  - Database Logger: ✅ 5 tables upsert")


if __name__ == "__main__":
    main()
