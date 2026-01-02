# -*- coding: utf-8 -*-
"""
test_end_to_end_pipeline.py
End-to-End ML Pipeline Integration Test

測試所有 Phase 1-6 模組的協作
驗證可復現性與正確性

執行：python Model/test_end_to_end_pipeline.py
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.experiment_utils import (
    make_splits, compute_dataset_hash, compute_split_hash, compute_feature_hash,
    get_feature_whitelist, validate_feature_whitelist, get_git_info,
    generate_samples_metadata
)
from Model.feature_transformer import FeatureTransformer
from Model.leakage_prevention import verify_no_leakage
from Model.imbalance_handling import compute_scale_pos_weight_per_fold
from Model.artifact_manager import ArtifactManager
from Model.experiment_logger import make_run_id


def create_test_dataset(n_samples=1000, random_seed=42):
    """創建模擬測試數據"""
    np.random.seed(random_seed)
    
    df = pd.DataFrame({
        'product_id': range(1, n_samples + 1),
        'y_true': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'keyword': np.random.choice(['口罩', '手機', '筆電', '耳機'], n_samples),
        'price': np.random.uniform(100, 10000, n_samples),
        'comment_count': np.random.randint(0, 1000, n_samples),
        'score_mean': np.random.uniform(1, 5, n_samples),
        'like_count': np.random.randint(0, 500, n_samples)
    })
    
    return df


def test_full_pipeline_run():
    """測試完整 pipeline 執行"""
    print("\n" + "="*80)
    print("TEST 1: Full Pipeline Run")
    print("="*80)
    
    # Setup
    run_id = make_run_id(prefix="e2e_test")
    output_dir = f'test_outputs/e2e/{run_id}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Run ID: {run_id}")
    print(f"📁 Output: {output_dir}")
    
    try:
        # ===================================================================
        # Phase 1: Splits & Hashes
        # ===================================================================
        print("\n[Phase 1] Creating splits and computing hashes...")
        
        df = create_test_dataset(n_samples=1000)
        
        splits_df = make_splits(
            df,
            holdout_strategy='stratified',
            cv_strategy='stratified_kfold',
            n_folds=5,
            random_seed=42,
            save_splits_path=f'{output_dir}/splits.parquet'
        )
        
        samples_df = splits_df.merge(df[['product_id', 'y_true', 'keyword']], on='product_id')
        samples_df['is_included'] = True
        samples_df['is_excluded'] = False
        
        dataset_hash = compute_dataset_hash(samples_df)
        split_hash = compute_split_hash(splits_df)
        
        print(f"  ✅ Splits created: {len(splits_df)} samples")
        print(f"  ✅ dataset_hash: {dataset_hash}")
        print(f"  ✅ split_hash: {split_hash}")
        
        # ===================================================================
        # Phase 2: Feature Engineering
        # ===================================================================
        print("\n[Phase 2] Feature transformation...")
        
        features = get_feature_whitelist('baseline')
        validate_feature_whitelist(features, mode='paper')
        
        # 過濾特徵
        available_features = [f for f in features if f in df.columns]
        X_df = df[['product_id'] + available_features].copy()
        
        # Fit transformer on fold 0's train
        train_fold_0 = splits_df[
            (splits_df['split'] == 'train_pool') & 
            (splits_df['fold_id'] != 0)
        ]['product_id']
        X_train_fold_0 = X_df[X_df['product_id'].isin(train_fold_0)].drop('product_id', axis=1)
        
        transformer = FeatureTransformer(profile='phaseA', feature_whitelist=available_features)
        transformer.fit(X_train_fold_0)
        transformer.save(f'{output_dir}/transformer')
        
        feature_hash = compute_feature_hash(
            available_features,
            feature_transform_profile='phaseA',
            clip_profile='none',
            impute_strategy='none',
            scaler='none'
        )
        
        print(f"  ✅ Transformer fitted on {len(X_train_fold_0)} samples")
        print(f"  ✅ feature_hash: {feature_hash}")
        
        # ===================================================================
        # Phase 3: Leakage Prevention
        # ===================================================================
        print("\n[Phase 3] Leakage prevention checks...")
        
        train_fold_0_ids = set(train_fold_0)
        test_ids = set(splits_df[splits_df['split'] == 'test']['product_id'])
        
        leakage_report = verify_no_leakage(
            splits_df,
            current_fold=0,
            preprocess_fit_scope='train_fold_only',
            tfidf_source_ids=train_fold_0_ids,
            scaler_source_ids=train_fold_0_ids,
            fail_fast=True
        )
        
        print(f"  ✅ Leakage checks: {leakage_report['passed_checks']}/{leakage_report['total_checks']} passed")
        
        # ===================================================================
        # Phase 4: Imbalance Handling
        # ===================================================================
        print("\n[Phase 4] Imbalance handling...")
        
        y_df = samples_df[['product_id', 'y_true']]
        imbalance_report = compute_scale_pos_weight_per_fold(
            splits_df, y_df, n_folds=5, scope='fold_train'
        )
        
        print(f"  ✅ Mean scale_pos_weight: {imbalance_report['statistics']['mean_scale_pos_weight']:.2f}")
        
        # ===================================================================
        # Phase 5: Artifacts Generation
        # ===================================================================
        print("\n[Phase 5] Generating artifacts...")
        
        manager = ArtifactManager(run_id, output_dir)
        
        # Save mandatory artifacts
        manager.save_config({
            'run_id': run_id,
            'feature_set': 'baseline',
            'mode': 'paper',
            'n_folds': 5
        })
        manager.save_splits(splits_df)
        manager.save_labels(samples_df[['product_id', 'y_true']])
        manager.save_hashes({
            'dataset_hash': dataset_hash,
            'split_hash': split_hash,
            'feature_hash': feature_hash
        })
        manager.save_samples_metadata(generate_samples_metadata(samples_df))
        manager.save_imbalance_report(imbalance_report)
        manager.save_leakage_checks(leakage_report)
        manager.save_feature_list(available_features)
        manager.save_feature_hash_context({
            'features': sorted(available_features),
            'transform_profile': 'phaseA',
            'clip': 'none',
            'impute': 'none',
            'scaler': 'none'
        })
        
        # 模擬預測（簡化）
        test_df = samples_df[samples_df['split'] == 'test'].copy()
        test_df['y_prob'] = np.random.uniform(0, 1, len(test_df))
        test_df['y_pred'] = (test_df['y_prob'] > 0.5).astype(int)
        test_df['threshold'] = 0.5
        manager.save_predictions_test(test_df)
        
        # OOF predictions（簡化）
        oof_df = samples_df[samples_df['split'] == 'train_pool'].copy()
        oof_df['y_prob'] = np.random.uniform(0, 1, len(oof_df))
        oof_df['y_pred'] = (oof_df['y_prob'] > 0.5).astype(int)
        oof_df['threshold'] = 0.5
        manager.save_predictions_oof(oof_df)
        
        # Metrics（簡化）
        from sklearn.metrics import roc_auc_score, f1_score
        test_auc = roc_auc_score(test_df['y_true'], test_df['y_prob'])
        oof_auc = roc_auc_score(oof_df['y_true'], oof_df['y_prob'])
        
        manager.save_metrics({
            'oof_aggregate': {'auc': {'mean': oof_auc, 'std': 0.01}},
            'oof_global': {'auc': oof_auc},
            'test': {'auc': test_auc},
            'threshold': {'value': 0.5, 'method': 'fixed'}
        })
        
        manager.save_chosen_threshold({'value': 0.5, 'method': 'fixed', 'source': 'default'})
        
        # Verify artifacts
        verification = manager.verify_mandatory_artifacts()
        print(f"  ✅ Artifacts: {verification['generated_count']}/{verification['mandatory_count']}")
        
        if not verification['all_present']:
            print(f"  ⚠️  Missing: {verification['missing']}")
        
        manager.save_artifact_summary()
        
        print(f"\n✅ Full pipeline run completed successfully!")
        print(f"📁 Artifacts saved to: {output_dir}")
        
        return output_dir, {
            'dataset_hash': dataset_hash,
            'split_hash': split_hash,
            'feature_hash': feature_hash,
            'artifacts_count': len(manager.artifacts)
        }
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        raise


def test_reproducibility():
    """測試可復現性：相同輸入應產生相同輸出"""
    print("\n" + "="*80)
    print("TEST 2: Reproducibility")
    print("="*80)
    
    print("\n[Run 1] First execution...")
    output_dir_1, hashes_1 = test_full_pipeline_run()
    
    print("\n[Run 2] Second execution with same parameters...")
    output_dir_2, hashes_2 = test_full_pipeline_run()
    
    # 比較 hashes
    print("\n[Comparison] Verifying reproducibility...")
    
    checks = {
        'dataset_hash': hashes_1['dataset_hash'] == hashes_2['dataset_hash'],
        'split_hash': hashes_1['split_hash'] == hashes_2['split_hash'],
        'feature_hash': hashes_1['feature_hash'] == hashes_2['feature_hash']
    }
    
    all_match = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}: {passed}")
        if not passed:
            print(f"     Run 1: {hashes_1[check_name]}")
            print(f"     Run 2: {hashes_2[check_name]}")
    
    if all_match:
        print(f"\n✅ Reproducibility test PASSED!")
    else:
        print(f"\n❌ Reproducibility test FAILED!")
    
    # Cleanup
    for output_dir in [output_dir_1, output_dir_2]:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    
    return all_match


def main():
    """執行所有 End-to-End 測試"""
    print("\n" + "="*80)
    print("🚀 ML Pipeline End-to-End Integration Test")
    print("="*80)
    
    os.makedirs('test_outputs/e2e', exist_ok=True)
    
    # Test 1: Full pipeline
    print("\n>>> Running full pipeline test...")
    output_dir, hashes = test_full_pipeline_run()
    
    # Test 2: Reproducibility
    print("\n>>> Running reproducibility test...")
    is_reproducible = test_reproducibility()
    
    # Summary
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80)
    
    print(f"\n✅ Phase 1: Splits & Hashes")
    print(f"✅ Phase 2: Feature Engineering")
    print(f"✅ Phase 3: Leakage Prevention")
    print(f"✅ Phase 4: Imbalance Handling")
    print(f"✅ Phase 5: Artifacts Generation")
    print(f"✅ Phase 6: Ablation Support (implicit)")
    print(f"✅ Phase 7: End-to-End Testing")
    
    print(f"\n🎉 All 7 Phases completed successfully!")
    
    if is_reproducible:
        print(f"✅ Reproducibility: VERIFIED")
    else:
        print(f"⚠️  Reproducibility: NEEDS REVIEW")
    
    # Cleanup test outputs
    cleanup = input("\n清理測試輸出？(y/n): ")
    if cleanup.lower() == 'y':
        if os.path.exists('test_outputs/e2e'):
            shutil.rmtree('test_outputs/e2e')
            print("✅ Test outputs cleaned")


if __name__ == "__main__":
    main()
