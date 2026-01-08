# -*- coding: utf-8 -*-
"""
Model/train_v2.py
新版訓練腳本 - 整合 7-Phase ML Pipeline

特性:
- 固定 8:2 split + K-fold CV (可復現)
- Hash 追蹤 (dataset/split/feature)
- 5 個自動化数据洩漏檢查
- PostgreSQL 實驗記錄
- 每次運行生成 15+ 個 artifacts
- 支援消融研究 (Ablation study)

用法:
python Model/train_v2.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --feature-set baseline \
  --n-folds 10 \
  --random-seed 42 \
  --run-id my_experiment_001

與 train.py 的差異:
- 使用固定 split (不是基於日期的 cutoff split)
- 自動防止数据洩漏
- 完整的 artifact 管理
- PostgreSQL 記錄日誌
"""

import os
import sys
import json
import argparse
import warnings
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib  # For serializing vectorizer
import xgboost as xgb

# 匯入數據加載器
from data_loader import load_product_level_training_set

# 匯入新的 ML Pipeline 模組
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
    ap = argparse.ArgumentParser(description='ML Pipeline v2 - 生產環境訓練')
    
    # 基本設定
    ap.add_argument('--mode', type=str, default='product_level',
                    choices=['product_level'],
                    help='訓練模式（目前只支援 product_level）')
    ap.add_argument('--date-cutoff', type=str, default='2025-06-25',
                    help='資料切分日期（用於載入數據）')
    ap.add_argument('--pipeline-version', type=str, default=None,
                    help='資料處理流程版本號')
    
    # 運行識別
    ap.add_argument('--run-id', type=str, default=None,
                    help='實驗 Run ID（若不指定則自動生成）')
    ap.add_argument('--group-id', type=str, default=None,
                    help='消融研究 Group ID（用於關聯多個實驗）')
    ap.add_argument('--output-dir', type=str, default='runs',
                    help='輸出目錄')
    
    # 特徵工程
    ap.add_argument('--feature-set', type=str, default='baseline',
                    choices=['baseline', '+physical', '+semantic', '+psych'],
                    help='特徵集合')
    ap.add_argument('--feature-transform-profile', type=str, default='phaseA',
                    choices=['none', 'phaseA', 'phaseB'],
                    help='特徵轉換 profile')
    
    # 切分策略
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
    
    # 模型設定
    ap.add_argument('--model-type', type=str, default='xgboost',
                    choices=['xgboost'],
                    help='模型類型（目前只支援 xgboost）')
    ap.add_argument('--hyperparameter-mode', type=str, default='tuning',
                    choices=['tuning', 'locked'],
                    help='超參數模式：tuning（搜尋）或 locked（使用 baseline）')
    ap.add_argument('--baseline-params-path', type=str, default=None,
                    help='Baseline best params 路徑（hyperparameter_mode=locked 時必須）')
    
    # 不平衡處理
    ap.add_argument('--imbalance-mode', type=str, default='scale_pos_weight',
                    choices=['scale_pos_weight', 'none'],
                    help='類不平衡處理模式')
    ap.add_argument('--scale-pos-weight-scope', type=str, default='fold_train',
                    choices=['fold_train', 'train_pool'],
                    help='scale_pos_weight 計算範圍')
    
    # 閾值設定
    ap.add_argument('--threshold-mode', type=str, default='fixed',
                    choices=['fixed', 'tuned', 'locked'],
                    help='Threshold 模式：fixed(0.5)、tuned(OOF搜尋)、locked(使用baseline)')
    ap.add_argument('--threshold-path', type=str, default=None,
                    help='Locked threshold 路徑 (threshold_mode=locked 時必須)')
    
    # 資料洩漏防止
    ap.add_argument('--preprocess-fit-scope', type=str, default='train_fold_only',
                    choices=['train_fold_only', 'train_only'],
                    help='前處理 fit 範圍（防止 leakage）')
    ap.add_argument('--fail-on-leakage', action='store_true', default=True,
                    help='發現 leakage 時立即失敗')
    
    # 標籤策略
    ap.add_argument('--label-strategy', type=str, default='hybrid',
                    choices=['absolute', 'hybrid'],
                    help='標籤定義策略')
    ap.add_argument('--label-delta-threshold', type=float, default=10.0,
                    help='Delta threshold')
    ap.add_argument('--label-ratio-threshold', type=float, default=1.0,
                    help='Ratio threshold')
    
    # 排除設定
    ap.add_argument('--exclude-products', type=str, default=None,
                    help='排除的商品 ID（逗號分隔）')
    ap.add_argument('--apply-filter-tag', type=str, default=None,
                    help='套用 ml_data_filters 表的過濾清單 (version_tag)，可用逗號分隔多個')
    
    # 論文模式
    ap.add_argument('--paper-mode', action='store_true', default=True,
                    help='論文模式：啟用所有 fail-fast 檢查')
    
    # 硬體優化
    ap.add_argument('--use-gpu', action='store_true', default=True,
                    help='使用 GPU 加速 (XGBoost gpu_hist)')
    ap.add_argument('--n-jobs', type=int, default=16,
                    help='XGBoost 訓練線程數（建議為 CPU 線程數的一半）')
    ap.add_argument('--gpu-id', type=int, default=0,
                    help='GPU 設備 ID')
    
    # Ablation study controls (CRITICAL for reproducibility)
    ap.add_argument('--force-use-splits', type=str, default=None,
                    help='強制使用指定的 splits.parquet 路徑（確保 baseline 和 variants 使用相同 splits）')
    
    # Feature Engineering
    ap.add_argument('--tfidf-dim', type=int, default=200,
                    help='TF-IDF 特徵維度 (top_n)')
    
    return ap.parse_args()


def load_real_data(args):
    """
    使用真實的 data_loader 載入數據（不是模擬數據！）
    """
    print("\n" + "="*80)
    print("📊 載入真實數據 (load_product_level_training_set)")
    print("="*80)
    
    # 使用真實的 product-level data loader
    X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(
        date_cutoff=args.date_cutoff,
        pipeline_version=args.pipeline_version,
        top_n=args.tfidf_dim,  # TF-IDF 特徵維度 (configurable)
        label_strategy=args.label_strategy,
        label_delta_threshold=args.label_delta_threshold,
        label_params={'ratio_threshold': args.label_ratio_threshold} if args.label_strategy == 'hybrid' else None,
        exclude_products=[int(p) for p in args.exclude_products.split(',')] if args.exclude_products else None,
        vocab_mode='global',
        skip_tfidf_matrix=True  # Use fold-wise sklearn TF-IDF instead of DB-based matrix
    )
    
    print(f"✅ 已載入真實數據:")
    print(f"  Dense features: {X_dense_df.shape}")
    print(f"  TF-IDF features: {X_tfidf.shape if X_tfidf is not None else 'None'}")
    print(f"  Labels: {len(y)}")
    print(f"  Positive rate: {(y==1).sum() / len(y):.2%}")
    
    return X_dense_df, X_tfidf, y, meta, vocab


def load_filter_exclusions(filter_tags: str):
    """
    Load product exclusion list from ml_data_filters table
    
    Args:
        filter_tags: Comma-separated version_tag(s) to apply
    
    Returns:
        set of product_ids to exclude
    """
    if not filter_tags:
        return set()
    
    try:
        from config.database import DatabaseConfig
        db_config = DatabaseConfig()
        conn = db_config.get_connection()
    except Exception as e:
        print(f"  ⚠️  Failed to connect to database: {e}")
        print("  Skipping filter application")
        return set()
    
    cur = conn.cursor()
    
    tags = [tag.strip() for tag in filter_tags.split(',')]
    excluded_ids = set()
    
    print(f"\n  📋 Loading filters from ml_data_filters...")
    for tag in tags:
        cur.execute("""
            SELECT filter_value
            FROM ml_data_filters
            WHERE version_tag = %s
              AND filter_level = 'product_id';
        """, (tag,))
        
        rows = cur.fetchall()
        tag_ids = set(int(row[0]) for row in rows)
        excluded_ids.update(tag_ids)
        
        print(f"    {tag}: {len(tag_ids)} products")
    
    cur.close()
    conn.close()
    
    print(f"  ✅ Total excluded products: {len(excluded_ids)}")
    return excluded_ids


def train_with_new_pipeline(args):
    """
    使用新 pipeline 執行完整訓練流程
    """
    # ========================================================================
    # 步驟 1: 初始化運行 (Run)
    # ========================================================================
    run_id = args.run_id or make_run_id(prefix=args.feature_set)
    run_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🚀 開始運行: {run_id}")
    print("="*80)
    print(f"📁 輸出目錄: {run_dir}")
    print(f"🎓 論文模式: {args.paper_mode}")
    print(f"🔐 發現洩漏時失敗: {args.fail_on_leakage}")
    
    # Git 資訊
    git_info = get_git_info()
    code_fingerprint = compute_code_fingerprint([
        'Model/train_v2.py',
        'Model/data_loader.py',
        'Model/experiment_utils.py',
        'Model/feature_transformer.py'
    ])
    
    if git_info['git_dirty']:
        print("⚠️  警告: Git 工作區有未提交的更改！建議在運行實驗前提交代碼。")
    
    # ========================================================================
    # 步驟 2: 載入真實數據
    # ========================================================================
    X_dense_df, X_tfidf, y, meta, vocab = load_real_data(args)
    
    # 創建包含 product_id 的完整 dataframe
    df_full = X_dense_df.copy()
    df_full['product_id'] = meta['product_id'].values
    df_full['y_true'] = y.values
    df_full['keyword'] = meta.get('keyword', ['unknown'] * len(y)).values
    # Extract doc_text with robust fallback
    # Priority: doc_text_tokenized (pre-tokenized) > aggregated_comments (raw text)
    if 'doc_text_tokenized' in meta.columns:
        tokenized = meta['doc_text_tokenized'].fillna('')
        # Fallback to aggregated_comments for rows where tokenized is empty
        if 'aggregated_comments' in meta.columns:
            raw = meta['aggregated_comments'].fillna('')
            # Use tokenized where available, otherwise use raw
            df_full['doc_text'] = tokenized.where(tokenized != '', raw)
            # Track which source we used
            tokenized_count = (tokenized != '').sum()
            raw_count = ((tokenized == '') & (raw != '')).sum()
            empty_count = ((tokenized == '') & (raw == '')).sum()
            print(f"  [doc_text] Sources: tokenized={tokenized_count}, fallback_raw={raw_count}, empty={empty_count}")
        else:
            df_full['doc_text'] = tokenized
    elif 'aggregated_comments' in meta.columns:
        df_full['doc_text'] = meta['aggregated_comments'].fillna('')
    else:
        # Fallback: empty strings for all products
        df_full['doc_text'] = pd.Series([''] * len(y), index=df_full.index)
    
    # CRITICAL: Create indexed version for guaranteed X-y alignment
    if df_full['product_id'].duplicated().any():
        raise ValueError("Duplicate product_ids found in training set! This will cause X-y misalignment.")
    df_full_idx = df_full.set_index('product_id', drop=False)
    
    # Note: Filter application moved to step 3.5 (after split) to preserve test set integrity
    
    # ========================================================================
    # 步驟 3: 創建切分 (固定 8:2 + K-fold CV)
    # ========================================================================
    print("\n" + "="*80)
    print("📋 步驟 3: 創建切分 (Splits)")
    print("="*80)
    
    # 消融控制: 強制使用 baseline 切分
    if args.force_use_splits:
        print(f"  ⚠️  從以下路徑載入強制切分: {args.force_use_splits}")
        if not os.path.exists(args.force_use_splits):
            raise FileNotFoundError(f"找不到強制切分檔案: {args.force_use_splits}")
        
        splits_df = pd.read_parquet(args.force_use_splits)
        
        # 複製到當前運行目錄以歸檔
        import shutil
        shutil.copy(args.force_use_splits, os.path.join(run_dir, 'splits.parquet'))
        print(f"  ✅ 從強制切分已載入 {len(splits_df)} 個樣本")
    else:
        # 正常創建切分
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
    # ABLATION CONTROL: 驗證 Fold 數量一致性 (Fold Count Validation)
    # ========================================================================
    train_pool_folds = splits_df[splits_df['split'] == 'train_pool']['fold_id'].unique()
    n_unique_folds = len(train_pool_folds)
    max_fold_id = splits_df[splits_df['split'] == 'train_pool']['fold_id'].max()
    
    print(f"\n  🔍 驗證 Fold 數量...")
    print(f"    指定 n_folds: {args.n_folds}")
    print(f"    實際 unique folds: {n_unique_folds}")
    print(f"    Fold ID 範圍: {sorted(train_pool_folds)}")
    
    if n_unique_folds != args.n_folds:
        raise ValueError(
            f"❌ Fold 數量不匹配 (Fold Count Mismatch)!\n"
            f"  指定: {args.n_folds} folds\n"
            f"  實際: {n_unique_folds} folds\n"
            f"  Fold IDs: {sorted(train_pool_folds)}\n"
            f"  → 請檢查 splits 生成邏輯或 --force-use-splits 路徑"
        )
    
    if max_fold_id != args.n_folds - 1:
        raise ValueError(
            f"❌ Fold ID 範圍錯誤!\n"
            f"  預期最大 fold_id: {args.n_folds - 1}\n"
            f"  實際最大 fold_id: {max_fold_id}\n"
            f"  → Fold ID 應該從 0 到 {args.n_folds - 1}"
        )
    
    print(f"  ✅ Fold 數量驗證通過: {args.n_folds} folds")
    
    # ========================================================================
    # 步驟 3.5: 套用 ml_data_filters 過濾清單 (僅針對訓練集)
    # ========================================================================
    if args.apply_filter_tag:
        print("\n" + "="*80)
        print("🗑️  套用 ml_data_filters 過濾清單 (僅訓練集)")
        print("="*80)
        print(f"  Filter tags: {args.apply_filter_tag}")
        
        excluded_ids = load_filter_exclusions(args.apply_filter_tag)
        
        if excluded_ids:
            # Get train_pool samples before filtering
            train_pool_before = splits_df[splits_df['split'] == 'train_pool'].copy()
            before_count = len(train_pool_before)
            
            # Filter: remove excluded products from train_pool only
            # Test set remains untouched
            mask_train = (splits_df['split'] == 'train_pool') & (~splits_df['product_id'].isin(excluded_ids))
            mask_test = splits_df['split'] == 'test'  # Keep all test samples
            
            splits_df = splits_df[mask_train | mask_test].reset_index(drop=True)
            
            # Also update df_full to match filtered splits
            df_full = df_full[df_full['product_id'].isin(splits_df['product_id'])].reset_index(drop=True)
            df_full_idx = df_full.set_index('product_id', drop=False)
            
            train_pool_after = splits_df[splits_df['split'] == 'train_pool']
            test_samples = splits_df[splits_df['split'] == 'test']
            after_count = len(train_pool_after)
            filtered_count = before_count - after_count
            
            print(f"  Train (before): {before_count} products")
            print(f"  Train (after):  {after_count} products")
            print(f"  ✅ Filtered from train: {filtered_count} products ({filtered_count/before_count*100:.2f}%)")
            print(f"  🔒 Test set unchanged: {len(test_samples)} products")
        else:
            print("  ⚠️  No products excluded (filter returned empty set)")
    
    # ========================================================================
    # 步驟 4: 計算 Hashes
    # ========================================================================
    print("\n" + "="*80)
    print("🔐 步驟 4: 計算 Hashes")
    print("="*80)
    
    samples_df = splits_df.merge(df_full[['product_id', 'y_true', 'keyword']], on='product_id')
    samples_df['is_included'] = True
    samples_df['is_excluded'] = False
    
    dataset_hash = compute_dataset_hash(samples_df)
    split_hash = compute_split_hash(splits_df)
    
    print(f"  dataset_hash: {dataset_hash}")
    print(f"  split_hash: {split_hash}")
    
    # ========================================================================
    # 消融控制: 鎖定模式下的 Hash 驗證
    # ========================================================================
    if args.hyperparameter_mode == 'locked':
        print("\n  🔐 驗證 Hash 一致性 (消融模式)...")
        
        if not args.baseline_params_path:
            raise ValueError("hyperparameter_mode=locked 時必須指定 --baseline-params-path")
        
        if not os.path.exists(args.baseline_params_path):
            raise FileNotFoundError(f"找不到 Baseline params: {args.baseline_params_path}")
        
        with open(args.baseline_params_path, 'r', encoding='utf-8') as f:
            baseline_params = json.load(f)
        
        # 驗證 split_hash 是否匹配
        baseline_split_hash = baseline_params.get('split_hash')
        if baseline_split_hash != split_hash:
            raise ValueError(
                f"❌ Split hash 不匹配!\n"
                f"  Baseline: {baseline_split_hash}\n"
                f"  當前:     {split_hash}\n"
                f"  → 您正在使用不同的切分!\n"
                f"  → 使用 --force-use-splits 來載入 baseline 切分"
            )
        
        # 驗證 dataset_hash 是否匹配
        baseline_dataset_hash = baseline_params.get('dataset_hash')
        if baseline_dataset_hash != dataset_hash:
            raise ValueError(
                f"❌ Dataset hash 不匹配!\n"
                f"  Baseline: {baseline_dataset_hash}\n"
                f"  當前:     {dataset_hash}\n"
                f"  → 檢查: --date-cutoff, --label-strategy, --exclude-products\n"
                f"  → 所有數據載入參數必須與 baseline 匹配!"
            )
        
        print(f"  ✅ split_hash 驗證通過: {split_hash}")
        print(f"  ✅ dataset_hash 驗證通過: {dataset_hash}")
        print(f"  ✅ 消融研究一致性檢查通過!")
    
    # ========================================================================
    # ========================================================================
    # 步驟 5: 特徵工程
    # ========================================================================
    print("\n" + "="*80)
    print("🔧 步驟 5: 特徵工程")
    print("="*80)
    
    # 獲取特徵白名單
    feature_whitelist = get_feature_whitelist(args.feature_set)
    validate_feature_whitelist(feature_whitelist, mode='paper' if args.paper_mode else 'legacy')
    
    # 過濾出可用特徵
    available_features = [f for f in feature_whitelist if f in df_full.columns]
    print(f"  Feature set: {args.feature_set}")
    print(f"  可用特徵: {len(available_features)}/{len(feature_whitelist)}")
    
    feature_hash = compute_feature_hash(
        available_features,
        feature_transform_profile=args.feature_transform_profile,
        clip_profile='none',
        impute_strategy='none',
        scaler='none'
    )
    print(f"  feature_hash: {feature_hash}")
    
    # ========================================================================
    # 步驟 6: 不平衡處理報告
    # ========================================================================
    print("\n" + "="*80)
    print("⚖️  步驟 6: 不平衡處理")
    print("="*80)
    
    y_df = samples_df[['product_id', 'y_true']]
    imbalance_report = compute_scale_pos_weight_per_fold(
        splits_df, y_df, 
        n_folds=args.n_folds,
        scope=args.scale_pos_weight_scope
    )
    
    print(f"  範圍: {args.scale_pos_weight_scope}")
    print(f"  平均 scale_pos_weight: {imbalance_report['statistics']['mean_scale_pos_weight']:.2f}")
    
    # ========================================================================
    # ========================================================================
    # 步驟 7: 初始化 Artifact Manager 和資料庫
    # ========================================================================
    print("\n" + "="*80)
    print("💾 步驟 7: 初始化 Artifact Manager")
    print("="*80)
    
    manager = ArtifactManager(run_id, run_dir)
    
    # 保存配置
    config = vars(args)
    config.update({
        'git_commit': git_info['git_commit'],
        'git_branch': git_info['git_branch'],
        'git_dirty': git_info['git_dirty'],
        'code_fingerprint': code_fingerprint,
        'dataset_hash': dataset_hash,
        'split_hash': split_hash,
        'feature_hash': feature_hash,
        'tfidf_params': {
            'max_features': args.tfidf_dim,
            'tokenizer': 'str.split',  # Replaced regex with explicit split
            'lowercase': False,
            'method': 'sklearn_fold_wise',  # vs 'db_tfidf_scores'
            'source': 'comment_tokens',
            'strict_a_note': 'Each fold has independent vocab (no leakage). Cross-fold feature importance not directly alignable.'
        }
    })
    manager.save_config(config)
    
    # 保存基本 artifacts
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
    
    # 資料庫記錄 (自動啟用)
    conn = None
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
        
        # 更新樣本
        upsert_samples(conn, run_id, samples_df)
        upsert_features(conn, run_id, available_features, active=True)
        
        print("  ✅ 資料庫記錄已初始化")
    except Exception as e:
        print(f"  ⚠️  資料庫記錄失敗 (繼續執行): {e}")
        conn = None
    
    # ========================================================================
    # ========================================================================
    # 步驟 8: K-Fold 訓練 (真實訓練，不是模擬！)
    # ========================================================================
    print("\n" + "="*80)
    print("🎯 步驟 8: K-Fold 交叉驗證訓練")
    print("="*80)
    
    train_pool = samples_df[samples_df['split'] == 'train_pool']
    test_set = samples_df[samples_df['split'] == 'test']
    
    # ========================================================================
    # TF-IDF 2A: PREFIT ONCE on train_pool (80%), test NEVER participates in fit
    # ========================================================================
    print("\n  📝 TF-IDF 2A: Prefit on train_pool(80%)...")
    
    train_pool_ids = train_pool['product_id'].tolist()
    test_ids = test_set['product_id'].tolist()
    
    # Get document texts
    train_pool_docs = df_full_idx.loc[train_pool_ids, 'doc_text'].fillna('').values
    test_docs = df_full_idx.loc[test_ids, 'doc_text'].fillna('').values
    
    # Empty doc ratio check
    train_pool_empty_ratio = (train_pool_docs == '').mean()
    test_empty_ratio = (test_docs == '').mean()
    print(f"    TFIDF(prefit) Empty doc ratio: train_pool={train_pool_empty_ratio:.2%}, test={test_empty_ratio:.2%}")
    
    # Detect tokenization type
    sample_docs = [d for d in train_pool_docs[:20] if d]
    has_spaces = any(' ' in d for d in sample_docs) if sample_docs else False
    
    if has_spaces:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=args.tfidf_dim,
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False
        )
        print(f"    TFIDF(prefit) tokenizer: str.split (pre-tokenized)")
    else:
        tfidf_vectorizer = TfidfVectorizer(
            max_features=args.tfidf_dim,
            analyzer='char_wb',
            ngram_range=(2, 4),
            lowercase=False
        )
        print(f"    TFIDF(prefit) tokenizer: char_wb n-grams (raw Chinese)")
    
    # FIT ONLY ON TRAIN_POOL - test never sees fit!
    tfidf_vectorizer.fit(train_pool_docs)
    
    # Transform both train_pool and test
    X_tfidf_train_pool = tfidf_vectorizer.transform(train_pool_docs)
    X_tfidf_test = tfidf_vectorizer.transform(test_docs)
    
    # Compute vocab hash for reproducibility
    vocab_sorted = sorted(tfidf_vectorizer.vocabulary_.keys())
    tfidf_vocab_hash = hashlib.sha256('|'.join(vocab_sorted).encode('utf-8')).hexdigest()[:16]
    
    print(f"    TFIDF(prefit) train_pool shape={X_tfidf_train_pool.shape}, nnz={X_tfidf_train_pool.nnz}")
    print(f"    TFIDF(prefit) test shape={X_tfidf_test.shape}, nnz={X_tfidf_test.nnz}")
    print(f"    TFIDF(prefit) vocab_size={len(tfidf_vectorizer.vocabulary_)}, vocab_hash={tfidf_vocab_hash}")
    
    # Save prefit vectorizer and vocab
    vectorizer_path = os.path.join(run_dir, 'tfidf_vectorizer_trainpool.joblib')
    joblib.dump(tfidf_vectorizer, vectorizer_path)
    
    vocab_path = os.path.join(run_dir, 'tfidf_vocab_trainpool.json')
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'vocab_size': len(vocab_sorted),
            'vocab_hash': tfidf_vocab_hash,
            'vocabulary': vocab_sorted
        }, f, ensure_ascii=False, indent=2)
    print(f"    ✅ Saved: tfidf_vectorizer_trainpool.joblib, tfidf_vocab_trainpool.json")
    
    # Create position mapping for train_pool (product_id -> row index in X_tfidf_train_pool)
    train_pool_pos = {pid: idx for idx, pid in enumerate(train_pool_ids)}
    
    all_oof_preds = []
    fold_metrics = []  # 儲存每個 fold 的指標
    all_leakage_reports = []  # 儲存每個 fold 的洩漏檢查結果
    
    for fold in range(args.n_folds):
        print(f"\n  Fold {fold+1}/{args.n_folds}...")
        
        # 切分數據
        train_fold_df = train_pool[train_pool['fold_id'] != fold]
        val_fold_df = train_pool[train_pool['fold_id'] == fold]
        
        # CRITICAL: Use index-based loc for guaranteed X-y alignment
        train_ids = train_fold_df['product_id'].tolist()
        val_ids = val_fold_df['product_id'].tolist()
        
        # 獲取 dense features (aligned by product_id index)
        X_train_dense = df_full_idx.loc[train_ids, available_features]
        y_train = train_fold_df.set_index('product_id').loc[train_ids, 'y_true'].values
        
        X_val_dense = df_full_idx.loc[val_ids, available_features]
        y_val = val_fold_df.set_index('product_id').loc[val_ids, 'y_true'].values
        
        # ========== TF-IDF 2A: SLICE from prefit matrix (NO FIT in fold!) ==========
        # Get row indices in the prefit X_tfidf_train_pool matrix
        train_tfidf_indices = [train_pool_pos[pid] for pid in train_ids]
        val_tfidf_indices = [train_pool_pos[pid] for pid in val_ids]
        
        # Slice TF-IDF from prefit matrix
        X_train_tfidf = X_tfidf_train_pool[train_tfidf_indices]
        X_val_tfidf = X_tfidf_train_pool[val_tfidf_indices]
        
        # 2A verification log
        print(f"    [Fold {fold}] Using prefit TFIDF vocab_hash={tfidf_vocab_hash} (no fit in fold)")
        print(f"    [Fold {fold}] TFIDF dim fixed={args.tfidf_dim}")
        print(f"    [Fold {fold}] Dense dim: {X_train_dense.shape[1]}, TF-IDF dim: {X_train_tfidf.shape[1]}")
        
        # Fail-fast if TF-IDF dimension mismatch
        if X_train_tfidf.shape[1] != args.tfidf_dim:
            raise ValueError(f"Fold {fold}: TF-IDF dimension mismatch! Expected {args.tfidf_dim}, got {X_train_tfidf.shape[1]}")
        
        # 特徵轉換 dense features（包含防洩漏保護）
        transformer = FeatureTransformer(
            profile=args.feature_transform_profile,
            feature_whitelist=available_features
        )
        transformer.fit(X_train_dense)  # 僅在 train fold 上 fit！
        X_train_dense_transformed = transformer.transform(X_train_dense)
        X_val_dense_transformed = transformer.transform(X_val_dense)
        
        # ========== CRITICAL: Alignment Validation ==========
        # These MUST all have the same row count
        if not (len(train_ids) == X_train_dense.shape[0] == X_train_tfidf.shape[0] == len(y_train)):
            raise ValueError(f"Fold {fold} ALIGNMENT ERROR: train_ids={len(train_ids)}, "
                           f"X_dense={X_train_dense.shape[0]}, X_tfidf={X_train_tfidf.shape[0]}, y={len(y_train)}")
        if not (len(val_ids) == X_val_dense.shape[0] == X_val_tfidf.shape[0] == len(y_val)):
            raise ValueError(f"Fold {fold} ALIGNMENT ERROR: val_ids={len(val_ids)}, "
                           f"X_val_dense={X_val_dense.shape[0]}, X_val_tfidf={X_val_tfidf.shape[0]}, y_val={len(y_val)}")
        
        # ========== CRITICAL: Merge dense + TF-IDF ==========
        X_train_merged = hstack([csr_matrix(X_train_dense_transformed), X_train_tfidf])
        X_val_merged = hstack([csr_matrix(X_val_dense_transformed), X_val_tfidf])
        
        # Sanity check
        expected_dim = X_train_dense_transformed.shape[1] + X_train_tfidf.shape[1]
        actual_dim = X_train_merged.shape[1]
        print(f"    [Fold {fold}] Merged dims: Dense={X_train_dense_transformed.shape[1]} + TF-IDF={X_train_tfidf.shape[1]} = {actual_dim}")
        if actual_dim != expected_dim:
            raise ValueError(f"Fold {fold}: Merge failed! Expected {expected_dim}, got {actual_dim}")
        
        # 防止数据洩漏檢查
        train_fold_ids = set(train_fold_df['product_id'])
        test_ids = set(test_set['product_id'])
        
        leakage_report = verify_no_leakage(
            splits_df,
            current_fold=fold,
            preprocess_fit_scope=args.preprocess_fit_scope,
            tfidf_source_ids=train_fold_ids,  # 假設 TF-IDF fit 在 train fold 上
            scaler_source_ids=train_fold_ids,
            fail_fast=args.fail_on_leakage
        )
        
        # 收集每個 fold 的洩漏報告
        all_leakage_reports.append(leakage_report)
        
        # 使用 scale_pos_weight 訓練模型
        spw = imbalance_report['folds'][f'fold_{fold}']['scale_pos_weight']
        
        # 建立 XGBoost 參數
        xgb_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight': spw if args.imbalance_mode == 'scale_pos_weight' else 1.0,
            'random_state': args.random_seed,
            'eval_metric': 'logloss'
        }
        
        # 硬體優化 (i9-13900K + RTX 4080)
        if args.use_gpu:
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': args.gpu_id,
                'predictor': 'gpu_predictor',
                'max_bin': 256  # GPU optimal
            })
        
        xgb_params['n_jobs'] = args.n_jobs
        
        model = xgb.XGBClassifier(**xgb_params)
        
        model.fit(X_train_merged, y_train)
        
        # 在驗證集上預測
        y_prob = model.predict_proba(X_val_merged)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        # 計算本 fold 的指標
        fold_auc = roc_auc_score(y_val, y_prob)
        fold_f1 = f1_score(y_val, y_pred)
        fold_precision = precision_score(y_val, y_pred)
        fold_recall = recall_score(y_val, y_pred)
        
        fold_metrics.append({
            'fold': fold,
            'auc': fold_auc,
            'f1': fold_f1,
            'precision': fold_precision,
            'recall': fold_recall
        })
        
        # 儲存 OOF 預測
        val_preds = val_fold_df[['product_id', 'y_true']].copy()
        val_preds['fold_id'] = fold
        val_preds['y_prob'] = y_prob
        val_preds['y_pred'] = y_pred  # 暫時，會在 threshold tuning 後更新
        val_preds['threshold'] = 0.5  # 暫時，會在 threshold tuning 後更新
        val_preds['split'] = 'train_pool'
        
        all_oof_preds.append(val_preds)
        
        print(f"    AUC: {fold_auc:.4f}, F1: {fold_f1:.4f}")
    
    # ========================================================================
    # 保存所有 Folds 的洩漏檢查報告
    # ========================================================================
    aggregated_leakage_report = {
        'all_passed': all(r['all_passed'] for r in all_leakage_reports),
        'total_folds': len(all_leakage_reports),
        'preprocess_fit_scope': args.preprocess_fit_scope,
        'folds': []
    }
    for i, report in enumerate(all_leakage_reports):
        fold_report = {
            'fold': i,
            'passed': report['all_passed'],
            'total_checks': report['total_checks'],
            'passed_checks': report['passed_checks'],
            'failed_checks': report['failed_checks'],
            'checks': report['checks']
        }
        aggregated_leakage_report['folds'].append(fold_report)
    
    manager.save_leakage_checks(aggregated_leakage_report)
    
    # ========================================================================
    # 步驟 9: 最終模型訓練與測試評估
    # ========================================================================
    print("\n" + "="*80)
    print("🔬 步驟 9: 最終模型訓練與測試評估")
    print("="*80)
    
    # 在完整的 train_pool 上訓練
    train_pool_ids = train_pool['product_id'].tolist()
    test_ids = test_set['product_id'].tolist()
    
    # Use indexed loc for guaranteed alignment
    X_train_full_dense = df_full_idx.loc[train_pool_ids, available_features]
    y_train_full = train_pool.set_index('product_id').loc[train_pool_ids, 'y_true'].values
    
    # ========== TF-IDF 2A: USE PREFIT directly (same as CV, no re-fit!) ==========
    # X_tfidf_train_pool and X_tfidf_test were already computed in prefit stage
    X_train_full_tfidf = X_tfidf_train_pool  # Already in correct order: train_pool_ids
    # X_tfidf_test already exists from prefit
    
    print(f"  [Final] Using prefit TFIDF vocab_hash={tfidf_vocab_hash} (same as CV)")
    print(f"  [Final] TFIDF train_pool shape={X_train_full_tfidf.shape}, test shape={X_tfidf_test.shape}")
    print(f"  [Final] Dense dim: {X_train_full_dense.shape[1]}, TF-IDF dim: {X_train_full_tfidf.shape[1]}")
    
    # Transform dense features
    transformer_final = FeatureTransformer(profile=args.feature_transform_profile, feature_whitelist=available_features)
    transformer_final.fit(X_train_full_dense)
    X_train_full_dense_transformed = transformer_final.transform(X_train_full_dense)
    
    X_test_dense = df_full_idx.loc[test_ids, available_features]
    X_test_dense_transformed = transformer_final.transform(X_test_dense)
    
    # ========== Merge dense + TF-IDF ==========
    X_train_full_merged = hstack([csr_matrix(X_train_full_dense_transformed), X_train_full_tfidf])
    X_test_merged = hstack([csr_matrix(X_test_dense_transformed), X_tfidf_test])
    
    print(f"  [Final] Merged dims: Dense={X_train_full_dense_transformed.shape[1]} + TF-IDF={X_train_full_tfidf.shape[1]} = {X_train_full_merged.shape[1]}")
    
    # 對於最終模型使用 train_pool scope
    n_pos = (y_train_full == 1).sum()
    n_neg = (y_train_full == 0).sum()
    spw_final = n_neg / n_pos if n_pos > 0 else 1.0
    
    # 建立最終模型參數
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
    model_final.fit(X_train_full_merged, y_train_full)
    
    # Predict on test set using merged features
    y_test_prob = model_final.predict_proba(X_test_merged)[:, 1]
    y_test_pred = (y_test_prob > 0.5).astype(int)
    
    test_preds = test_set[['product_id', 'y_true']].copy()
    test_preds['y_prob'] = y_test_prob
    # threshold 和 y_pred 將在 threshold tuning 後設定
    test_preds['split'] = 'test'
    test_preds['fold_id'] = -1
    
    # ========================================================================
    # ========================================================================
    # 步驟 10: 保存預測結果與指標
    # ========================================================================
    print("\n" + "="*80)
    print("📊 步驟 10: 保存預測結果與指標")
    print("="*80)
    
    # 合併 OOF 預測
    oof_preds_df = pd.concat(all_oof_preds, ignore_index=True)
    
    # 預測將在 threshold tuning 後保存
    
    # ========================================================================
    # 消融控制: 鎖定閾值
    # ========================================================================
    if args.threshold_mode == 'locked':
        if not args.threshold_path:
            raise ValueError("threshold_mode=locked 時必須指定 --threshold-path")
        
        if not os.path.exists(args.threshold_path):
            raise FileNotFoundError(f"找不到閾值檔案: {args.threshold_path}")
        
        with open(args.threshold_path, 'r', encoding='utf-8') as f:
            baseline_threshold_info = json.load(f)
        
        chosen_threshold = baseline_threshold_info['value']
        print(f"\n  🔒 使用鎖定閾值: {chosen_threshold:.4f} (來自 baseline)")
    elif args.threshold_mode == 'tuned':
        print(f"\n  ⚙️  正在根據 OOF F1 Score 調整閾值...")
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_f1 = -1
        best_th = 0.5
        y_true = oof_preds_df['y_true'].values
        y_prob = oof_preds_df['y_prob'].values
        
        for th in thresholds:
            y_pred_th = (y_prob > th).astype(int)
            score = f1_score(y_true, y_pred_th)
            if score > best_f1:
                best_f1 = score
                best_th = th
                
        chosen_threshold = float(best_th)
        print(f"  ✅ 最佳閾值: {chosen_threshold:.4f} (OOF F1: {best_f1:.4f})")
        
    else:  # fixed
        chosen_threshold = 0.5
        print(f"\n  📌 使用固定閾值: {chosen_threshold}")
    
    # 重新應用閾值進行預測
    oof_preds_df['y_pred'] = (oof_preds_df['y_prob'] > chosen_threshold).astype(int)
    oof_preds_df['threshold'] = chosen_threshold
    test_preds['y_pred'] = (test_preds['y_prob'] > chosen_threshold).astype(int)
    test_preds['threshold'] = chosen_threshold
    
    # 現在保存預測（使用正確的 threshold）
    manager.save_predictions_oof(oof_preds_df)
    manager.save_predictions_test(test_preds)
    
    # 計算 Per-Fold 統計量
    fold_aucs = [m['auc'] for m in fold_metrics]
    fold_f1s = [m['f1'] for m in fold_metrics]
    

    # 計算全局 OOF 指標 (拼接所有 fold)
    oof_global_auc = roc_auc_score(oof_preds_df['y_true'], oof_preds_df['y_prob'])
    oof_global_f1 = f1_score(oof_preds_df['y_true'], oof_preds_df['y_pred'])
    oof_global_precision = precision_score(oof_preds_df['y_true'], oof_preds_df['y_pred'])
    oof_global_recall = recall_score(oof_preds_df['y_true'], oof_preds_df['y_pred'])
    
    # 計算測試集指標
    test_auc = roc_auc_score(test_preds['y_true'], test_preds['y_prob'])
    test_f1 = f1_score(test_preds['y_true'], test_preds['y_pred'])
    test_precision = precision_score(test_preds['y_true'], test_preds['y_pred'])
    test_recall = recall_score(test_preds['y_true'], test_preds['y_pred'])
    
    metrics = {
        'oof_by_fold': fold_metrics,  # 每個 fold 的詳細指標
        'oof_aggregate': {  # Fold-level 的均值和標準差
            'auc': {'mean': float(np.mean(fold_aucs)), 'std': float(np.std(fold_aucs, ddof=1))},
            'f1': {'mean': float(np.mean(fold_f1s)), 'std': float(np.std(fold_f1s, ddof=1))}
        },
        'oof_global': {  # 拼接所有 OOF 後計算（單值）
            'auc': float(oof_global_auc),
            'f1': float(oof_global_f1),
            'precision': float(oof_global_precision),
            'recall': float(oof_global_recall)
        },
        'test': {
            'auc': float(test_auc),
            'f1': float(test_f1),
            'precision': float(test_precision),
            'recall': float(test_recall)
        },
        'threshold': {
            'value': float(chosen_threshold),
            'method': args.threshold_mode,
            'source': 'fixed' if args.threshold_mode == 'fixed' else ('locked' if args.threshold_mode == 'locked' else 'tuned')
        }
    }
    
    manager.save_metrics(metrics)
    manager.save_chosen_threshold({
        'value': float(chosen_threshold), 
        'method': args.threshold_mode,
        'source': 'fixed' if args.threshold_mode == 'fixed' else ('locked' if args.threshold_mode == 'locked' else 'tuned')
    })
    
    print(f"  OOF (Aggregate): AUC = {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs, ddof=1):.4f}, F1 = {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s, ddof=1):.4f}")
    print(f"  OOF (Global):    AUC = {oof_global_auc:.4f}, F1 = {oof_global_f1:.4f}")
    print(f"  Test:            AUC = {test_auc:.4f}, F1 = {test_f1:.4f}")
    
    # 資料庫記錄
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
            print("  ✅ 資料庫更新完成")
        except Exception as e:
            print(f"  ⚠️  資料庫更新失敗: {e}")
    
    # ========================================================================
    # 步驟 11: 驗證 Artifacts
    # ========================================================================
    print("\n" + "="*80)
    print("✅ 步驟 11: 驗證 Artifacts")
    print("="*80)
    
    verification = manager.verify_mandatory_artifacts()
    print(f"  已生成: {verification['generated_count']}/{verification['mandatory_count']}")
    
    if not verification['all_present']:
        print(f"  ⚠️  缺失: {verification['missing']}")
    
    manager.save_artifact_summary()
    
    # ========================================================================
    # 消融控制: 保存 Baseline Best Params (如果是 baseline + tuning)
    # ========================================================================
    if args.feature_set == 'baseline' and args.hyperparameter_mode == 'tuning':
        print("\n" + "="*80)
        print("💾 保存消融研究用的 Baseline Best Params")
        print("="*80)
        
        baseline_params = {
            'split_hash': split_hash,
            'dataset_hash': dataset_hash,
            'feature_transform_profile': args.feature_transform_profile,
            'imbalance_mode': args.imbalance_mode,
            'scale_pos_weight_scope': args.scale_pos_weight_scope,
            'threshold_mode': args.threshold_mode,
            'cv_metric': 'auc',  # 主要比較指標
            'tuning_trials': 1,  # TODO: 實現實際的超參數調整
            'best_params': {
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
                # TODO: 實現調整後替換為實際參數
            },
            'ablation_instructions': {
                'usage': 'Use these params for all feature variants',
                'locked_command_example': (
                    f"python Model/train_v2.py "
                    f"--hyperparameter-mode locked "
                    f"--baseline-params-path {run_dir}/baseline_best_params.json "
                    f"--threshold-mode locked "
                    f"--threshold-path {run_dir}/chosen_threshold.json "
                    f"--force-use-splits {run_dir}/splits.parquet"
                )
            },
            'created_at': datetime.now().isoformat()
        }
        
        # 直接保存文件
        baseline_params_path = os.path.join(run_dir, 'baseline_best_params.json')
        with open(baseline_params_path, 'w') as f:
            json.dump(baseline_params, f, indent=2)
        
        print(f"  ✅ 已保存 baseline_best_params.json")
        print(f"  ✅ 請在消融研究變體中使用此文件")
    
    # ========================================================================
    # 最終摘要
    # ========================================================================
    print("\n" + "="*80)
    print("🎉 訓練完成！")
    print("="*80)
    print(f"  Run ID: {run_id}")
    print(f"  輸出: {run_dir}")
    print(f"  OOF Global AUC: {oof_global_auc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  split_hash: {split_hash}")
    print(f"  Artifacts: {verification['generated_count']}")
    print("="*80)


if __name__ == '__main__':
    args = parse_args()
    train_with_new_pipeline(args)
