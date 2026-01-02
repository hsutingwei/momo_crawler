# -*- coding: utf-8 -*-
"""
experiment_utils.py
實驗追蹤工具：Split Manager、Hash 計算、Feature Whitelist

實現規格參考：implementation_plan.md
- Phase 1.1: Split Manager (固定 8:2 + K-fold CV)
- Phase 1.2: Hash System (dataset/split/feature hash)
- Phase 1.3: Feature Whitelist System
"""

import os
import hashlib
import json
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold


# =============================================================================
# Constants
# =============================================================================

FORBIDDEN_FEATURES = ['product_id', 'y_true', 'fold_id', 'split', 'keyword', 'category']


# =============================================================================
# 1. Split Manager
# =============================================================================

def make_splits(
    df: pd.DataFrame,
    holdout_strategy: str = 'stratified',
    cv_strategy: str = 'stratified_kfold',
    group_key: Optional[str] = None,
    test_size: float = 0.2,
    n_folds: int = 10,
    random_seed: int = 42,
    save_splits_path: Optional[str] = None,
    force_resplit: bool = False
) -> pd.DataFrame:
    """
    創建固定的 8:2 holdout split + K-fold CV 分配
    
    規格：implementation_plan.md #1.1, #23
    
    Args:
        df: 必須包含 product_id, y_true 欄位（以及 keyword/category 若使用 group）
        holdout_strategy: 'stratified' (by y_true) | 'group' | 'none'
        cv_strategy: 'stratified_kfold' | 'group_kfold'
        group_key: 'keyword' | 'category' (required for group strategies)
        test_size: Test set 比例（預設 0.2）
        n_folds: CV fold 數量（預設 10）
        random_seed: 隨機種子
        save_splits_path: 存檔路徑（若存在則讀取）
        force_resplit: 是否強制重新生成 splits
    
    Returns:
        DataFrame with columns: product_id, split ('train_pool'/'test'), fold_id (0-9 or -1)
    """
    # 優先序規則：若檔案存在且不強制重生成，則讀取
    if save_splits_path and os.path.exists(save_splits_path) and not force_resplit:
        print(f"[Split Manager] 讀取現有 splits: {save_splits_path}")
        splits_df = pd.read_parquet(save_splits_path)
        return splits_df
    
    print(f"[Split Manager] 生成新splits: holdout={holdout_strategy}, cv={cv_strategy}")
    
    # 驗證必要欄位
    required_cols = ['product_id', 'y_true']
    if holdout_strategy == 'group' or cv_strategy == 'group_kfold':
        if group_key is None:
            raise ValueError("group_key is required for group-based strategies")
        required_cols.append(group_key)
    
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Initialize splits dataframe
    df_work = df[required_cols].copy()
    n_total = len(df_work)
    n_test = int(n_total * test_size)
    
    # =========================
    # Step 1: 8:2 Holdout Split
    # =========================
    np.random.seed(random_seed)
    
    if holdout_strategy == 'stratified':
        # Stratify by y_true (論文預設)
        test_ids = _stratified_sample(df_work, 'y_true', n_test, random_seed)
    
    elif holdout_strategy == 'group':
        # Group-based: 整個 group 分配到 train_pool 或 test
        test_ids = _group_sample(df_work, group_key, n_test, random_seed)
    
    elif holdout_strategy == 'none':
        # Random split (不推薦用於論文)
        test_ids = df_work.sample(n=n_test, random_state=random_seed)['product_id'].tolist()
    
    else:
        raise ValueError(f"Unknown holdout_strategy: {holdout_strategy}")
    
    # 分配 split
    df_work['split'] = 'train_pool'
    df_work.loc[df_work['product_id'].isin(test_ids), 'split'] = 'test'
    
    # =========================
    # Step 2: K-Fold CV within train_pool
    # =========================
    train_mask = df_work['split'] == 'train_pool'
    df_train = df_work[train_mask].copy()
    
    if cv_strategy == 'stratified_kfold':
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        fold_splits = skf.split(df_train, df_train['y_true'])
    
    elif cv_strategy == 'group_kfold':
        gkf = GroupKFold(n_splits=n_folds)
        fold_splits = gkf.split(df_train, df_train['y_true'], groups=df_train[group_key])
    
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")
    
    # 分配 fold_id
    df_work['fold_id'] = -1  # Default: test set
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        product_ids_in_fold = df_train.iloc[val_idx]['product_id'].tolist()
        df_work.loc[df_work['product_id'].isin(product_ids_in_fold), 'fold_id'] = fold_idx
    
    # 對 train_pool 中未分配到 fold 的樣本（理論上不應發生）
    unassigned = df_work[(df_work['split'] == 'train_pool') & (df_work['fold_id'] == -1)]
    if len(unassigned) > 0:
        warnings.warn(f"{len(unassigned)} train_pool samples not assigned to any fold")
        # 隨機分配
        for pid in unassigned['product_id']:
            df_work.loc[df_work['product_id'] == pid, 'fold_id'] = np.random.randint(0, n_folds)
    
    # =========================
    # Step 3: Guardrails
    # =========================
    # 驗證 train_pool ∩ test = ∅
    train_ids = set(df_work[df_work['split'] == 'train_pool']['product_id'])
    test_ids_set = set(df_work[df_work['split'] == 'test']['product_id'])
    overlap = train_ids & test_ids_set
    if overlap:
        raise ValueError(f"Leakage detected: {len(overlap)} products in both train and test!")
    
    # 只保留必要欄位
    splits_df = df_work[['product_id', 'split', 'fold_id']].copy()
    
    # 存檔
    if save_splits_path:
        os.makedirs(os.path.dirname(save_splits_path), exist_ok=True)
        splits_df.to_parquet(save_splits_path, index=False)
        print(f"[Split Manager] Splits saved to: {save_splits_path}")
    
    return splits_df


def _stratified_sample(df: pd.DataFrame, stratify_col: str, n_sample: int, seed: int) -> List:
    """分層抽樣"""
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(
        df, 
        test_size=n_sample, 
        stratify=df[stratify_col], 
        random_state=seed
    )
    return test_df['product_id'].tolist()


def _group_sample(df: pd.DataFrame, group_col: str, n_sample: int, seed: int) -> List:
    """Group-based sampling: 整個 group 分配到 test"""
    np.random.seed(seed)
    groups = df[group_col].unique()
    np.random.shuffle(groups)
    
    test_ids = []
    for group in groups:
        group_ids = df[df[group_col] == group]['product_id'].tolist()
        test_ids.extend(group_ids)
        if len(test_ids) >= n_sample:
            break
    
    return test_ids[:n_sample]


# =============================================================================
# 2. Hash Computation System
# =============================================================================

def compute_dataset_hash(samples_df: pd.DataFrame) -> str:
    """
    計算 dataset_hash：基於最終納入訓練的樣本 (product_id, y_true)
    
    規格：implementation_plan.md #1.2, #19
    
    Args:
        samples_df: 必須包含 product_id, y_true, is_included 欄位
    
    Returns:
        16-character hash
    """
    if 'is_included' in samples_df.columns:
        final_samples = samples_df[samples_df['is_included'] == True].copy()
    else:
        final_samples = samples_df.copy()
    
    if len(final_samples) == 0:
        warnings.warn("No included samples for dataset_hash computation!")
        return "EMPTY_DATASET"
    
    # 只用 product_id + y_true（不含 keyword）
    sorted_df = final_samples[['product_id', 'y_true']].sort_values('product_id')
    payload = sorted_df.to_csv(index=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]


def compute_split_hash(splits_df: pd.DataFrame) -> str:
    """
    計算 split_hash：基於 (product_id, split, fold_id) 排序後 hash
    
    規格：implementation_plan.md #1.2
    
    Args:
        splits_df: 必須包含 product_id, split, fold_id 欄位
    
    Returns:
        16-character hash
    """
    sorted_df = splits_df[['product_id', 'split', 'fold_id']].sort_values('product_id')
    payload = sorted_df.to_csv(index=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]


def compute_feature_hash(
    feature_names: List[str],
    feature_transform_profile: str = 'none',
    clip_profile: str = 'none',
    impute_strategy: str = 'none',
    scaler: str = 'none'
) -> str:
    """
    計算 feature_hash：包含特徵名稱 + transform context
    
    規格：implementation_plan.md #1.2, #10
    
    Args:
        feature_names: 活躍特徵名稱列表
        feature_transform_profile: 'none' | 'phaseA' | 'phaseB'
        clip_profile: 'none' | 'p99' | 'p995'
        impute_strategy: 'none' | 'median' | 'zero'
        scaler: 'none' | 'standard' | 'robust'
    
    Returns:
        16-character hash
    """
    context = {
        'features': sorted(feature_names),
        'transform_profile': feature_transform_profile,
        'clip': clip_profile,
        'impute': impute_strategy,
        'scaler': scaler
    }
    payload = json.dumps(context, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]


# =============================================================================
# 3. Feature Whitelist System
# =============================================================================

def get_feature_whitelist(feature_set: str) -> List[str]:
    """
    獲取指定 feature_set 的 whitelist
    
    規格：implementation_plan.md #1.3
    
    Args:
        feature_set: 'baseline' | '+physical' | '+semantic' | '+psych'
    
    Returns:
        允許的特徵名稱列表
    """
    # Baseline features (基礎特徵)
    baseline = [
        'price', 'comment_count_pre', 'score_mean', 'like_count_sum',
        'comment_count_7d', 'comment_count_30d', 'comment_count_90d',
        'days_since_last_comment', 'comment_7d_ratio', 'comment_1st_30d',
        'comment_2nd_30d', 'comment_3rd_30d', 'score_std', 'comment_var',
        'avg_comment_len', 'has_high_score'
    ]
    
    # Physical features (動力學特徵)
    physical = [
        'kin_v_1', 'kin_v_2', 'kin_v_3',
        'kin_acc_abs', 'kin_acc_rel',
        'kin_jerk_abs',
        'early_bird_momentum', 'quality_driven_momentum'
    ]
    
    # Semantic features (語義特徵)
    semantic = [
        'bert_arousal', 'bert_novelty', 'bert_repurchase',
        'bert_negative', 'bert_advertisement',
        'category_fit_score'
    ]
    
    # Psychological features (心理學特徵)
    psych = [
        'diversity_tfidf', 'diversity_sbert',
        'organic_ratio', 'burst_intensity',
        'entropy_score'
    ]
    
    if feature_set == 'baseline':
        return baseline
    elif feature_set == '+physical':
        return baseline + physical
    elif feature_set == '+semantic':
        return baseline + physical + semantic
    elif feature_set == '+psych':
        return baseline + physical + semantic + psych
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


def validate_feature_whitelist(
    feature_names: List[str],
    mode: str = 'paper'
) -> None:
    """
    驗證特徵是否包含禁止的 metadata
    
    規格：implementation_plan.md #1.3, #15
    
    Args:
        feature_names: 要驗證的特徵名稱列表
        mode: 'paper' (fail-fast) | 'legacy' (warning only)
    
    Raises:
        ValueError: paper mode 下發現禁止特徵時
    """
    forbidden = set(feature_names) & set(FORBIDDEN_FEATURES)
    
    if forbidden:
        msg = f"Forbidden features detected: {forbidden}"
        if mode == 'paper':
            raise ValueError(f"[FAIL-FAST] {msg}")
        else:
            warnings.warn(f"[LEGACY MODE] {msg}")


# =============================================================================
# 4. Git & Version Fingerprinting
# =============================================================================

def get_git_info() -> Dict[str, Any]:
    """
    自動獲取 git 資訊
    
    規格：implementation_plan.md #1.2, #7
    
    Returns:
        包含 git_commit, git_branch, git_dirty 的字典
    """
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode().strip()
        is_dirty = len(status) > 0
        
        return {
            'git_commit': commit,
            'git_branch': branch,
            'git_dirty': is_dirty
        }
    except Exception as e:
        warnings.warn(f"Failed to get git info: {e}")
        return {
            'git_commit': 'UNKNOWN',
            'git_branch': 'UNKNOWN',
            'git_dirty': True
        }


def compute_code_fingerprint(files: List[str]) -> str:
    """
    計算關鍵程式碼檔案的 fingerprint
    
    Args:
        files: 要計算 hash 的檔案路徑列表
    
    Returns:
        16-character hash
    """
    hasher = hashlib.sha256()
    for filepath in sorted(files):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                hasher.update(f.read())
    return hasher.hexdigest()[:16]


# =============================================================================
# 5. Metadata Generation
# =============================================================================

def generate_samples_metadata(samples_df: pd.DataFrame) -> Dict[str, Any]:
    """
    生成樣本元資料
    
    規格：implementation_plan.md #1.2, #13, #19
    
    Args:
        samples_df: 必須包含 y_true, is_included, is_excluded 等欄位
    
    Returns:
        metadata字典
    """
    total = len(samples_df)
    included = samples_df['is_included'].sum() if 'is_included' in samples_df.columns else total
    excluded = samples_df['is_excluded'].sum() if 'is_excluded' in samples_df.columns else 0
    
    included_samples = samples_df[samples_df.get('is_included', True) == True]
    positives = (included_samples['y_true'] == 1).sum()
    negatives = (included_samples['y_true'] == 0).sum()
    positive_rate = positives / len(included_samples) if len(included_samples) > 0 else 0
    
    metadata = {
        'total_products': total,
        'included_products': int(included),
        'excluded_products': int(excluded),
        'positives': int(positives),
        'negatives': int(negatives),
        'positive_rate': float(positive_rate)
    }
    
    # Keyword distribution (若有 keyword 欄位)
    if 'keyword' in samples_df.columns:
        keyword_counts = included_samples['keyword'].value_counts().head(20).to_dict()
        metadata['keyword_distribution'] = keyword_counts
        metadata['num_unique_keywords'] = int(included_samples['keyword'].nunique())
        
        # Keyword entropy
        from scipy.stats import entropy
        keyword_probs = included_samples['keyword'].value_counts(normalize=True).values
        metadata['keyword_entropy'] = float(entropy(keyword_probs))
    
    return metadata


if __name__ == "__main__":
    # 簡單測試
    print("experiment_utils.py loaded successfully")
    
    # Test git info
    git_info = get_git_info()
    print(f"Git info: {git_info}")
