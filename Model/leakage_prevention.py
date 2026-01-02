# -*- coding: utf-8 -*-
"""
leakage_prevention.py
Data Leakage Prevention Utilities

實現規格參考：implementation_plan.md
- Phase 3: TF-IDF Strict Isolation
- Phase 3: Automated Leakage Checks (Fail-Fast)
"""

import json
import warnings
from typing import Dict, List, Set, Optional, Tuple, Any

import pandas as pd
import numpy as np


# =============================================================================
# TF-IDF Strict Isolation
# =============================================================================

def verify_tfidf_isolation(
    train_product_ids: Set[int],
    test_product_ids: Set[int],
    tfidf_source_product_ids: Set[int]
) -> Dict[str, Any]:
    """
    驗證 TF-IDF vocabulary 是否只來自 train set
    
    規格：implementation_plan.md #3.2
    
    Args:
        train_product_ids: 訓練集 product IDs
        test_product_ids: 測試集 product IDs
        tfidf_source_product_ids: TF-IDF vocab 來源 product IDs
    
    Returns:
        檢查結果字典 {'passed': bool, 'leakage_count': int, 'details': str}
    """
    # 檢查是否有 test product IDs 混入
    leaked_ids = tfidf_source_product_ids & test_product_ids
    
    result = {
        'check_name': 'tfidf_vocab_isolation',
        'passed': len(leaked_ids) == 0,
        'leakage_count': len(leaked_ids),
        'train_source_count': len(tfidf_source_product_ids & train_product_ids),
        'test_source_count': len(leaked_ids),
        'details': f'TF-IDF vocab built from {len(tfidf_source_product_ids)} products'
    }
    
    if not result['passed']:
        result['error'] = f"❌ LEAKAGE: {len(leaked_ids)} test products in TF-IDF vocab source!"
        result['leaked_sample'] = list(leaked_ids)[:5]
    
    return result


def verify_clip_bounds_isolation(
    train_product_ids: Set[int],
    test_product_ids: Set[int],
    clip_source_product_ids: Optional[Set[int]] = None
) -> Dict[str, Any]:
    """
    驗證 Clip bounds 是否只來自 train set
    
    規格：implementation_plan.md #3.3
    
    Args:
        train_product_ids: 訓練集 product IDs
        test_product_ids: 測試集 product IDs
        clip_source_product_ids: Clip 計算來源 product IDs
    
    Returns:
        檢查結果字典
    """
    if clip_source_product_ids is None:
        return {
            'check_name': 'clip_bounds_isolation',
            'passed': True,
            'skipped': True,
            'details': 'No clipping performed (skip check)'
        }
    
    leaked_ids = clip_source_product_ids & test_product_ids
    
    result = {
        'check_name': 'clip_bounds_isolation',
        'passed': len(leaked_ids) == 0,
        'leakage_count': len(leaked_ids),
        'train_source_count': len(clip_source_product_ids & train_product_ids),
        'test_source_count': len(leaked_ids),
        'details': f'Clip bounds computed from {len(clip_source_product_ids)} products'
    }
    
    if not result['passed']:
        result['error'] = f"❌ LEAKAGE: {len(leaked_ids)} test products in clip source!"
        result['leaked_sample'] = list(leaked_ids)[:5]
    
    return result


def verify_impute_values_isolation(
    train_product_ids: Set[int],
    test_product_ids: Set[int],
    impute_source_product_ids: Optional[Set[int]] = None
) -> Dict[str, Any]:
    """
    驗證 Imputation values 是否只來自 train set
    
    規格：implementation_plan.md #3.3
    """
    if impute_source_product_ids is None:
        return {
            'check_name': 'impute_values_isolation',
            'passed': True,
            'skipped': True,
            'details': 'No imputation performed (skip check)'
        }
    
    leaked_ids = impute_source_product_ids & test_product_ids
    
    result = {
        'check_name': 'impute_values_isolation',
        'passed': len(leaked_ids) == 0,
        'leakage_count': len(leaked_ids),
        'train_source_count': len(impute_source_product_ids & train_product_ids),
        'test_source_count': len(leaked_ids),
        'details': f'Imputation computed from {len(impute_source_product_ids)} products'
    }
    
    if not result['passed']:
        result['error'] = f"❌ LEAKAGE: {len(leaked_ids)} test products in imputation source!"
        result['leaked_sample'] = list(leaked_ids)[:5]
    
    return result


def verify_scaler_isolation(
    train_product_ids: Set[int],
    test_product_ids: Set[int],
    scaler_source_product_ids: Optional[Set[int]] = None
) -> Dict[str, Any]:
    """
    驗證 Scaler 是否只在 train set 上 fit
    
    規格：implementation_plan.md #3.3
    """
    if scaler_source_product_ids is None:
        return {
            'check_name': 'scaler_isolation',
            'passed': True,
            'skipped': True,
            'details': 'No scaling performed (skip check)'
        }
    
    leaked_ids = scaler_source_product_ids & test_product_ids
    
    result = {
        'check_name': 'scaler_isolation',
        'passed': len(leaked_ids) == 0,
        'leakage_count': len(leaked_ids),
        'train_source_count': len(scaler_source_product_ids & train_product_ids),
        'test_source_count': len(leaked_ids),
        'details': f'Scaler fit on {len(scaler_source_product_ids)} products'
    }
    
    if not result['passed']:
        result['error'] = f"❌ LEAKAGE: {len(leaked_ids)} test products in scaler source!"
        result['leaked_sample'] = list(leaked_ids)[:5]
    
    return result


def verify_no_test_in_train_fold(
    train_fold_product_ids: Set[int],
    test_product_ids: Set[int]
) -> Dict[str, Any]:
    """
    驗證 test set 沒有混入 train fold
    
    規格：implementation_plan.md #3.3
    """
    leaked_ids = train_fold_product_ids & test_product_ids
    
    result = {
        'check_name': 'test_isolation_from_train',
        'passed': len(leaked_ids) == 0,
        'leakage_count': len(leaked_ids),
        'train_fold_count': len(train_fold_product_ids),
        'test_count': len(test_product_ids),
        'details': f'Train fold: {len(train_fold_product_ids)}, Test: {len(test_product_ids)}'
    }
    
    if not result['passed']:
        result['error'] = f"❌ CRITICAL LEAKAGE: {len(leaked_ids)} test products in train fold!"
        result['leaked_sample'] = list(leaked_ids)[:10]  # 顯示更多（這是嚴重錯誤）
    
    return result


# =============================================================================
# Comprehensive Leakage Check
# =============================================================================

def verify_no_leakage(
    splits_df: pd.DataFrame,
    current_fold: int,
    preprocess_fit_scope: str,
    tfidf_source_ids: Optional[Set[int]] = None,
    clip_source_ids: Optional[Set[int]] = None,
    impute_source_ids: Optional[Set[int]] = None,
    scaler_source_ids: Optional[Set[int]] = None,
    fail_fast: bool = True
) -> Dict[str, Any]:
    """
    執行完整的 leakage 檢查（5個關鍵點）
    
    規格：implementation_plan.md #3.3
    
    Args:
        splits_df: Splits DataFrame (product_id, split, fold_id)
        current_fold: 當前訓練的 fold (用於 K-fold CV)
        preprocess_fit_scope: 'train_fold_only' | 'train_only' | 'full_train'
        tfidf_source_ids: TF-IDF vocab 來源 product IDs
        clip_source_ids: Clip bounds 來源 product IDs
        impute_source_ids: Imputation 來源 product IDs
        scaler_source_ids: Scaler fit 來源 product IDs
        fail_fast: 若發現 leakage 是否立即拋出錯誤
    
    Returns:
        完整檢查報告
    """
    # 定義 train fold 和 test set
    if preprocess_fit_scope == 'train_fold_only':
        # Train fold = train_pool 中排除當前 val fold
        train_fold_df = splits_df[
            (splits_df['split'] == 'train_pool') & 
            (splits_df['fold_id'] != current_fold)
        ]
    elif preprocess_fit_scope == 'train_only':
        # Train only = 整個 train_pool (80%)
        train_fold_df = splits_df[splits_df['split'] == 'train_pool']
    else:
        raise ValueError(f"Unknown preprocess_fit_scope: {preprocess_fit_scope}")
    
    train_fold_ids = set(train_fold_df['product_id'])
    test_ids = set(splits_df[splits_df['split'] == 'test']['product_id'])
    
    # 執行 5 個檢查
    checks = []
    
    # Check 1: Test 沒混入 train fold (最關鍵)
    check1 = verify_no_test_in_train_fold(train_fold_ids, test_ids)
    checks.append(check1)
    
    # Check 2: TF-IDF isolation
    if tfidf_source_ids is not None:
        check2 = verify_tfidf_isolation(train_fold_ids, test_ids, tfidf_source_ids)
        checks.append(check2)
    
    # Check 3: Clip isolation
    if clip_source_ids is not None:
        check3 = verify_clip_bounds_isolation(train_fold_ids, test_ids, clip_source_ids)
        checks.append(check3)
    
    # Check 4: Impute isolation
    if impute_source_ids is not None:
        check4 = verify_impute_values_isolation(train_fold_ids, test_ids, impute_source_ids)
        checks.append(check4)
    
    # Check 5: Scaler isolation
    if scaler_source_ids is not None:
        check5 = verify_scaler_isolation(train_fold_ids, test_ids, scaler_source_ids)
        checks.append(check5)
    
    # 彙總結果
    all_passed = all(check['passed'] for check in checks)
    failed_checks = [check for check in checks if not check['passed']]
    
    report = {
        'all_passed': all_passed,
        'total_checks': len(checks),
        'passed_checks': len(checks) - len(failed_checks),
        'failed_checks': len(failed_checks),
        'preprocess_fit_scope': preprocess_fit_scope,
        'current_fold': current_fold,
        'train_fold_size': len(train_fold_ids),
        'test_size': len(test_ids),
        'checks': checks
    }
    
    # Fail-fast
    if not all_passed and fail_fast:
        error_msgs = [check['error'] for check in failed_checks if 'error' in check]
        raise ValueError(
            f"❌ LEAKAGE DETECTED! {len(failed_checks)} check(s) failed:\n" +
            "\n".join(error_msgs)
        )
    
    return report


def save_leakage_report(report: Dict[str, Any], output_path: str) -> None:
    """保存 leakage 檢查報告"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    if report['all_passed']:
        print(f"✅ Leakage check PASSED - Report saved to {output_path}")
    else:
        print(f"❌ Leakage check FAILED - Report saved to {output_path}")


# =============================================================================
# Metadata Generation for Leakage Prevention
# =============================================================================

def generate_tfidf_metadata(
    vocab: Dict[str, int],
    source_product_ids: Set[int],
    selection_method: str = 'df',
    top_n: Optional[int] = None
) -> Dict[str, Any]:
    """
    生成 TF-IDF vocabulary metadata
    
    規格：implementation_plan.md #22 (TF-IDF Top-N Selection Method)
    
    Args:
        vocab: TF-IDF vocabulary dictionary
        source_product_ids: 來源 product IDs
        selection_method: 'tf' (term frequency) | 'df' (document frequency)
        top_n: 若使用 top-N，記錄 N 值
    
    Returns:
        metadata 字典
    """
    metadata = {
        'selection_method': selection_method,
        'vocab_size': len(vocab),
        'source_product_count': len(source_product_ids),
        'source_doc_count': len(source_product_ids),  # 對 product-level 來說相同
    }
    
    if top_n is not None:
        metadata['top_n'] = top_n
        metadata['is_filtered'] = True
    else:
        metadata['is_filtered'] = False
    
    return metadata


if __name__ == "__main__":
    print("leakage_prevention.py loaded successfully")
    
    # 簡單測試
    np.random.seed(42)
    
    # 模擬 splits
    splits_df = pd.DataFrame({
        'product_id': range(1, 101),
        'split': ['train_pool'] * 80 + ['test'] * 20,
        'fold_id': [i % 10 for i in range(80)] + [-1] * 20
    })
    
    train_ids = set(splits_df[splits_df['split'] == 'train_pool']['product_id'])
    test_ids = set(splits_df[splits_df['split'] == 'test']['product_id'])
    
    print(f"\nTest data: {len(train_ids)} train, {len(test_ids)} test")
    
    # 測試：正常情況（無 leakage）
    print("\n[Test 1] No leakage (should pass)")
    train_fold_0 = set(splits_df[
        (splits_df['split'] == 'train_pool') & (splits_df['fold_id'] != 0)
    ]['product_id'])
    
    report = verify_no_leakage(
        splits_df,
        current_fold=0,
        preprocess_fit_scope='train_fold_only',
        tfidf_source_ids=train_fold_0,
        fail_fast=False
    )
    print(f"Result: {'✅ PASSED' if report['all_passed'] else '❌ FAILED'}")
    
    # 測試：異常情況（有 leakage）
    print("\n[Test 2] With leakage (should fail)")
    leaked_source = train_fold_0 | set(list(test_ids)[:5])  # 混入 5 個 test IDs
    
    try:
        report = verify_no_leakage(
            splits_df,
            current_fold=0,
            preprocess_fit_scope='train_fold_only',
            tfidf_source_ids=leaked_source,
            fail_fast=True  # 應該拋出錯誤
        )
    except ValueError as e:
        print(f"✅ Correctly caught leakage: {str(e)[:100]}...")
