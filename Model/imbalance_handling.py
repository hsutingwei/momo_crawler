# -*- coding: utf-8 -*-
"""
imbalance_handling.py
類別不平衡處理工具 (Class Imbalance Handling Utilities)

實現規格參考：implementation_plan.md
- Phase 4: Consistent Scale Pos Weight (一致的 Scale Pos Weight)
- Specification #21: scale_pos_weight Calculation Basis (計算基礎)
"""

import json
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np


# =============================================================================
# Scale Pos Weight Calculation
# =============================================================================

def compute_scale_pos_weight(
    y: pd.Series,
    scope: str = 'fold_train'
) -> float:
    """
    計算 scale_pos_weight
    
    規格：implementation_plan.md #4.1, #21
    
    Args:
        y: 標籤序列
        scope: 'fold_train' | 'train_pool'
    
    Returns:
        scale_pos_weight = n_neg / n_pos
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    
    if n_pos == 0:
        raise ValueError(f"在 {scope} 中沒有正樣本! 無法計算 scale_pos_weight。")
    
    scale_pos_weight = n_neg / n_pos
    
    return float(scale_pos_weight)


def compute_scale_pos_weight_per_fold(
    splits_df: pd.DataFrame,
    y_df: pd.DataFrame,
    n_folds: int,
    scope: str = 'fold_train'
) -> Dict[str, Any]:
    """
    為每個 fold 計算 scale_pos_weight
    
    規格：implementation_plan.md #21
    
    Args:
        splits_df: Splits DataFrame (product_id, split, fold_id)
        y_df: 標籤 DataFrame (product_id, y_true)
        n_folds: Fold 數量
        scope: 'fold_train' | 'train_pool'
    
    Returns:
        imbalance_report 字典
    """
    # Merge splits with labels
    data = splits_df.merge(y_df, on='product_id', how='left')
    
    imbalance_report = {
        'scope': scope,
        'n_folds': n_folds,
        'folds': {}
    }
    
    if scope == 'fold_train':
        # 每個 fold 分別計算（train_fold = train_pool 排除該 fold）
        for fold in range(n_folds):
            fold_data = data[
                (data['split'] == 'train_pool') & 
                (data['fold_id'] != fold)
            ]
            
            n_pos = (fold_data['y_true'] == 1).sum()
            n_neg = (fold_data['y_true'] == 0).sum()
            
            if n_pos == 0:
                raise ValueError(f"Fold {fold}: 沒有正樣本!")
            
            spw = n_neg / n_pos
            
            imbalance_report['folds'][f'fold_{fold}'] = {
                'n_pos': int(n_pos),
                'n_neg': int(n_neg),
                'scale_pos_weight': float(spw),
                'positive_rate': float(n_pos / (n_pos + n_neg))
            }
    
    elif scope == 'train_pool':
        # 所有 fold 使用同一個值（整個 train_pool）
        train_pool_data = data[data['split'] == 'train_pool']
        
        n_pos = (train_pool_data['y_true'] == 1).sum()
        n_neg = (train_pool_data['y_true'] == 0).sum()
        
        if n_pos == 0:
            raise ValueError("train_pool: 沒有正樣本!")
        
        spw = n_neg / n_pos
        
        # 所有 fold 使用相同值
        for fold in range(n_folds):
            imbalance_report['folds'][f'fold_{fold}'] = {
                'n_pos': int(n_pos),
                'n_neg': int(n_neg),
                'scale_pos_weight': float(spw),
                'positive_rate': float(n_pos / (n_pos + n_neg)),
                'note': '所有 folds 使用相同值 (train_pool scope)'
            }
    
    else:
        raise ValueError(f"未知的 scope: {scope}")
    
    # 計算統計
    spw_values = [f['scale_pos_weight'] for f in imbalance_report['folds'].values()]
    imbalance_report['statistics'] = {
        'mean_scale_pos_weight': float(np.mean(spw_values)),
        'std_scale_pos_weight': float(np.std(spw_values)),
        'min_scale_pos_weight': float(np.min(spw_values)),
        'max_scale_pos_weight': float(np.max(spw_values))
    }
    
    return imbalance_report


def save_imbalance_report(
    report: Dict[str, Any],
    output_path: str
) -> None:
    """保存 imbalance report"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Imbalance report 已保存至 {output_path}")
    
    # 輸出摘要
    stats = report['statistics']
    print(f"  Scope: {report['scope']}")
    print(f"  Mean scale_pos_weight: {stats['mean_scale_pos_weight']:.2f}")
    print(f"  Std: {stats['std_scale_pos_weight']:.4f}")


def verify_imbalance_consistency(
    report1: Dict[str, Any],
    report2: Dict[str, Any],
    tolerance: float = 0.01
) -> bool:
    """
    驗證兩個 run 的 imbalance handling 是否一致
    
    用於 ablation study：確保所有 variant 使用相同的 imbalance 處理
    
    Args:
        report1: Run 1 的 imbalance_report
        report2: Run 2 的 imbalance_report
        tolerance: 允許的誤差範圍
    
    Returns:
        是否一致
    """
    # 檢查 scope 是否相同
    if report1['scope'] != report2['scope']:
        print(f"❌ Scope 不匹配: {report1['scope']} vs {report2['scope']}")
        return False
    
    # 檢查 n_folds 是否相同
    if report1['n_folds'] != report2['n_folds']:
        print(f"❌ n_folds 不匹配: {report1['n_folds']} vs {report2['n_folds']}")
        return False
    
    # 檢查每個 fold 的 scale_pos_weight 是否一致
    for fold_key in report1['folds']:
        if fold_key not in report2['folds']:
            print(f"❌ Fold 在 report2 中缺失: {fold_key}")
            return False
        
        spw1 = report1['folds'][fold_key]['scale_pos_weight']
        spw2 = report2['folds'][fold_key]['scale_pos_weight']
        
        if abs(spw1 - spw2) > tolerance:
            print(f"❌ {fold_key} 的 scale_pos_weight 不匹配: {spw1:.4f} vs {spw2:.4f}")
            return False
    
    print("✅ Run 之間的 Imbalance handling 一致")
    return True


# =============================================================================
# Imbalance Strategy Selection
# =============================================================================

def get_imbalance_config(
    imbalance_mode: str = 'scale_pos_weight',
    scope: str = 'fold_train'
) -> Dict[str, Any]:
    """
    獲取 imbalance handling 配置
    
    Args:
        imbalance_mode: 'scale_pos_weight' | 'none'
        scope: 'fold_train' | 'train_pool'
    
    Returns:
        配置字典
    """
    if imbalance_mode == 'scale_pos_weight':
        return {
            'mode': 'scale_pos_weight',
            'scope': scope,
            'description': f'使用從 {scope} 計算的 scale_pos_weight'
        }
    elif imbalance_mode == 'none':
        return {
            'mode': 'none',
            'scope': None,
            'description': '無不平衡處理'
        }
    else:
        raise ValueError(f"未知的 imbalance_mode: {imbalance_mode}")


# =============================================================================
# XGBoost Model Parameter Integration
# =============================================================================

def prepare_xgboost_params(
    base_params: Dict[str, Any],
    scale_pos_weight: Optional[float] = None
) -> Dict[str, Any]:
    """
    準備 XGBoost 參數（含 scale_pos_weight）
    
    Args:
        base_params: 基礎參數字典
        scale_pos_weight: scale_pos_weight 值
    
    Returns:
        完整參數字典
    """
    params = base_params.copy()
    
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight
    
    return params


if __name__ == "__main__":
    print("imbalance_handling.py 加載成功")
    
    # 簡單測試
    np.random.seed(42)
    
    # 創建模擬數據
    n_samples = 1000
    splits_df = pd.DataFrame({
        'product_id': range(1, n_samples + 1),
        'split': ['train_pool'] * 800 + ['test'] * 200,
        'fold_id': [i % 10 for i in range(800)] + [-1] * 200
    })
    
    # 5% positive rate
    y_df = pd.DataFrame({
        'product_id': range(1, n_samples + 1),
        'y_true': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    })
    
    print(f"\n模擬數據: {n_samples} samples, ~5% positive")
    
    # Test fold_train scope
    print("\n[測試 1] fold_train scope:")
    report1 = compute_scale_pos_weight_per_fold(
        splits_df, y_df, n_folds=10, scope='fold_train'
    )
    print(f"  Mean scale_pos_weight: {report1['statistics']['mean_scale_pos_weight']:.2f}")
    print(f"  Std: {report1['statistics']['std_scale_pos_weight']:.4f}")
    
    # Test train_pool scope
    print("\n[測試 2] train_pool scope:")
    report2 = compute_scale_pos_weight_per_fold(
        splits_df, y_df, n_folds=10, scope='train_pool'
    )
    print(f"  Mean scale_pos_weight: {report2['statistics']['mean_scale_pos_weight']:.2f}")
    print(f"  Std: {report2['statistics']['std_scale_pos_weight']:.4f}")
    print(f"  (對於 train_pool scope 應該為 0)")
    
    # Test consistency verification
    print("\n[測試 3] 一致性驗證:")
    # Same report should be consistent
    is_consistent = verify_imbalance_consistency(report2, report2)
    
    # Different scopes should NOT be consistent
    print("\n[測試 4] 不同 scopes (應失敗):")
    is_consistent = verify_imbalance_consistency(report1, report2)
