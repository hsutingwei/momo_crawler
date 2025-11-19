#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析特定 keyword 的模型表現

用法：
    python analysis/analyze_keyword_performance.py --keyword 益生菌 --predictions Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_predictions.csv --model-run Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_model_run_v1.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database import DatabaseConfig


def load_predictions(predictions_path: str) -> pd.DataFrame:
    """載入預測結果 CSV"""
    print(f"[1/5] 載入預測結果: {predictions_path}")
    df = pd.read_csv(predictions_path)
    print(f"  總樣本數: {len(df)}")
    print(f"  欄位: {list(df.columns)}")
    return df


def load_products_from_db(conn, product_ids: List[int]) -> pd.DataFrame:
    """從資料庫載入商品資訊"""
    print(f"[2/5] 從資料庫載入商品資訊 (共 {len(product_ids)} 個商品)")
    
    # 將 product_ids 轉為 tuple 以便 SQL IN 查詢
    product_ids_tuple = tuple(product_ids)
    
    query = """
        SELECT id AS product_id, name AS product_name, keyword
        FROM products
        WHERE id = ANY(%s)
    """
    
    df = pd.read_sql(query, conn, params=[product_ids])
    print(f"  成功載入 {len(df)} 筆商品資料")
    return df


def filter_by_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    """篩選特定 keyword 的樣本"""
    print(f"[3/5] 篩選 keyword = '{keyword}' 的樣本")
    
    # 如果 predictions 中已經有 keyword 欄位，可以直接篩選
    if 'keyword' in df.columns:
        filtered = df[df['keyword'] == keyword].copy()
    else:
        # 如果沒有 keyword 欄位，需要從 products 表 merge
        print("  警告：predictions 中沒有 keyword 欄位，需要從 products 表 merge")
        filtered = df.copy()
    
    print(f"  篩選後樣本數: {len(filtered)}")
    return filtered


def calculate_metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    """計算在指定 threshold 下的指標"""
    y_pred = (y_score >= threshold).astype(int)
    
    precision_1 = precision_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)[1]
    recall_1 = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)[1]
    f1_1 = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)[1]
    
    return {
        'precision_1': float(precision_1),
        'recall_1': float(recall_1),
        'f1_1': float(f1_1)
    }


def analyze_keyword_performance(
    predictions_path: str,
    model_run_path: str,
    keyword: str,
    output_dir: str = "analysis"
):
    """分析特定 keyword 的模型表現"""
    
    # 1. 載入預測結果
    df_pred = load_predictions(predictions_path)
    
    # 檢查必要欄位
    required_cols = ['product_id', 'y_true', 'y_score']
    missing_cols = [col for col in required_cols if col not in df_pred.columns]
    if missing_cols:
        raise ValueError(f"預測結果檔案缺少必要欄位: {missing_cols}")
    
    # 如果有 y_pred 欄位，使用它；否則需要從 y_score 計算
    if 'y_pred' not in df_pred.columns:
        df_pred['y_pred'] = (df_pred['y_score'] >= 0.5).astype(int)
    
    # 2. 從資料庫載入商品資訊
    db_config = DatabaseConfig()
    conn = db_config.get_connection()
    
    try:
        # 取得所有唯一的 product_id
        unique_product_ids = df_pred['product_id'].unique().tolist()
        df_products = load_products_from_db(conn, unique_product_ids)
        
        # 3. 篩選特定 keyword（如果 predictions 中已有 keyword 欄位）
        if 'keyword' in df_pred.columns:
            print(f"[3/5] 從預測結果中篩選 keyword = '{keyword}'")
            df_keyword = df_pred[df_pred['keyword'] == keyword].copy()
            print(f"  篩選後樣本數: {len(df_keyword)}")
            
            # 只 merge product_name
            df_keyword = df_keyword.merge(
                df_products[['product_id', 'product_name']],
                on='product_id',
                how='left'
            )
        else:
            # 如果沒有 keyword 欄位，需要從 products 表 merge
            print(f"[3/5] 合併預測結果與商品資訊")
            df_merged = df_pred.merge(
                df_products[['product_id', 'product_name', 'keyword']],
                on='product_id',
                how='left'
            )
            
            # 篩選特定 keyword
            df_keyword = filter_by_keyword(df_merged, keyword)
        
        if len(df_keyword) == 0:
            print(f"  警告：沒有找到 keyword = '{keyword}' 的樣本")
            return
        
        # 5. 計算基本統計
        print(f"[4/5] 計算指標")
        n_samples = len(df_keyword)
        n_pos = int((df_keyword['y_true'] == 1).sum())
        n_neg = int((df_keyword['y_true'] == 0).sum())
        pos_rate = n_pos / n_samples if n_samples > 0 else 0.0
        
        print(f"  樣本數: {n_samples}")
        print(f"  正類 (y=1): {n_pos}")
        print(f"  負類 (y=0): {n_neg}")
        print(f"  正類比例: {pos_rate:.4f}")
        
        # 6. 計算 threshold = 0.5 時的指標
        y_true = df_keyword['y_true'].values
        y_score = df_keyword['y_score'].values
        
        metrics_05 = calculate_metrics_at_threshold(y_true, y_score, threshold=0.5)
        print(f"\n  [Threshold = 0.5]")
        print(f"    Precision_1: {metrics_05['precision_1']:.4f}")
        print(f"    Recall_1: {metrics_05['recall_1']:.4f}")
        print(f"    F1_1: {metrics_05['f1_1']:.4f}")
        
        # 7. 讀取最佳 threshold
        print(f"[5/5] 讀取最佳 threshold")
        with open(model_run_path, 'r', encoding='utf-8-sig') as f:
            model_run = json.load(f)
        
        threshold_search = model_run.get('metrics', {}).get('threshold_search', {})
        best_threshold = threshold_search.get('chosen_threshold', 0.5)
        chosen_for = threshold_search.get('chosen_for', 'f1_1')
        
        print(f"  最佳 threshold: {best_threshold} (選擇依據: {chosen_for})")
        
        # 8. 計算最佳 threshold 下的指標
        metrics_best = calculate_metrics_at_threshold(y_true, y_score, threshold=best_threshold)
        print(f"\n  [Threshold = {best_threshold}]")
        print(f"    Precision_1: {metrics_best['precision_1']:.4f}")
        print(f"    Recall_1: {metrics_best['recall_1']:.4f}")
        print(f"    F1_1: {metrics_best['f1_1']:.4f}")
        
        # 9. Top 20 最高分樣本
        print(f"\n=== Top 20 最高 y_score 樣本 ===")
        df_top20 = df_keyword.nlargest(20, 'y_score')[
            ['product_id', 'product_name', 'y_true', 'y_score']
        ].copy()
        
        # 計算 TP/FP
        y_pred_best = (df_top20['y_score'].values >= best_threshold).astype(int)
        tp_count = int(((df_top20['y_true'] == 1) & (y_pred_best == 1)).sum())
        fp_count = int(((df_top20['y_true'] == 0) & (y_pred_best == 1)).sum())
        tn_count = int(((df_top20['y_true'] == 0) & (y_pred_best == 0)).sum())
        fn_count = int(((df_top20['y_true'] == 1) & (y_pred_best == 0)).sum())
        
        print(f"  TP: {tp_count}, FP: {fp_count}, TN: {tn_count}, FN: {fn_count}")
        print(f"\n  詳細列表:")
        for idx, row in df_top20.iterrows():
            pred_label = "Y" if y_pred_best[df_top20.index.get_loc(idx)] == 1 else "N"
            true_label = "Y" if row['y_true'] == 1 else "N"
            status = "TP" if (row['y_true'] == 1 and y_pred_best[df_top20.index.get_loc(idx)] == 1) else \
                     "FP" if (row['y_true'] == 0 and y_pred_best[df_top20.index.get_loc(idx)] == 1) else \
                     "TN" if (row['y_true'] == 0 and y_pred_best[df_top20.index.get_loc(idx)] == 0) else "FN"
            product_name = str(row['product_name']) if pd.notna(row['product_name']) else "N/A"
            print(f"    {row['product_id']:>10} | {product_name[:40]:<40} | "
                  f"y_true={true_label} pred={pred_label} ({status}) | score={row['y_score']:.4f}")
        
        # 10. 所有 FN 樣本
        print(f"\n=== 所有 FN 樣本 (y_true=1, y_pred=0 at threshold={best_threshold}) ===")
        y_pred_best_all = (df_keyword['y_score'].values >= best_threshold).astype(int)
        df_fn = df_keyword[
            (df_keyword['y_true'] == 1) & (y_pred_best_all == 0)
        ][['product_id', 'product_name', 'y_true', 'y_score']].copy()
        
        print(f"  FN 樣本數: {len(df_fn)}")
        if len(df_fn) > 0:
            print(f"  詳細列表:")
            for idx, row in df_fn.iterrows():
                product_name = str(row['product_name']) if pd.notna(row['product_name']) else "N/A"
                print(f"    {row['product_id']:>10} | {product_name[:40]:<40} | score={row['y_score']:.4f}")
        else:
            print("  無 FN 樣本")
        
        # 11. 輸出 JSON 結果
        os.makedirs(output_dir, exist_ok=True)
        output_json_path = os.path.join(output_dir, f"keyword_{keyword}_performance.json")
        
        result = {
            'keyword': keyword,
            'n_samples': int(n_samples),
            'n_pos': int(n_pos),
            'n_neg': int(n_neg),
            'pos_rate': float(pos_rate),
            'precision_1_05': metrics_05['precision_1'],
            'recall_1_05': metrics_05['recall_1'],
            'f1_1_05': metrics_05['f1_1'],
            'best_threshold': float(best_threshold),
            'chosen_for': chosen_for,
            'precision_1_best': metrics_best['precision_1'],
            'recall_1_best': metrics_best['recall_1'],
            'f1_1_best': metrics_best['f1_1'],
            'top20_tp': int(tp_count),
            'top20_fp': int(fp_count),
            'top20_tn': int(tn_count),
            'top20_fn': int(fn_count),
            'total_fn': int(len(df_fn))
        }
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 結果已儲存 ===")
        print(f"  JSON 檔案: {output_json_path}")
        
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='分析特定 keyword 的模型表現')
    parser.add_argument(
        '--keyword',
        type=str,
        required=True,
        help='要分析的 keyword（例如：益生菌）'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        default='Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_predictions.csv',
        help='預測結果 CSV 檔案路徑'
    )
    parser.add_argument(
        '--model-run',
        type=str,
        default='Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_model_run_v1.json',
        help='model_run_v1.json 檔案路徑'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='analysis',
        help='輸出目錄（預設：analysis）'
    )
    
    args = parser.parse_args()
    
    analyze_keyword_performance(
        predictions_path=args.predictions,
        model_run_path=args.model_run,
        keyword=args.keyword,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

