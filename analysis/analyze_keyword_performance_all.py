#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用版 per-keyword 模型表現分析工具

【用途說明】
- 這是一個離線評估工具，不會影響訓練流程
- 從已完成的 run 預測結果中，自動分析每個 keyword 的表現
- 結果可用於：
  1. Dashboard 中顯示「各品類難度比較」
  2. 論文中的「per-keyword performance」分析
  3. 識別哪些 keyword 的預測難度較高，需要特別關注

用法：
    python analysis/analyze_keyword_performance_all.py \
        --predictions Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_predictions.csv \
        --model-run Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_model_run_v1.json \
        --output-dir analysis
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.database import DatabaseConfig


# 【設計理由】預設的 keyword blacklist
# 這些 keyword 在分析時會被排除，因為：
# - 「口罩」：COVID-19 期間銷售模式極度異常，不適合作為一般商品分析
# 未來若要調整，請修改此常數
DEFAULT_KEYWORD_BLACKLIST = ["口罩"]


def normalize_keyword_encoding(kw: str) -> str:
    """
    修正 keyword 的編碼問題
    
    【設計理由】確保比對時使用正確的 UTF-8 編碼
    常見問題：Windows 系統或某些終端機編碼導致「口罩」變成 "??蔗"
    """
    if kw == "??蔗" or kw == "??" or (isinstance(kw, bytes) and b'\xbf\xbf' in kw):
        return "口罩"
    return kw


def load_predictions(predictions_path: str) -> pd.DataFrame:
    """載入預測結果 CSV"""
    print(f"[1/4] 載入預測結果: {predictions_path}")
    df = pd.read_csv(predictions_path)
    print(f"  總樣本數: {len(df)}")
    print(f"  欄位: {list(df.columns)}")
    return df


def load_products_from_db(conn, product_ids: List[int]) -> pd.DataFrame:
    """從資料庫載入商品資訊"""
    print(f"[2/4] 從資料庫載入商品資訊 (共 {len(product_ids)} 個商品)")
    
    query = """
        SELECT id AS product_id, name AS product_name, keyword
        FROM products
        WHERE id = ANY(%s)
    """
    
    df = pd.read_sql(query, conn, params=[product_ids])
    print(f"  成功載入 {len(df)} 筆商品資料")
    return df


def load_model_run(model_run_path: str) -> Dict:
    """載入 model_run_v1.json 以取得最佳 threshold"""
    print(f"[3/4] 載入 model_run JSON: {model_run_path}")
    with open(model_run_path, 'r', encoding='utf-8-sig') as f:
        model_run = json.load(f)
    
    threshold_search = model_run.get('metrics', {}).get('threshold_search', {})
    best_threshold = threshold_search.get('chosen_threshold', 0.5)
    chosen_for = threshold_search.get('chosen_for', 'f1_1')
    
    print(f"  最佳 threshold: {best_threshold} (選擇依據: {chosen_for})")
    return model_run, best_threshold


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


def analyze_all_keywords(
    predictions_path: str,
    model_run_path: str,
    output_dir: str = "analysis",
    keyword_blacklist: Optional[List[str]] = None,
    min_pos_samples: int = 5,
    min_total_samples: int = 30,
    output_individual_json: bool = False
):
    """
    分析所有 keyword 的模型表現
    
    Args:
        predictions_path: 預測結果 CSV 檔案路徑
        model_run_path: model_run_v1.json 檔案路徑
        output_dir: 輸出目錄
        keyword_blacklist: 要排除的 keyword 列表（預設為 ["口罩"]）
        min_pos_samples: 每個 keyword 至少需要多少正類樣本才視為可評估（預設 5）
        min_total_samples: 每個 keyword 至少需要多少總樣本才視為可評估（預設 30）
        output_individual_json: 是否為每個 keyword 輸出獨立的 JSON 檔案
    """
    
    # 1. 載入預測結果
    df_pred = load_predictions(predictions_path)
    
    # 檢查必要欄位
    required_cols = ['product_id', 'y_true', 'y_score']
    missing_cols = [col for col in required_cols if col not in df_pred.columns]
    if missing_cols:
        raise ValueError(f"預測結果檔案缺少必要欄位: {missing_cols}")
    
    # 2. 從資料庫載入商品資訊
    db_config = DatabaseConfig()
    conn = db_config.get_connection()
    
    try:
        # 取得所有唯一的 product_id
        unique_product_ids = df_pred['product_id'].unique().tolist()
        df_products = load_products_from_db(conn, unique_product_ids)
        
        # 3. Merge predictions 和 products
        print(f"[3/4] 合併預測結果與商品資訊")
        df_merged = df_pred.merge(
            df_products[['product_id', 'product_name', 'keyword']],
            on='product_id',
            how='left'
        )
        
        # 如果 predictions 中已有 keyword 欄位，優先使用（但需要 merge product_name）
        if 'keyword' in df_pred.columns:
            # 保留 predictions 中的 keyword，但補充 product_name
            df_merged = df_merged.drop(columns=['keyword'], errors='ignore')
            df_merged = df_merged.merge(
                df_products[['product_id', 'product_name', 'keyword']],
                on='product_id',
                how='left'
            )
        
        # 4. 載入 model_run 以取得最佳 threshold
        model_run, best_threshold = load_model_run(model_run_path)
        
        # 5. 取得所有 keyword（排除 blacklist）
        if keyword_blacklist is None:
            keyword_blacklist = DEFAULT_KEYWORD_BLACKLIST
        
        # 修正 blacklist 中的編碼問題
        normalized_blacklist = [normalize_keyword_encoding(kw) for kw in keyword_blacklist]
        
        all_keywords = df_merged['keyword'].dropna().unique().tolist()
        # 排除 blacklist 中的 keyword
        keywords_to_analyze = [kw for kw in all_keywords if normalize_keyword_encoding(kw) not in normalized_blacklist]
        
        print(f"[4/4] 分析 {len(keywords_to_analyze)} 個 keyword（已排除 {len(normalized_blacklist)} 個 blacklist keyword）")
        
        # 6. 對每個 keyword 計算指標
        results = []
        individual_results = {}
        
        for keyword in sorted(keywords_to_analyze):
            df_keyword = df_merged[df_merged['keyword'] == keyword].copy()
            
            if len(df_keyword) == 0:
                continue
            
            n_samples = len(df_keyword)
            n_pos = int((df_keyword['y_true'] == 1).sum())
            n_neg = int((df_keyword['y_true'] == 0).sum())
            pos_rate = n_pos / n_samples if n_samples > 0 else 0.0
            
            # 檢查是否可評估
            valid_for_keyword_eval = (n_pos >= min_pos_samples) and (n_samples >= min_total_samples)
            
            # 計算指標（只有在可評估時才計算）
            if valid_for_keyword_eval:
                y_true = df_keyword['y_true'].values
                y_score = df_keyword['y_score'].values
                metrics_best = calculate_metrics_at_threshold(y_true, y_score, threshold=best_threshold)
            else:
                metrics_best = {
                    'precision_1': None,
                    'recall_1': None,
                    'f1_1': None
                }
            
            result = {
                'keyword': keyword,
                'n_samples': int(n_samples),
                'n_pos': int(n_pos),
                'n_neg': int(n_neg),
                'pos_rate': float(pos_rate),
                'best_threshold': float(best_threshold),
                'precision_1_best': metrics_best['precision_1'],
                'recall_1_best': metrics_best['recall_1'],
                'f1_1_best': metrics_best['f1_1'],
                'valid_for_keyword_eval': valid_for_keyword_eval
            }
            
            results.append(result)
            individual_results[keyword] = result
            
            # 輸出每個 keyword 的獨立 JSON（如果啟用）
            if output_individual_json and valid_for_keyword_eval:
                os.makedirs(output_dir, exist_ok=True)
                json_path = os.path.join(output_dir, f"keyword_{keyword}_performance.json")
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 7. 輸出彙總 CSV
        os.makedirs(output_dir, exist_ok=True)
        df_summary = pd.DataFrame(results)
        summary_path = os.path.join(output_dir, "keyword_performance_summary.csv")
        df_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"\n=== 分析完成 ===")
        print(f"  總共分析 {len(results)} 個 keyword")
        print(f"  可評估的 keyword: {sum(1 for r in results if r['valid_for_keyword_eval'])}")
        print(f"  彙總 CSV: {summary_path}")
        
        # 8. 輸出統計摘要
        print(f"\n=== 統計摘要 ===")
        valid_results = [r for r in results if r['valid_for_keyword_eval']]
        if valid_results:
            print(f"  可評估 keyword 的平均指標：")
            avg_precision = np.mean([r['precision_1_best'] for r in valid_results if r['precision_1_best'] is not None])
            avg_recall = np.mean([r['recall_1_best'] for r in valid_results if r['recall_1_best'] is not None])
            avg_f1 = np.mean([r['f1_1_best'] for r in valid_results if r['f1_1_best'] is not None])
            print(f"    Precision_1 (平均): {avg_precision:.4f}")
            print(f"    Recall_1 (平均): {avg_recall:.4f}")
            print(f"    F1_1 (平均): {avg_f1:.4f}")
        
        # 顯示表現最好和最差的 keyword（僅限可評估的）
        if valid_results:
            best_f1 = max(valid_results, key=lambda x: x['f1_1_best'] if x['f1_1_best'] is not None else -1)
            worst_f1 = min(valid_results, key=lambda x: x['f1_1_best'] if x['f1_1_best'] is not None else float('inf'))
            print(f"\n  表現最好的 keyword: {best_f1['keyword']} (F1_1={best_f1['f1_1_best']:.4f}, n_samples={best_f1['n_samples']})")
            print(f"  表現最差的 keyword: {worst_f1['keyword']} (F1_1={worst_f1['f1_1_best']:.4f}, n_samples={worst_f1['n_samples']})")
        
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='分析所有 keyword 的模型表現')
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
    parser.add_argument(
        '--keyword-blacklist',
        type=str,
        default=None,
        help='要排除的 keyword 列表（逗號分隔），預設為 "口罩"'
    )
    parser.add_argument(
        '--min-pos-samples',
        type=int,
        default=5,
        help='每個 keyword 至少需要多少正類樣本才視為可評估（預設 5）'
    )
    parser.add_argument(
        '--min-total-samples',
        type=int,
        default=30,
        help='每個 keyword 至少需要多少總樣本才視為可評估（預設 30）'
    )
    parser.add_argument(
        '--output-individual-json',
        action='store_true',
        help='是否為每個 keyword 輸出獨立的 JSON 檔案'
    )
    
    args = parser.parse_args()
    
    keyword_blacklist = None
    if args.keyword_blacklist:
        keyword_blacklist = [kw.strip() for kw in args.keyword_blacklist.split(",")]
    
    analyze_all_keywords(
        predictions_path=args.predictions,
        model_run_path=args.model_run,
        output_dir=args.output_dir,
        keyword_blacklist=keyword_blacklist,
        min_pos_samples=args.min_pos_samples,
        min_total_samples=args.min_total_samples,
        output_individual_json=args.output_individual_json
    )


if __name__ == '__main__':
    main()

