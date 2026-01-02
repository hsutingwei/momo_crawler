# -*- coding: utf-8 -*-
"""
view_results.py
ML Pipeline 結果查看器

功能：
1. 自動讀取 runs/ 目錄下的實驗
2. 解析 metrics.json 顯示分數
3. 讀取 .parquet 預測檔並顯示摘要
4. 顯示 Top 特徵重要性
"""

import os
import argparse
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime

RUNS_DIR = 'runs'

def get_latest_run():
    if not os.path.exists(RUNS_DIR):
        return None
    
    # 獲取所有 run 目錄
    runs = [d for d in glob.glob(os.path.join(RUNS_DIR, '*')) if os.path.isdir(d)]
    if not runs:
        return None
    
    # 按修改時間排序（最新的在最後）
    latest_run = max(runs, key=os.path.getmtime)
    return latest_run

def format_metric(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def view_run(run_path):
    run_id = os.path.basename(run_path)
    print(f"\n{'='*60}")
    print(f"📊 實驗結果報告: {run_id}")
    print(f"📁 路徑: {run_path}")
    print(f"{'='*60}")

    # 1. 讀取 Config
    config_path = os.path.join(run_path, 'run_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(f"\n🔍 [配置摘要]")
            print(f"  - Date Cutoff: {config.get('date_cutoff', 'N/A')}")
            print(f"  - Feature Set: {config.get('feature_set', 'N/A')}")
            print(f"  - Model Type: {config.get('model_type', 'N/A')}")
            print(f"  - CV Strategy: {config.get('cv_strategy', 'N/A')}")

    # 2. 讀取 Metrics
    metrics_path = os.path.join(run_path, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            print(f"\n📈 [評估指標 (Metrics)]")
            
            # 顯示 Test Set 指標
            test_metrics = metrics.get('test', {})
            print(f"  ▶️ Test Set:")
            for k, v in test_metrics.items():
                print(f"    - {k:<15}: {format_metric(v)}")
            
            # 顯示 OOF 指標
            oof_metrics = metrics.get('oof_aggregate', {})
            print(f"  ▶️ OOF (CV Aggregate):")
            for k, v in oof_metrics.items():
                print(f"    - {k:<15}: {format_metric(v)}")

    # 3. 讀取 Feature Importance
    fi_path = os.path.join(run_path, 'feature_importance.csv')
    if os.path.exists(fi_path):
        try:
            fi_df = pd.read_csv(fi_path)
            print(f"\n⭐ [特徵重要性 Top 10]")
            print(fi_df.head(10).to_string(index=False))
        except Exception as e:
            print(f"  (讀取失敗: {e})")

    # 4. 讀取 Predictions (Test)
    test_pred_path = os.path.join(run_path, 'predictions_test.parquet')
    if os.path.exists(test_pred_path):
        try:
            print(f"\n🔮 [預測分佈 (Test Set)]讀取中: {test_pred_path}...")
            df = pd.read_parquet(test_pred_path)
            
            n_samples = len(df)
            n_pos = df['y_true'].sum()
            pos_rate = n_pos / n_samples
            
            # 預測統計
            avg_prob = df['y_prob'].mean()
            pred_pos = df['y_pred'].sum()
            
            print(f"  - 樣本數: {n_samples}")
            print(f"  - 真實正樣本: {n_pos} ({pos_rate:.2%})")
            print(f"  - 平均預測機率: {avg_prob:.4f}")
            print(f"  - 預測正樣本數 (Threshold): {pred_pos}")
            
            print(f"\n  📝 [預測樣本預覽]")
            print(df[['product_id', 'y_true', 'y_prob', 'y_pred']].head(5).to_string(index=False))
        except Exception as e:
            print(f"  (讀取 Parquet 失敗: {e})")
    
    # 5. 讀取 Predictions (OOF)
    oof_pred_path = os.path.join(run_path, 'predictions_oof.parquet')
    if os.path.exists(oof_pred_path):
        print(f"\n🔮 [OOF 預測] (存在: {oof_pred_path})")
    
    print("\n✅ 完成")

def main():
    parser = argparse.ArgumentParser(description='查看 ML Pipeline 實驗結果')
    parser.add_argument('--run-id', type=str, help='指定 Run ID')
    parser.add_argument('--latest', action='store_true', default=True, help='查看最新的 Run (預設)')
    parser.add_argument('--list', action='store_true', help='列出所有 Runs')
    
    args = parser.parse_args()
    
    if args.list:
        runs = sorted(glob.glob(os.path.join(RUNS_DIR, '*')), key=os.path.getmtime, reverse=True)
        print(f"📂 發現 {len(runs)} 個實驗 runs:")
        for r in runs:
            timestamp = datetime.fromtimestamp(os.path.getmtime(r)).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  - {os.path.basename(r)} ({timestamp})")
        return

    if args.run_id:
        run_path = os.path.join(RUNS_DIR, args.run_id)
        if not os.path.exists(run_path):
            print(f"❌ 找不到 Run ID: {args.run_id}")
            return
        view_run(run_path)
    else:
        # Default to latest
        latest = get_latest_run()
        if latest:
            view_run(latest)
        else:
            print(f"❌ 在 {RUNS_DIR}/ 目錄下找不到任何實驗記錄。")

if __name__ == "__main__":
    main()
