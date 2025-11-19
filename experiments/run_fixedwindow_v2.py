#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
實驗配置：fixed_window v2 - 避免 eval set 沒有正類的問題

設計目標：
- 使用 fixed_window 模式，確保時間尺度一致
- 使用 time-based split，真正模擬預測未來的情境
- 避免過度過濾，確保 eval set 有足夠的正負類樣本
- 使用較寬鬆的過濾條件，確保樣本量足夠

預期結果：
- 正類比例：約 0.5% ~ 5%（取決於 label 定義）
- eval set 中至少有 10 個正類和 10 個負類樣本
- valid_for_evaluation = true

與舊 run (global_top100_xgboost_no_fs) 的差異：
- 舊 run：可能使用了過度過濾或時間區間太窄，導致 eval set 沒有正類
- 新 run：使用較寬鬆的過濾條件，並確保時間區間足夠大
"""

import subprocess
import sys
import os

# 設定實驗參數
config = {
    "mode": "product_level",
    "label_mode": "fixed_window",
    "label_delta_threshold": 12.0,
    "label_ratio_threshold": 0.2,
    "label_window_days": 7.0,
    "align_max_gap_days": 1.0,
    "min_comments": 5,  # 較寬鬆：從 10 降到 5，確保樣本量足夠
    "keyword_blacklist": "口罩",  # 排除口罩，但不過度限制
    "train_end_date": "2025-06-15",  # 較早的切分點，確保訓練集足夠大
    "val_end_date": "2025-06-25",  # 較晚的切分點，確保驗證集和測試集有足夠樣本
    "date_cutoff": "2025-06-25",  # 用於 metadata，不影響資料過濾（fixed_window 模式）
    "algorithms": "xgboost",
    "fs_methods": "no_fs",
    "top_n": "100",
    "cv": 1,  # time-based split 時不使用交叉驗證
    "outdir": "Model/outputs_fixedwindow_v2",
    "oversample": "none",
    "min_eval_pos_samples": 10,  # QA 檢查：至少 10 個正類
    "min_eval_neg_samples": 10,  # QA 檢查：至少 10 個負類
}

def build_command(config):
    """構建訓練命令"""
    cmd = [
        "python", "Model/train.py",
        "--mode", config["mode"],
        "--label-mode", config["label_mode"],
        "--label-delta-threshold", str(config["label_delta_threshold"]),
        "--label-ratio-threshold", str(config["label_ratio_threshold"]),
        "--label-window-days", str(config["label_window_days"]),
        "--align-max-gap-days", str(config["align_max_gap_days"]),
        "--min-comments", str(config["min_comments"]),
        "--keyword-blacklist", config["keyword_blacklist"],
        "--train-end-date", config["train_end_date"],
        "--val-end-date", config["val_end_date"],
        "--date-cutoff", config["date_cutoff"],
        "--algorithms", config["algorithms"],
        "--fs-methods", config["fs_methods"],
        "--top-n", config["top_n"],
        "--cv", str(config["cv"]),
        "--outdir", config["outdir"],
        "--oversample", config["oversample"],
        "--min-eval-pos-samples", str(config["min_eval_pos_samples"]),
        "--min-eval-neg-samples", str(config["min_eval_neg_samples"]),
    ]
    return cmd

def main():
    print("=" * 80)
    print("實驗配置：fixed_window v2")
    print("=" * 80)
    print("\n配置參數：")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n預期結果：")
    print("  - 正類比例：約 0.5% ~ 5%")
    print("  - eval set 中至少有 10 個正類和 10 個負類樣本")
    print("  - valid_for_evaluation = true")
    print("\n與舊 run 的差異：")
    print("  - 舊 run (global_top100_xgboost_no_fs)：可能使用了過度過濾或時間區間太窄")
    print("  - 新 run：使用較寬鬆的過濾條件（min_comments=5），並確保時間區間足夠大")
    print("=" * 80)
    
    cmd = build_command(config)
    print(f"\n執行命令：\n{' '.join(cmd)}\n")
    
    # 確認執行
    response = input("是否執行訓練？(y/n): ")
    if response.lower() != 'y':
        print("取消執行")
        return
    
    # 執行訓練
    print("\n開始訓練...")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if result.returncode == 0:
        print("\n訓練完成！")
        print(f"\n請檢查輸出目錄：{config['outdir']}")
        print("確認 summary CSV 中：")
        print("  - n_test_1 >= 10")
        print("  - n_test_0 >= 10")
        print("  - valid_for_evaluation = true")
    else:
        print(f"\n訓練失敗，退出碼：{result.returncode}")
        sys.exit(1)

if __name__ == "__main__":
    main()

