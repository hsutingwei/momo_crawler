#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試修正後的模型訓練
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

from Model.train import main

if __name__ == "__main__":
    # 使用較小的參數進行快速測試
    sys.argv = [
        "test_model_fixes.py",
        "--top-n", "50",  # 減少特徵數量
        "--date-cutoff", "2025-06-25",
        "--fs-topk", "30",  # 特徵選取數量
        "--out", "test_results"
    ]
    
    print("開始測試修正後的模型訓練...")
    main()
    print("測試完成！")

