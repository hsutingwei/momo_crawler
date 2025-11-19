#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试脚本：直接运行分析"""

from analyze_keyword_performance import analyze_keyword_performance

if __name__ == '__main__':
    analyze_keyword_performance(
        predictions_path='Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_predictions.csv',
        model_run_path='Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_model_run_v1.json',
        keyword='益生菌',
        output_dir='analysis'
    )

