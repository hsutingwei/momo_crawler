# Run Reports 目錄

此目錄存放每次實驗的 Run Report（單次實驗說明檔）。

## 命名規則

`run_YYYYMMDD_xxx_report.md`

例如：`run_20250625_global_top100_xgboost_no_fs_20251119-225842_report.md`

## Run Report 結構

每個 Run Report 應包含：

1. **Run 基本資訊**
   - Run ID
   - 任務類型 / label 模式
   - 資料時間範圍
   - `valid_for_evaluation` 狀態

2. **資料摘要**
   - 樣本數、正負類分布
   - 重要過濾條件

3. **模型與切分**
   - 模型類型與超參數
   - 資料切分策略
   - 不平衡處理方法

4. **整體指標**
   - ROC-AUC、PR-AUC
   - 最佳 threshold 下的指標
   - Top-K 指標

5. **錯誤分析摘錄**
   - False positives 類型
   - False negatives 特徵
   - Feature importance TOP 10

6. **這個 run 告訴我的事**
   - 3-5 個 bullet points 總結發現

7. **下一步實驗建議**
   - 具體改進方向

## 相關檔案

- 對應的 `model_run_v1.json` 位於 `Model/outputs/` 或相關輸出目錄
- 方法解釋檔：`docs/methodology_explainer_v*.md`

