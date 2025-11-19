# Keyword 模型表現分析工具

## 功能說明

`analyze_keyword_performance.py` 用於分析特定 keyword 的模型表現，包括：

1. 載入預測結果並篩選特定 keyword
2. 從資料庫載入商品資訊（product_name）
3. 計算基本統計（樣本數、正負類比例）
4. 計算 threshold = 0.5 和最佳 threshold 下的指標（precision, recall, F1）
5. 輸出 Top 20 最高分樣本及其 TP/FP 分析
6. 輸出所有 FN 樣本（False Negative）

## 使用方法

```bash
python analysis/analyze_keyword_performance.py --keyword 益生菌
```

### 參數說明

- `--keyword` (必填): 要分析的 keyword，例如「益生菌」、「維他命」等
- `--predictions` (選填): 預測結果 CSV 檔案路徑，預設為 `Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_predictions.csv`
- `--model-run` (選填): model_run_v1.json 檔案路徑，預設為 `Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_model_run_v1.json`
- `--output-dir` (選填): 輸出目錄，預設為 `analysis`

## 輸出

### Console 輸出

腳本會在 console 輸出：
- 基本統計資訊（樣本數、正負類數量、正類比例）
- Threshold = 0.5 和最佳 threshold 下的指標
- Top 20 最高分樣本列表（含 TP/FP 分析）
- 所有 FN 樣本列表

### JSON 檔案

結果會儲存到 `analysis/keyword_{keyword}_performance.json`，包含：

```json
{
  "keyword": "益生菌",
  "n_samples": 100,
  "n_pos": 10,
  "n_neg": 90,
  "pos_rate": 0.1,
  "precision_1_05": 0.2,
  "recall_1_05": 0.15,
  "f1_1_05": 0.17,
  "best_threshold": 0.05,
  "chosen_for": "f1_1",
  "precision_1_best": 0.25,
  "recall_1_best": 0.20,
  "f1_1_best": 0.22,
  "top20_tp": 3,
  "top20_fp": 2,
  "top20_tn": 10,
  "top20_fn": 5,
  "total_fn": 8
}
```

## 範例

```bash
# 分析「益生菌」的模型表現
python analysis/analyze_keyword_performance.py --keyword 益生菌

# 使用自訂的預測結果檔案
python analysis/analyze_keyword_performance.py --keyword 益生菌 --predictions path/to/predictions.csv --model-run path/to/model_run_v1.json
```

