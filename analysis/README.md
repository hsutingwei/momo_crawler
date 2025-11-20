# Keyword 模型表現分析工具

## 功能說明

本目錄包含兩個分析工具：

1. **`analyze_keyword_performance.py`**：分析特定 keyword 的模型表現
2. **`analyze_keyword_performance_all.py`**：自動分析所有 keyword 的模型表現（通用版）

---

## 1. 單一 Keyword 分析

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

## 2. 所有 Keyword 分析（通用版）

`analyze_keyword_performance_all.py` 會自動分析所有 keyword 的模型表現：

### 功能特點

- **自動識別所有 keyword**：從預測結果中自動找出所有出現過的 keyword（排除 blacklist）
- **批量計算指標**：對每個 keyword 計算 n_samples, n_pos, n_neg, pos_rate, precision_1, recall_1, f1_1
- **QA 檢查**：若 keyword 的 n_pos < 5 或 n_samples < 30，標註 `valid_for_keyword_eval = false`
- **輸出格式**：
  - 彙總 CSV：`analysis/keyword_performance_summary.csv`
  - 可選：每個 keyword 的獨立 JSON 檔案

### 使用方法

```bash
# 基本用法（分析所有 keyword）
python analysis/analyze_keyword_performance_all.py

# 完整參數
python analysis/analyze_keyword_performance_all.py \
  --predictions Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_predictions.csv \
  --model-run Model/outputs_fixedwindow_v2/run_20250625_global_top100_xgboost_no_fs_20251119-225842_model_run_v1.json \
  --output-dir analysis \
  --min-pos-samples 5 \
  --min-total-samples 30 \
  --output-individual-json
```

### 參數說明

- `--predictions`：預測結果 CSV 檔案路徑
- `--model-run`：model_run_v1.json 檔案路徑
- `--output-dir`：輸出目錄（預設：analysis）
- `--keyword-blacklist`：要排除的 keyword 列表（逗號分隔），預設為 "口罩"
- `--min-pos-samples`：每個 keyword 至少需要多少正類樣本才視為可評估（預設 5）
- `--min-total-samples`：每個 keyword 至少需要多少總樣本才視為可評估（預設 30）
- `--output-individual-json`：是否為每個 keyword 輸出獨立的 JSON 檔案

### 輸出檔案

1. **`keyword_performance_summary.csv`**：包含所有 keyword 的彙總指標
   - 欄位：keyword, n_samples, n_pos, n_neg, pos_rate, best_threshold, precision_1_best, recall_1_best, f1_1_best, valid_for_keyword_eval

2. **`keyword_{keyword}_performance.json`**（如果啟用 `--output-individual-json`）：
   - 每個 keyword 的詳細指標，格式與單一 keyword 分析相同

### 用途

- **Dashboard 顯示**：用於顯示「各品類難度比較」
- **論文分析**：用於「per-keyword performance」分析
- **模型診斷**：識別哪些 keyword 的預測難度較高，需要特別關注

---

## 範例

```bash
# 分析「益生菌」的模型表現
python analysis/analyze_keyword_performance.py --keyword 益生菌

# 使用自訂的預測結果檔案
python analysis/analyze_keyword_performance.py --keyword 益生菌 --predictions path/to/predictions.csv --model-run path/to/model_run_v1.json

# 分析所有 keyword 的模型表現
python analysis/analyze_keyword_performance_all.py
```

