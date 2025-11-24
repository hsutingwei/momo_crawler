# Hybrid Strict Dropped Samples Analysis

## 檔案說明

- **對應實驗**：`hybrid_strict`（Label Strategy = hybrid, delta=10, ratio>=0.3）。
- **目的**：分析在 strict 條件下，哪些原本在 baseline_v1 中被視為「成長」的樣本，因為比例門檻未達 0.3 而被重新標為 0（Dropped Samples），並檢視這些樣本的特性。
- **內容包含**：
  - 各實驗版本的樣本數與 Pos/Neg 分布比較（Overview）。
  - 對於被 hybrid_strict 剔除的樣本（Dropped Samples），統計其 `max_raw_delta`、`max_raw_ratio` 等特徵分布。
- **用法**：
  - 協助理解為什麼要採用 hybrid_strict 策略。
  - 確認被刪除的樣本是否確實為「高基數、低比例」的非顯著成長商品（符合預期）。

## 1. 標籤邏輯與剔除原因 (Logic & Causality)

### 為什麼是 27 筆？
這 27 筆樣本的產生，完全源自於 `baseline_v1` 與 `hybrid_strict` 定義的交集差：

1.  **Baseline v1 (Absolute)**:
    - 規則：`max_raw_delta >= 10`
    - 結果：共有 **464** 筆符合。
2.  **Hybrid Strict**:
    - 規則：`max_raw_delta >= 10` **AND** `max_raw_ratio >= 0.3`
    - 結果：共有 **437** 筆符合。
3.  **Dropped Samples (剔除樣本)**:
    - 定義：符合 Baseline 但 **不符合** Strict 比例門檻的樣本。
    - 數學關係：`464 - 437 = 27`
    - 特徵：`max_raw_delta >= 10` 但 `max_raw_ratio < 0.3`。

### 程式實作對應
- **計算指標**：在 `data_loader.py` 的 SQL 中計算 `max_raw_delta` 與 `max_raw_ratio`。
- **標籤判定**：
  - `baseline`: `label_params={"delta_threshold": 10}` (忽略 ratio)
  - `strict`: `label_params={"delta_threshold": 10, "ratio_threshold": 0.3}`
- **交叉比對**：在 `inspect_label_strategies.py` 中執行：
  ```python
  # 找出在 Baseline 為 1 但在 Strict 為 0 的樣本
  mask_dropped = (df_master["y_baseline"] == 1) & (df_master["y_strict"] == 0)
  df_dropped = df_master[mask_dropped].copy()
  ```

## 2. 剔除樣本實例分析 (Case Study)

這 27 筆被剔除的樣本具有高度一致的特性：**高基數、固定級距成長**。

### 典型數據
從 `hybrid_strict_dropped_samples.csv` 中可見：
- **Max Raw Delta**: 大多為 **2000.0** (或更高)
- **Max Raw Ratio**: 全部為 **0.25**

### 數據還原 (Reverse Engineering)
根據 `ratio = (sales / prev_sales) - 1 = 0.25` 與 `delta = sales - prev_sales = 2000` 推算：
- **Prev Sales (前期銷量)**: 8000
- **Current Sales (當期銷量)**: 10000
- **驗證**:
  - Delta: 10000 - 8000 = 2000 (>= 10, Baseline Pass)
  - Ratio: 2000 / 8000 = 0.25 (< 0.3, Strict Fail)

### 結論
這 27 筆樣本並非「資料錯誤」或「0 -> something」的極端值，而是**已經在熱賣 (Base=8000)** 的商品，銷量往上跳了一個級距 (8000 -> 10000)。雖然絕對成長量很大 (+2000)，但相對成長幅度 (25%) 未達我們設定的「爆發」標準 (30%)，因此被 Strict 策略濾除。這符合我們「排除高基數小波動」的設計初衷。

## 3. 設計動機 (Motivation)

為什麼要設計 `hybrid_strict` 並接受這 27 筆的損失？

1.  **定義「質變」而非「量變」**：
    - 對於一個月賣 10 個的商品，多賣 10 個 (+100%) 是巨大的爆發。
    - 對於一個月賣 8000 個的商品，多賣 2000 個 (+25%) 可能只是正常的行銷波動或庫存回補。
2.  **專注於「爆品潛力股」**：
    - 我們希望模型找出那些「原本沒那麼紅，突然變紅」的商品 (High Ratio)。
    - 已經很紅的商品 (High Base) 繼續維持高檔，雖然對平台營收重要，但不是「爆品預測模型」要抓的主要目標。
3.  **標籤純度**：
    - 透過 `ratio >= 0.3`，我們確保所有被標記為 `1` 的樣本，都有顯著的相對成長，減少模型學到「只要基數大就是 1」的錯誤捷徑。

---


## Overview
- **Baseline Positives:** 464
- **Dropped by Strict (Ratio<0.3):** 27
- **Drop Rate:** 5.82%

## Statistics Comparison

### All Baseline Positives (y=1)
| Metric | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_raw_delta | 464.0000 | 9175.6466 | 37324.0231 | 50.0000 | 50.0000 | 500.0000 | 2000.0000 | 500000.0000 |
| max_raw_ratio | 419.0000 | 1.5243 | 1.4665 | 0.0000 | 0.6667 | 1.0000 | 2.0000 | 9.0000 |

### Dropped Samples (y_base=1, y_strict=0)
| Metric | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_raw_delta | 27.0000 | 9259.2593 | 26154.2651 | 2000.0000 | 2000.0000 | 2000.0000 | 2000.0000 | 100000.0000 |
| max_raw_ratio | 27.0000 | 0.2500 | 0.0000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

## Conclusion
The dropped samples have an average max growth ratio of **0.2500** and average delta of **9259.26**.
These represent products that met the absolute delta threshold (10) but failed to meet the 30% relative growth requirement.
This indicates they likely had a high base sales volume, making a delta of 10+ insignificant in relative terms.
