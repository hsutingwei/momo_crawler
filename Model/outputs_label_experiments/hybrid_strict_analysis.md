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
