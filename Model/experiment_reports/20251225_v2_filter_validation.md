# 實驗計畫與結果報告：驗證 v2_error_prod 過濾成效
**日期**: 2025-12-25
**目標**: 嚴謹驗證 `v2_error_prod` 資料清洗策略（基於集成模型共識的困難樣本挖掘）是否能顯著提升模型成效。

## 1. 實驗設計 (控制變因)

我們進行了一項受控的 **A/B 測試**，其中**唯一的變數**是「是否使用 `v2_error_prod` 過濾器」。
為了確保公平性，我們修復了 `train.py` 中的類別平衡機制 (Auto-Balancing)，確保 Baseline 與實驗組都在有權重保護其間的情況下競爭。

| 配置 | 對照組 (Baseline / Control) | 實驗組 (Experiment / Test) |
| :--- | :--- | :--- |
| **過濾策略** | **無** (僅排除手動 ID `8918452`) | **v2_error_prod** (手動 + 共識異音過濾) |
| **類別平衡** | **Active** (Pos Weight ~55) | **Active** (Pos Weight ~55) |
| **標籤定義** | **Hybrid** (Delta>=10, Ratio>=1.0) | **Hybrid** (Delta>=10, Ratio>=1.0) |
| **銷售距 (Gap)** | **14.0 days (Strict)** | **14.0 days (Strict)** |
| **數據規模** | ~7,200 樣本 (Pos Rate 1.7%) | ~6,000 樣本 (Pos Rate 1.7%) |
| **模型算法** | XGBoost (GPU Hist) | XGBoost (GPU Hist) |

> [!NOTE]
> **關於基準線的落差 (Discrepancy)**: 
> 歷史實驗 (328 Positives) 使用的是 `label_max_gap_days=None (Infinite)`，而本次 Comparison A 使用了預設值 `14.0` (Strict)。這導致了約 200 個正樣本被移除，使得 Baseline F1 (15%) 低於歷史回憶 (28%)。
> 我們將執行 **Phase 1.5** (見下文) 來驗證寬鬆標準下的表現。

## 2. 實驗結果 (2025-12-25 最終驗證)

**狀態**: **成功驗證**。

### 3.1 最佳化閥值比較 (Best Threshold Optimization)
*(使用最佳 F1 切點進行比較，排除閥值敏感度影響)*

| 指標 (Metric) | Baseline (已修復平衡) | v2 Filter (過濾後) | 差異 (Diff) | 提升倍數 |
| :--- | :--- | :--- | :--- | :--- |
| **Best F1-Score** | 0.1550 (@ Th=0.03) | **0.3247 (@ Th=0.07)** | +0.1697 | **2.1 倍** |
| **Max AUC** | 0.8485 | **0.9707** | +0.1222 | +14% |
| **Precision (@ Best F1)** | ~0.18 | **~0.48** | 大幅提升 | **2.6 倍** |

### 3.2 關鍵洞察 (Key Insights)
1.  **類別不平衡不是主因**：修復了 `scale_pos_weight` 後，Baseline 的最佳 F1 僅從 15.46% 微升至 15.50%。這證明了**單純加重權重無法解決資料本身的模糊性** (Ambiguity)。
2.  **資料品質是瓶頸**：Baseline 在這份資料 (1.7% 爆品率) 上表現極差，說明充滿了「長得像爆品但沒爆」的噪音。
3.  **過濾策略有效清洗**：`v2 Filter` 成功移除了這些噪音，在損失極少正樣本的情況下 (Pos Rate 維持 1.7%)，將 F1 翻倍至 **32%**，AUC 達到 **0.97**。

## 3. Phase 1.5: 歷史標準重現 (Loose Gap Replication)
**目標**: 將 `label_max_gap_days` 設為 3650 (無限大)，驗證是否能重現 28% 的 Baseline F1。

| Metric | Loose Baseline (Gap=Inf) | Loose v2 Filter (Gap=Inf) |
| :--- | :--- | :--- |
| **Best F1** | *Running...* | *Pending...* |
| **Positives** | *Pending...* | *Pending...* |

## 4. 下一步 (Next Steps)
*   **進行 Comparison B**：既然清洗策略已證實有效（且優於任何權重調整手段），我們將在此乾淨基底上，測試「價格交互特徵 (Price Interaction Features)」的邊際效應。
