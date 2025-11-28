# 類別不平衡與閾值調整實驗分析報告

## 1. 實驗結果對比 (Best Threshold F1)

我們將 Phase 1 的三個實驗加上 `scale_pos_weight` 與 `threshold tuning` 後（標記為 `_balanced`），與目前的 `binary_explosive` 進行對比：

| Experiment | Original F1 (Default Th) | **Balanced Best F1** | **Balanced Precision** | **Balanced Recall** |
| :--- | :--- | :--- | :--- | :--- |
| **baseline_v1** | 0.0168 | **0.2062** | 0.1546 | 0.3191 |
| **hybrid_relaxed** | 0.0168 | **0.2062** | 0.1546 | 0.3191 |
| **hybrid_strict** | 0.0267 | **0.2192** | 0.1758 | 0.3803 |
| **binary_explosive** | - | **0.1913** | 0.1678 | 0.2317 |

*(註：binary_explosive 的定義更嚴格，要求 ratio >= 1.0，而 hybrid_strict 僅需 ratio >= 0.3)*

## 2. 假說驗證與結論

針對研究問題：「單純加上類別不平衡處理與 threshold tuning，無法讓 Phase 1 版本的 F1 提升到和 `binary_explosive` 一樣高？」

**結論：此敘述為「錯 (False)」。**

**理由：**
1.  **F1 大幅提升**：Phase 1 的實驗在加入類別平衡處理後，F1 從原本的 ~0.02 暴增至 **0.20 ~ 0.22** 的水準。
2.  **甚至超越 binary_explosive**：`hybrid_strict_balanced` 的最佳 F1 (0.2192) 甚至略高於 `binary_explosive` (0.1913)。

**深入解讀：**
這表示 `binary_explosive` 之前看似巨大的性能優勢（0.19 vs 0.02），**絕大部分來自於工程手段（Class Weight + Threshold Tuning）**，而非單純源自於「更嚴格的爆品定義」或「新加入的特徵」。

然而，這不代表 `binary_explosive` 的設計無效。考慮到 `binary_explosive` 預測的是難度更高的「翻倍成長 (Ratio >= 1.0)」，而 `hybrid_strict` 預測的是「30% 成長」，兩者在 F1 上能達到相近水準 (0.19 vs 0.22)，顯示 `binary_explosive` 在特徵工程（Temporal/Content）上的努力，可能幫助它在更難的任務上維持了競爭力。

**總結：**
若要公平比較不同 Label 定義的優劣，必須在相同的 Imbalance 處理策略下進行。本次實驗證明，**Class Weight 與 Threshold Tuning 是提升此類稀疏目標預測能力的關鍵基石**。
