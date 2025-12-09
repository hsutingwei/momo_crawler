# 里程碑報告：ML Data Filters 清洗策略構建 (Data Cleaning Strategy)

**日期**: 2025-12-09
**版本**: v1.0
**目標**: 建立「軟刪除 (Soft Delete)」機制，透過多種策略篩選訓練數據中的噪聲，並存入 `ml_data_filters` 資料表以供版本控制。

---

## 1. 篩選策略總覽 (Strategy Overview)

我們實作了三種不同粒度的篩選策略，並將結果存入 PostgreSql 資料庫。

| 版本標籤 (Version Tag) | 篩選層級 | 篩選邏輯 (Logic) | 閥值設定 |
| :--- | :--- | :--- | :--- |
| **v1_corr_kw** | 關鍵字 (Category) | **相關性過低**<br>計算每個關鍵字的 `comment_count` 與 `成長標籤(Target)` 的相關係數。 | Correlation < 0.05 |
| **v2_error_prod** | 商品 (Product) | **集成模型誤差 (Ensemble Error)**<br>訓練 4 個不同特徵集的 XGBoost 模型，統計每個商品被誤判的次數。 | Misclassified by $\ge$ 3 models (out of 4) |
| **v3_error_kw** | 關鍵字 (Category) | **錯誤率過高**<br>計算每個關鍵字下的平均預測錯誤率。 | Error Rate > 0.5 |

*註：所有相關模型訓練皆已加入類別不平衡處理 (`scale_pos_weight` $\approx$ 14.5)。*

---

## 2. 篩選統計結果 (Statistics)

基於全量數據 (N=7197) 的執行結果：

| 策略版本 | 篩除對象數 (Count) | 篩除比例 | 剩餘可用數據 (Estimated) | 狀態 |
| :--- | :--- | :--- | :--- | :--- |
| **v1_corr_kw** | 0 Keywords | 0.0% | 7,197 (100%) | 無顯著噪聲類別 |
| **v2_error_prod** | **1,210 Products** | **16.8%** | **5,987** (83.2%) | **主要生效策略** |
| **v3_error_kw** | 0 Keywords | 0.0% | 7,197 (100%) | 無顯著難預測類別 |

**洞察 (Insights)**：
1.  **類別層級 (Keyword Level)** 的篩選條件 (`v1`, `v3`) 過於寬鬆，或是我們的關鍵字挑選已經相對精準，因此沒有整批被移除的關鍵字。
2.  **商品層級 (Product Level)** 的篩選 (`v2`) 成功識別出約 17% 的「頑固份子」。這些商品在 Price, Kinematics, Novelty 等不同特徵視角下都無法被正確預測，極有可能是標註錯誤或市場行為異常的噪聲。

---

## 3. 使用方式 (SQL Usage)

在訓練模型時，透過 `NOT IN` 子句排除特定版本的噪聲數據。

### 3.1 查詢特定版本的篩選名單

```sql
-- 查看 v2_error_prod 篩選掉的商品 ID 與原因
SELECT filter_value AS product_id, reason, score
FROM ml_data_filters
WHERE version_tag = 'v2_error_prod';
```

### 3.2 在訓練時排除數據 (與 features 表關聯)

假設您的主表為 `product_features` 或在 Python 中載入數據：

```sql
SELECT *
FROM product_features p
WHERE p.product_id NOT IN (
    -- 排除 v2 版本的黑名單商品
    SELECT CAST(filter_value AS INTEGER)
    FROM ml_data_filters
    WHERE version_tag = 'v2_error_prod'
      AND filter_level = 'product_id'
);
```

### 3.3 實驗比較 (A/B Test)

您可以輕鬆進行實驗比較：
*   **Experiment A (Baseline)**: 不加任何過濾條件。
*   **Experiment B (Cleaned)**: 加入 `WHERE product_id NOT IN (...)` 過濾。

---

## 4. 附錄：Ensemble 特徵集定義

`v2_error_prod` 使用了以下四組特徵進行投票：

1.  **Base**: 基礎統計特徵 (評論數、按讚數、情緒分數) + TF-IDF。
2.  **Price (Theme A)**: Base + `price_weighted_arousal`, `bert_arousal_mean` 等。
3.  **Kin (Theme B)**: Base + `kin_acc_abs`, `early_bird_momentum` 等。
4.  **Nov (Theme C)**: Base + `category_fit_score`, `quality_driven_momentum` 等。

只有當一個商品在 **3 個以上** 的模型中都被誤判時，才會被列入 `v2_error_prod` 黑名單。
