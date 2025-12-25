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

---

## 5. 附錄：v2_error_prod 的具體過濾條件 (Detailed Filtering Logic)

v2_error_prod 是基於產品層級的「模型一致錯誤 (Consensus Error)」進行過濾。具體流程如下：

### 5.1 定義標籤任務
*   固定標籤 $y$：使用 `binary_explosive` 任務定義（在 cutoff 之後，銷量是否有爆發：`ratio >= 1.0` 且 `delta >= 10`）。

### 5.2 訓練 4 組不同視角的模型
我們對每個商品訓練 4 組使用不同特徵子集 (Feature Subsets) 的 XGBoost 模型：
1.  **Base**：基本結構型特徵（價格、評分、歷史評論量、TF-IDF 等）。
2.  **Price**：`Base` + 價格交互特徵 (Price-Moderated Features)。
3.  **Kin**：`Base` + 動能特徵 (Review Kinematics: velocity, acceleration 等)。
4.  **Nov**：`Base` + 新奇度特徵 (Novelty, BERT, Category Fit 等)。

### 5.3 計算一致錯誤 (Ensemble Error)
對每一組模型，使用交叉驗證 (Cross-Validation) 得到每個商品的預測標籤 $y_{pred}^{model}$。
對每一個商品 $P$，計算它被幾個模型判錯：

$$ \text{is\_wrong}_{model} = \begin{cases} 1 & \text{if } y_{pred}^{model} \neq y_{true} \\ 0 & \text{otherwise} \end{cases} $$

$$ \text{fail\_count} = \text{is\_wrong}_{Base} + \text{is\_wrong}_{Price} + \text{is\_wrong}_{Kin} + \text{is\_wrong}_{Nov} $$

### 5.4 篩選條件 (Deletion Criteria)
若某個商品在 4 個模型中有 **至少 3 個** 判錯：

$$ \text{fail\_count} \ge 3 $$

則該商品會被視為「在多種特徵視角下都難以學習／高度不穩定」的樣本，並寫入 `ml_data_filters`：
*   `version_tag` = 'v2_error_prod'
*   `filter_level` = 'product_id'
*   `reason` = 'ensemble_error_count'
*   `score` = fail_count

**一句話總結**：
`v2_error_prod` = 在 4 種不同特徵視角的模型中，被至少 3 個模型判錯的商品，直接標記為「問題樣本」並排除。

---
### 5.5 為什麼不直接用簡單規則 (如長度)？

其實我們**已經有**使用簡單規則在前處理階段了。`v2_error_prod` 負責的是「規則篩不掉的」高級噪聲。

## 6. 附錄：系統中現存的具體篩選規則 (Existing Hard Rules)

除了 `v2_error_prod` (模型篩選) 之外，我們的 Pipeline 在前處理階段已經包含以下**具體規則**：

| 元件 (Component) | 具體條件 (Condition) | 目的 (Purpose) | 程式碼位置 |
| :--- | :--- | :--- | :--- |
| **BERT 情緒特徵** | `length(text) <= 5` | **忽略過短評論**<br>字數少於 5 字 (如"好"、"讚") 不進 BERT 計算，避免產生無意義的情緒分數。 | `compute_bert_features.py` |
| **NCD 壓縮特徵** | `length(text) < 10` | **忽略短文本壓縮**<br>字數少於 10 字無法有效計算壓縮率，直接跳過 (設為預設值)。 | `data_loader.py` |
| **TF-IDF 關鍵字** | `len(token) < 2` OR `len(token) > 4` | **詞彙長度限制**<br>只保留 2~4 個字的詞彙，過濾單字 (停用詞) 或過長句子。 | `data_loader.py` |
| **Category Fit** | `group_size < 2` | **群體過小**<br>若該關鍵字下的商品數少於 2 個，無法計算「群體共識」，Fit Score 設為 0。 | `data_loader.py` |

**結論**：
*   **簡單的噪聲 (如字數太短)**已經被上述規則擋下了。
*   **`v2_error_prod` 刪除的** 是那些「字數正常、格式正常」，但**內容邏輯矛盾** (例如：給了 5 星好評卻寫負評，導致 Price/Nov 模型都預測失敗) 的進階噪聲。
