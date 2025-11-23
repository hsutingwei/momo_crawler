# Label 實驗與特徵設計盤點（截至目前版本）

## 一、目前實際存在的模型／實驗版本

根據 `config/label_experiments.yaml` 與 `Model/outputs_label_experiments` 的紀錄，目前系統中實際運行過且有結果的版本共有 **3 個**。  
它們皆基於 **Product-Level 資料集**，且共用相同的 `date_cutoff` 設定。

| Version ID      | Label Mode  | Label Strategy | 關鍵參數                 | 程式入口                                       |
|----------------|------------|----------------|--------------------------|------------------------------------------------|
| `baseline_v1`   | `next_batch` | `absolute`     | `delta=10`               | `Model/run_experiments.py`（讀取 `config/label_experiments.yaml`） |
| `hybrid_relaxed`| `next_batch` | `hybrid`       | `delta=10, ratio=0.1`    | 同上                                           |
| `hybrid_strict` | `next_batch` | `hybrid`       | `delta=10, ratio=0.3`    | 同上                                           |

> 備註：`Model/train.py` 的預設參數也是 `label_mode="next_batch"`，  
> 因此若直接執行 `python Model/train.py`，行為會與 `baseline_v1` 接近（但其它參數可能略有不同）。

---

## 二、特徵構成分析

上述三個版本共用相同的特徵產生邏輯，由 `load_product_level_training_set` 函式控制。

### 2.1 TF-IDF / BoW 特徵

- **設計型態**：Binary Bag-of-Words（0/1）。
  - 雖然 SQL 查詢的是 `tfidf_scores` 表，但在 `build_sparse_binary_matrix` 函式中，只要 **token 存在即設為 1**  
    （例如：`data = np.ones(...)`）。
- **Vocab 來源**：
  - 來源表：`tfidf_scores` JOIN `product_comments`。
  - Top N 決定方式：對每個 token 計算 `SUM(ts.tfidf)`，代表其在所有評論中的總 TF-IDF 權重，  
    再依總和排序取前 N 個（預設 `top_n=100`）。
- **Cutoff 影響（vocab 層級）**：
  - **沒有**使用 `date_cutoff`。  
  - `fetch_top_terms` 函式目前 **不接受** `date_cutoff` 參數，SQL 中也沒有任何時間過濾條件。
  - 因此 vocab 計算是基於 **全量歷史評論資料**。
- **Product 特徵值（X_tfidf）**：
  - 邏輯：檢查該 product 在 **`date_cutoff` 之前** 的評論中，是否有出現過某個 vocab token。
  - 來源 SQL：`load_product_level_training_set` 中的 `sql_pairs`。
  - 實作上有加入條件：`WHERE pc.capture_time <= %(cutoff)s`，確保只使用 cutoff 前的評論。

### 2.2 數值／統計特徵（Dense Features）

- **主要來源表**：
  - `product_comments`（聚合資料）
  - `sales_snapshots`（歷史銷售變化）
  - `products`（靜態商品資訊）
- **欄位範例**：
  - 靜態特徵：
    - `price`
  - 評論聚合特徵：
    - `score_mean`（平均星數）
    - `like_count_sum`
    - `comment_count_pre`（評論數）
    - `has_image_urls`, `image_urls_count`
    - `has_video_url`
    - `has_reply_content`
  - 歷史銷售特徵（cutoff 前）：
    - `had_any_change_pre`（cutoff 前是否有過銷售變化）
    - `num_increases_pre`（cutoff 前銷售增加次數）
- **Cutoff 影響（dense 層級）**：
  - 有使用 cutoff。  
  - 相關 SQL 中明確使用 `WHERE pc.capture_time <= %(cutoff)s`  
    來過濾用於計算特徵的評論與（部分）銷售紀錄。
  - 語意：dense 特徵只反映 **cutoff 之前** 的歷史狀態。

---

## 三、Cutoff 日期（2025-06-25）的角色

### 3.1 使用位置與語意

在目前使用的 `label_mode="next_batch"` 模式下，`date_cutoff="2025-06-25"` 扮演 **「過去 vs 未來」的分界線**，具體如下：

1. **特徵（X）的邊界**：
   - 所有 Dense Features 和 TF-IDF Features（product-level）都只統計  
     `capture_time <= '2025-06-25'` 的資料。
   - 語意：  
     > 模擬在 **2025-06-25 當下**，我們所能看到的歷史資訊。

2. **標籤（y）的篩選**：
   - Label 計算邏輯如：
     ```sql
     WHEN batch_time > %(cutoff)s ...
     ```
   - 語意：
     > 我們只關心 **2025-06-25 之後** 發生的銷售變化，  
     > 即「cutoff 之後是否曾出現過符合條件的成長事件」。

3. **整體訓練語意**：
   - 訓練資料的：
     - X：截至 2025-06-25 的累積評論與歷史銷售特徵。
     - y：2025-06-25 之後的某段時間內（next_batch 規則）是否出現符合條件的銷售成長。
   - 這是一種 **snapshot-based 的 time-based 設計**：  
     > 用「過去 snapshot」預測「未來是否會有指定事件發生」。

### 3.2 設計理由（目前狀態）

- 在程式碼與文件中，**沒有明確說明** 為何選定 `"2025-06-25"` 這個日期。
- 合理推測：
  - 可能是資料爬取或實驗設計時的某個 **資料截點** 或 **最後更新日期附近的切割點**。
- 實際影響：
  - 模型學習的是：
    > 「截至 2025-06-25 的累積評論與歷史行為」  
    > 如何影響「2025-06-25 之後是否出現某種銷售成長事件」。

---

## 四、TF-IDF 與 Cutoff 的互動關係（潛在風險）

這是目前實作中最需要注意的一塊：**vocab 的選取有輕微的未來資訊洩漏**。

### 4.1 Vocab（Top N）的計算：**有洩漏**

- **現況**：
  - `fetch_top_terms` 函式沒有接受 `date_cutoff` 參數。
  - 對應的 SQL 也沒有任何時間過濾條件。
- **結果**：
  - Vocab 是由 **全量歷史資料（包含 2025-06-25 之後）** 的評論統計出來的。
- **潛在風險**：
  - 若 cutoff 之後出現某個「爆紅的新詞」（例如新產品名稱、流行詞、活動名稱），它有機會被選入 Top N vocab。
  - 雖然在特徵值計算時，舊樣本不會被賦予這個 token（因為 cutoff 前沒有出現），但：
    - 「**選詞這個動作本身**」已經使用了未來資訊 → 嚴格來說是一種 **vocab-level leakage**。

### 4.2 Product 特徵值（X）的計算：**無洩漏（安全）**

- **現況**：
  - 在 `load_product_level_training_set` 中計算 `sql_pairs` 時，明確加上：
    ```sql
    WHERE pc.capture_time <= %(cutoff)s
    ```
- **結果**：
  - 即使 vocab 內存在「未來才出現的 token」，  
    對於每一筆 product-level 訓練樣本，程式只會檢查 **cutoff 之前** 的評論。
  - 若在 cutoff 前沒有出現該 token，該 product 的該維度就是 0。
- **結論**：
  - **特徵值本身** 是不含未來資訊的，  
    符合「用過去特徵預測未來 y」的要求。

### 4.3 總結與建議

- **總體評估**：
  - 目前實作整體上符合：
    > 「用 cutoff 前的特徵預測 cutoff 後的 y」  
  - 唯一放寬的是：
    - **vocab 選取** 使用了全域資料（包含未來），存在輕微的「未來資訊參考」。
- **嚴重性**：
  - 一般情況下，vocab leakage 的實際影響偏小：
    - 尤其在 vocab 規模不算極小、未來特有詞佔比不高時。
- **改善建議（可作為 Phase 2 優化）**：
  1. 修改 `fetch_top_terms`：
     - 讓它也接受 `date_cutoff` 參數。
     - 在 SQL 中加入 `pc.capture_time <= cutoff` 條件，  
       確保 vocab 只來自訓練期間的評論。
  2. 之後可以設計對照實驗：
     - vocab 全域 vs vocab 截到 cutoff。
     - 比較兩者對模型表現與解釋性的差異，並在論文中一併討論。

---
