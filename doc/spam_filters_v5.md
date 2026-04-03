# Spam 訊號商品過濾機制 (v5)

## 概述

本過濾機制包含三個版本的過濾條件，專門用來抓出「**可能具有刷評嫌疑（不符合自然 eWOM 擴散機制）**」的異常商品。
執行生成腳本後，符合條件的商品 ID (`product_id`) 將會寫入資料庫的 `public.ml_data_filters` 資料表中，供機器學習訓練腳本 (`train_v2.py`) 透過 `--apply-filter-tag` 參數進行排除，防止這些極端異常資料干擾模型訓練。

---

## 核心異常訊號 (Features)

這三個版本皆主要基於以下三個異常訊號進行判斷：

1. **`feat_entropy_emb`（語意熵）**：
   衡量該商品底下評論的「主題多樣性」。值越低，代表評論長得越像、同質性越高（越有可能是罐頭刷評）。
2. **`feat_ncd_spam`（標準化壓縮距離）**：
   衡量該商品評論與已知「刷評範本」的相似度。值越低，代表與刷評範本越像。
3. **`kin_acc_abs`（評論聲量絕對加速度）**：
   衡量近期評論數量的爆發力。值越高，代表近期聲量異常飆升（典型刷評的特徵之一）。

---

## 空缺值 (Null) 與極端值處理規則

為了確保每一次執行產生的過濾名單都穩定且可重現，針對有缺值或異常值的商品，統一採取以下處理規則：

* **`feat_entropy_emb` 是 NULL 時**：補值為 `1.0`。視為語意多樣性正常（最不可疑），不會因為缺漏值就將其當作 spam。
* **`feat_ncd_spam` 是 NULL 時**：補值為 `1.0`。視為與範本距離遙遠（最不可疑）。
* **`kin_acc_abs` 是 NULL 或負值時**：NULL 補值為 `0`；若為負數，一律統一裁切 (clip) 成 `0`。確保不會有負值影響後續正規化計算。
* **`normalized_kin_acc_abs` 計算**：會先找出全體的最大加速度 (`max(kin_acc_abs)`)，如果最大值也是 0，則所有商品的正規化加速度皆為 `0`。

---

## 三種過濾版本詳細定義

### 1. `v5_spam_fixed_prod` (固定閾值版)

採用**絕對且嚴格的門檻**，必須同時滿足所有極端異常條件才會被抓入黑名單。寧缺勿濫，抓出來的商品嫌疑非常高。

* **過濾條件（需同時滿足）**：
  1. `feat_entropy_emb < 0.5` （評論語意高度集中、太單一）
  2. `feat_ncd_spam < 0.4` （與已知刷評範本長得非常像）
  3. `comment_count_90d > 5` （近 90 天內至少有 5 則以上評論，避免樣本太少造成數據不穩定的誤判）
* **資料庫紀錄欄位**：
  * `reason`: `spam_fixed_threshold`
  * `score`: 該商品的 `kin_acc_abs` (補 0 後的值)

### 2. `v5_spam_quantile_prod` (分位數/大盤相對版)

根據「**當下資料庫中的全體商品分布**」動態決定門檻。尋找在各項異常訊號上「最極端」的群體。

* **分位數計算母體 (Population)**：
  > 僅在上述三個關鍵欄位均非 NULL，**且**近 90 天評論數 > 5 筆的所有商品中計算分位數。
* **過濾條件（需同時滿足）**：
  1. `feat_entropy_emb` ≤ 母體的 **10% 分位數 (P10)**
  2. `feat_ncd_spam` ≤ 母體的 **10% 分位數 (P10)**
  3. `kin_acc_abs` ≥ 母體的 **80% 分位數 (P80)**
  4. `comment_count_90d > 5` （防護機制）
* **資料庫紀錄欄位**：
  * `reason`: `spam_quantile_threshold`
  * `score`: 該商品的 `kin_acc_abs` (補 0 後的值)

### 3. `v5_spam_score_top5_prod` (Spam Score 綜合評分版)

不要求所有條件都極端，而是將三個異常訊號融合計算成一個「**綜合嫌疑分數 (`spam_score`)**」，抓取分數最高的前 5% 商品。可以有效抓出「偏科型」刷評（例如：聲量爆發平平，但語意完全是範本照抄的商品）。

* **分數計算公式**：
  `spam_score = (1 - feat_entropy_emb) * 0.4 + (1 - feat_ncd_spam) * 0.4 + normalized_kin_acc_abs * 0.2`
  *(分數越高，代表同質性越高、越像範本、或爆發力越強，綜合起來越可疑)*
* **過濾條件**：
  * 該商品的 `spam_score` 落在全體商品的 **Top 5%** 以上（即 ≥ P95）。
* **資料庫紀錄欄位**：
  * `reason`: `spam_score_top5`
  * `score`: 計算出的 `spam_score` 數值

---

## 指令參考

### 產生過濾清單寫入資料庫
由 `build_spam_filters.py` 腳本執行。每一次執行前，腳本會自動將資料庫內相同 `version_tag` 的舊資料刪除，確保每次產生的都是最新淨化的完整清單（具備冪等性）。

```bash
# 一次產生 3 個版本的過濾清單
python Model/build_spam_filters.py --date-cutoff 2025-06-25

# 查詢驗證
# psql -c "SELECT version_tag, COUNT(*) AS cnt FROM public.ml_data_filters WHERE version_tag IN ('v5_spam_fixed_prod', 'v5_spam_quantile_prod', 'v5_spam_score_top5_prod') GROUP BY version_tag;"
```

### 模型訓練套用
模型訓練會抓取資料庫當中的名單，直接在 Feature Splitting 前自動排除這些商品。

```bash
# 範例一：套用固定閾值版
python Model/train_v2.py \
    --date-cutoff 2025-06-25 \
    --feature-set baseline \
    --run-id baseline_v5_fixed \
    --group-id v5_spam_test \
    --n-folds 5 \
    --apply-filter-tag v5_spam_fixed_prod

# 範例二：套用綜合評分版
python Model/train_v2.py \
    --date-cutoff 2025-06-25 \
    --feature-set baseline \
    --run-id baseline_v5_score \
    --group-id v5_spam_test \
    --n-folds 5 \
    --apply-filter-tag v5_spam_score_top5_prod
```
