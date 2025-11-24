# binary_explosive_error_analysis.csv 說明

## 檔案用途
這個 CSV 檔由 `analyze_binary_errors.py` 產生，是 `binary_explosive` 任務錯誤分析的「逐樣本原始資料」。每一列對應一個商品樣本，包含其真實標籤、模型預測機率以及關鍵特徵值。

## 欄位說明

### 1. 基本識別與標籤
- `product_id`: 商品唯一識別碼。
- `y_true`: 真實標籤（0 = 非爆品, 1 = 爆品）。
- `y_prob`: 模型預測為爆品（Class 1）的機率值 (0.0 ~ 1.0)。
- `y_pred_best`: 在最佳閾值（Best Threshold）下的預測結果（0 或 1）。
- `best_threshold_used`: 該次實驗所使用的最佳閾值。

### 2. 靜態與累積特徵
- `price`: 商品價格。
- `comment_count_pre`: 截止日前的累積評論總數。
- `score_mean`: 截止日前的平均評分。
- `like_count_sum`: 截止日前的累積按讚數。
- `had_any_change_pre`: 截止日前是否曾有銷售變動 (0/1)。
- `num_increases_pre`: 截止日前銷售成長次數。
- `has_image_urls`: 評論是否包含圖片 (0/1)。
- `has_video_url`: 評論是否包含影片 (0/1)。
- `has_reply_content`: 是否有賣家回覆 (0/1)。

### 3. 時間維度特徵 (Temporal Features)
- `days_since_last_comment`: 最後一則評論距離截止日的天數（Recency）。
- `comment_count_7d`: 截止日前 7 天內的評論數。
- `comment_count_30d`: 截止日前 30 天內的評論數。
- `comment_count_90d`: 截止日前 90 天內的評論數。
- `comment_7d_ratio`: 近 7 天評論數佔總評論數的比例。

### 4. 內容維度特徵 (Content Features - 90天視窗)
- `sentiment_mean_recent`: 近 90 天評論的平均評分。
- `neg_ratio_recent`: 近 90 天負評（<=2分）佔比。
- `promo_ratio_recent`: 近 90 天包含促銷關鍵字（如特價、打折）的評論佔比。
- `repurchase_ratio_recent`: 近 90 天包含回購關鍵字（如回購、囤貨）的評論佔比。

## 用法建議
- 可使用 Excel 或 Pandas 載入此 CSV。
- 根據 `y_true` 與 `y_pred_best` 篩選出 TP/FP/FN/TN 樣本。
- 針對特定誤判樣本（如 FN），查看其 `product_id` 並回溯原始評論內容，進行 Case Study。
