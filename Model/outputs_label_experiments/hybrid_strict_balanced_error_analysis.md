# hybrid_strict_balanced Error Analysis

## 檔案說明
- 本報告針對實驗 `hybrid_strict_balanced`（delta>=10, ratio>=0.3 的成長任務），分析 TP/FP/FN/TN 在各種特徵上的分布差異。
- 特別包含與 `binary_explosive`（爆品任務）的交集比較。

**Total Samples:** 7197
**Best Threshold Used:** 0.6000

## 1. Category Distribution

| Category | Count | Percentage |
| --- | --- | --- |
| TP | 102 | 1.42% |
| FP | 480 | 6.67% |
| FN | 335 | 4.65% |
| TN | 6280 | 87.26% |

## 2. Feature Comparison (Mean / Median)

### price

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 1441.0392 | 1305.0000 | 1383.6144 |
| FP | 1487.5896 | 1180.0000 | 1460.9450 |
| FN | 1692.6448 | 1042.0000 | 2039.8159 |
| TN | 1853.5624 | 1180.0000 | 2509.4791 |

### comment_count_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 30.1667 | 8.5000 | 48.3828 |
| FP | 25.2417 | 7.0000 | 45.2837 |
| FN | 25.7224 | 8.0000 | 52.8208 |
| TN | 13.6159 | 1.0000 | 65.2234 |

### score_mean

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 4.8633 | 4.9058 | 0.1789 |
| FP | 4.8261 | 4.9000 | 0.3847 |
| FN | 4.7002 | 4.8810 | 0.7535 |
| TN | 2.8204 | 4.6000 | 2.3721 |

### like_count_sum

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 1.1765 | 0.0000 | 3.9512 |
| FP | 1.1958 | 0.0000 | 4.1014 |
| FN | 1.2119 | 0.0000 | 5.0102 |
| TN | 0.6465 | 0.0000 | 5.1043 |

### had_any_change_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0098 | 0.0000 | 0.0990 |
| FP | 0.0250 | 0.0000 | 0.1563 |
| FN | 0.0179 | 0.0000 | 0.1328 |
| TN | 0.0081 | 0.0000 | 0.0898 |

### num_increases_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0098 | 0.0000 | 0.0990 |
| FP | 0.0250 | 0.0000 | 0.1563 |
| FN | 0.0179 | 0.0000 | 0.1328 |
| TN | 0.0081 | 0.0000 | 0.0898 |

### has_image_urls

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.4804 | 0.0000 | 0.5021 |
| FP | 0.3979 | 0.0000 | 0.4900 |
| FN | 0.4448 | 0.0000 | 0.4977 |
| TN | 0.2416 | 0.0000 | 0.4281 |

### has_video_url

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.2843 | 0.0000 | 0.4533 |
| FP | 0.2437 | 0.0000 | 0.4298 |
| FN | 0.2239 | 0.0000 | 0.4175 |
| TN | 0.1080 | 0.0000 | 0.3104 |

### has_reply_content

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.2843 | 0.0000 | 0.4533 |
| FP | 0.3021 | 0.0000 | 0.4596 |
| FN | 0.2776 | 0.0000 | 0.4485 |
| TN | 0.1271 | 0.0000 | 0.3331 |

### comment_count_7d

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0000 | 0.0000 | 0.0000 |
| FP | 0.0000 | 0.0000 | 0.0000 |
| FN | 0.0000 | 0.0000 | 0.0000 |
| TN | 0.0000 | 0.0000 | 0.0000 |

### comment_count_30d

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0000 | 0.0000 | 0.0000 |
| FP | 0.0000 | 0.0000 | 0.0000 |
| FN | 0.0000 | 0.0000 | 0.0000 |
| TN | 0.0000 | 0.0000 | 0.0000 |

### comment_count_90d

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 30.1667 | 8.5000 | 48.3828 |
| FP | 25.2417 | 7.0000 | 45.2837 |
| FN | 25.7224 | 8.0000 | 52.8208 |
| TN | 13.6159 | 1.0000 | 65.2234 |

### days_since_last_comment

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 35.2831 | 32.5542 | 4.9990 |
| FP | 39.7470 | 39.4686 | 22.0921 |
| FN | 49.5526 | 40.4823 | 46.8379 |
| TN | 175.9705 | 51.1781 | 157.5630 |

### comment_7d_ratio

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0000 | 0.0000 | 0.0000 |
| FP | 0.0000 | 0.0000 | 0.0000 |
| FN | 0.0000 | 0.0000 | 0.0000 |
| TN | 0.0000 | 0.0000 | 0.0000 |

### sentiment_mean_recent

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 4.8633 | 4.9058 | 0.1789 |
| FP | 4.8386 | 4.9000 | 0.2539 |
| FN | 4.7629 | 4.8810 | 0.4017 |
| TN | 4.0491 | 4.6000 | 0.9338 |

### neg_ratio_recent

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0035 | 0.0000 | 0.0155 |
| FP | 0.0032 | 0.0000 | 0.0202 |
| FN | 0.0046 | 0.0000 | 0.0228 |
| TN | 0.0046 | 0.0000 | 0.0360 |

### promo_ratio_recent

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0206 | 0.0000 | 0.0403 |
| FP | 0.0351 | 0.0000 | 0.0786 |
| FN | 0.0415 | 0.0000 | 0.0826 |
| TN | 0.0189 | 0.0000 | 0.0604 |

### repurchase_ratio_recent

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.1496 | 0.1592 | 0.1341 |
| FP | 0.1651 | 0.1720 | 0.1443 |
| FN | 0.1808 | 0.1702 | 0.1643 |
| TN | 0.1060 | 0.0000 | 0.1597 |

## 3. 與 binary_explosive 的交集分析

### 3.1 真實標籤 (Ground Truth) 重疊度
- **Strict (Ratio>=0.3)**: 包含較廣泛的成長商品。
- **Explosive (Ratio>=1.0)**: 嚴格的爆品子集。

#### Label Confusion Matrix (Strict vs Explosive)
|   Strict |    0 |   1 |
|---------:|-----:|----:|
|        0 | 6760 |   0 |
|        1 |  109 | 328 |

### 3.2 模型預測 (Prediction) 重疊度
|   Pred_Strict |    0 |   1 |
|--------------:|-----:|----:|
|             0 | 6437 | 178 |
|             1 |  390 | 192 |

### 3.3 針對「真爆品 (y_explosive=1)」的捕捉能力
- **Total True Explosives**: 328
- **Caught by Strict**: 79 (24.09%)
- **Caught by Explosive**: 62 (18.90%)

#### Strict 抓到但 Explosive 漏掉的爆品 (Strict Wins)
Count: 41
特徵平均值：
|    |   price |   comment_count_pre |   score_mean |   like_count_sum |   had_any_change_pre |   num_increases_pre |   has_image_urls |   has_video_url |   has_reply_content |   comment_count_7d |   comment_count_30d |   comment_count_90d |   days_since_last_comment |   comment_7d_ratio |   sentiment_mean_recent |   neg_ratio_recent |   promo_ratio_recent |   repurchase_ratio_recent |
|---:|--------:|--------------------:|-------------:|-----------------:|---------------------:|--------------------:|-----------------:|----------------:|--------------------:|-------------------:|--------------------:|--------------------:|--------------------------:|-------------------:|------------------------:|-------------------:|---------------------:|--------------------------:|
|  0 | 1311.41 |             23.2439 |      4.85036 |         0.682927 |                    0 |                   0 |         0.560976 |        0.268293 |            0.170732 |                  0 |                   0 |             23.2439 |                   35.0817 |                  0 |                 4.85036 |         0.00271003 |            0.0223435 |                  0.194017 |

#### Explosive 抓到但 Strict 漏掉的爆品 (Explosive Wins)
Count: 24
特徵平均值：
|    |   price |   comment_count_pre |   score_mean |   like_count_sum |   had_any_change_pre |   num_increases_pre |   has_image_urls |   has_video_url |   has_reply_content |   comment_count_7d |   comment_count_30d |   comment_count_90d |   days_since_last_comment |   comment_7d_ratio |   sentiment_mean_recent |   neg_ratio_recent |   promo_ratio_recent |   repurchase_ratio_recent |
|---:|--------:|--------------------:|-------------:|-----------------:|---------------------:|--------------------:|-----------------:|----------------:|--------------------:|-------------------:|--------------------:|--------------------:|--------------------------:|-------------------:|------------------------:|-------------------:|---------------------:|--------------------------:|
|  0 | 1764.17 |             5.08333 |      4.82098 |            0.125 |                    0 |                   0 |         0.166667 |       0.0833333 |                0.25 |                  0 |                   0 |             5.08333 |                   38.7705 |                  0 |                 4.82098 |                  0 |             0.013465 |                 0.0914773 |



## Temporal Trend Features (90-day window)

在此次實驗中，我們加入了 90 天內的三段式評論數統計與加速比率特徵，以捕捉爆發前的趨勢：

- **comment_3rd_30d**: 最近 30 天 (0-30 days before cutoff)
- **comment_2nd_30d**: 中間 30 天 (31-60 days before cutoff)
- **comment_1st_30d**: 最早 30 天 (61-90 days before cutoff)
- **ratio_recent30_to_prev60**: 加速比率 = `comment_3rd_30d / (comment_1st_30d + comment_2nd_30d + 1e-6)`

### 特徵平均值比較 (Mean Values by Group)

| group   |   comment_1st_30d |   comment_2nd_30d |   comment_3rd_30d |   ratio_recent30_to_prev60 |
|:--------|------------------:|------------------:|------------------:|---------------------------:|
| TP      |                 0 |           30.1667 |                 0 |                          0 |
| FN      |                 0 |           25.7224 |                 0 |                          0 |
| FP      |                 0 |           25.2417 |                 0 |                          0 |
| TN      |                 0 |           13.6159 |                 0 |                          0 |
- **comment_3rd_30d**: 最近 30 天 (0-30 days before cutoff)
- **comment_2nd_30d**: 中間 30 天 (31-60 days before cutoff)
- **comment_1st_30d**: 最早 30 天 (61-90 days before cutoff)
- **ratio_recent30_to_prev60**: 加速比率 = `comment_3rd_30d / (comment_1st_30d + comment_2nd_30d + 1e-6)`

### 特徵平均值比較 (Mean Values by Group)

| group   |   comment_1st_30d |   comment_2nd_30d |   comment_3rd_30d |   ratio_recent30_to_prev60 |
|:--------|------------------:|------------------:|------------------:|---------------------------:|
| TP      |          1.61497  |           2.06952 |          2.83957  |                   272728   |
| FN      |          3.216    |           3.272   |          4.216    |                   236001   |
| FP      |          1.81942  |           1.91652 |          2.73083  |                   379899   |
| TN      |          0.889519 |           1.03321 |          0.978131 |                    61558.5 |

### 初步觀察結論

- **正向訊號明顯**：TP 的加速比率 (272728.10) 顯著高於 TN (61558.53)，顯示爆品在 cutoff 前確實有加速跡象。
- **漏抓原因**：FN 的加速比率 (236000.55) 低於 TP，可能因為這些爆品是「突然爆發」或「穩定成長型」，在 cutoff 前尚未展現明顯加速。
