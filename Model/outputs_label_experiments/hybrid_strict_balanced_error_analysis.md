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


## Arousal & Novelty Analysis (Emotional Intensity)

為了區分「穩定熱銷 (Sustained Hits)」與「爆發新品 (Viral Hits)」，我們引入了情緒強度特徵：

- **arousal_ratio**: 驚嘆、誇張、強烈情緒關鍵字比例 (e.g. 驚豔, 太神, 扯)。
- **novelty_ratio**: 初次體驗、觀望很久關鍵字比例 (e.g. 第一次買, 終於下手)。
- **intensity_score**: 複合指標 = `(arousal + novelty) / (repurchase + 0.1)`。設計用來獎勵高情緒/新奇，懲罰無聊的回購。

### 特徵平均值比較 (Mean Values by Group)

| group   |   arousal_ratio |   novelty_ratio |   intensity_score |   repurchase_ratio_recent |
|:--------|----------------:|----------------:|------------------:|--------------------------:|
| TP      |      0.00278357 |      0.0082418  |         0.063806  |                 0.155597  |
| FN      |      0.00647686 |      0.0105699  |         0.086069  |                 0.195248  |
| FP      |      0.0019865  |      0.0067827  |         0.0523874 |                 0.182707  |
| TN      |      0.00226985 |      0.00376564 |         0.0374285 |                 0.0676064 |

### 初步觀察結論

- **強度區分有效**：TP 的 Intensity Score (0.0638) 高於 TN (0.0374)，顯示爆品通常帶有較強烈的情緒或新奇感。


## Interaction Features Analysis (Cross Features)

為了進一步區分雜訊與真實爆發，我們引入了交互特徵：

- **validated_velocity**: `Acceleration * log1p(Volume)`。確認高成長是否由足夠的評論量支撐。
- **price_weighted_arousal**: `Arousal * log1p(Price)`。區分「廉價炒作 (FP)」與「高價驚豔 (TP)」。
- **novelty_momentum**: `Acceleration * (1 - Repurchase)`。確認成長動力來自「新客」而非「回購」。
- **is_mature_product**: 標記是否為成熟商品 (高累積評論或高回購)。

### 特徵平均值比較 (Mean Values by Group)

| group   |   arousal_ratio |   novelty_ratio |   intensity_score |   repurchase_ratio_recent |   validated_velocity |   price_weighted_arousal |   novelty_momentum |   is_mature_product |
|:--------|----------------:|----------------:|------------------:|--------------------------:|---------------------:|-------------------------:|-------------------:|--------------------:|
| TP      |      0.00448752 |      0.00957546 |         0.0778452 |                 0.168318  |             532727   |                0.0345654 |           267659   |            0.436508 |
| FN      |      0.00541352 |      0.00954605 |         0.0745267 |                 0.191423  |             152222   |                0.0447227 |           126757   |            0.508108 |
| FP      |      0.00231233 |      0.00922019 |         0.0695034 |                 0.189139  |             365630   |                0.0145661 |           284241   |            0.482723 |
| TN      |      0.00223314 |      0.00318481 |         0.0337211 |                 0.0596877 |              43572.6 |                0.0150882 |            43576.6 |            0.153833 |

### 深入觀察結論

- **速度驗證有效**：TP 的 Validated Velocity (532727.2428) 高於 FP (365629.5833)，顯示真爆品通常伴隨更紮實的評論量成長。
- **價格過濾有效**：TP 的 Price Weighted Arousal (0.0346) 高於 FP (0.0146)，成功區分出高價值的驚豔商品，壓制了廉價品的雜訊。
- **新客動力區分**：TP 的 Novelty Momentum (267659.3872) 顯著高於 FN (126757.1372)，證實了爆品成長主要來自新客，而 FN 多為回購驅動。


## BERT Semantic Feature Analysis

我們已導入 BERT Zero-Shot Classification 取代舊的 Regex 關鍵字。以下分析新特徵的效果：

- **clean_arousal_score**: `Arousal * (1 - Negative) * (1 - Advertisement)`。排除負評與廣告後的純淨驚豔分數。
- **bert_negative_mean**: BERT 判定的負面抱怨機率。
- **bert_advertisement_mean**: BERT 判定的廣告業配機率。
- **bert_novelty_mean**: BERT 判定的新奇感機率。

### 特徵平均值比較 (Mean Values by Group)

| group   |   clean_arousal_score |   bert_arousal_mean |   bert_novelty_mean |   bert_repurchase_mean |   bert_negative_mean |   bert_advertisement_mean |   intensity_score |   validated_velocity |   is_mature_product |
|:--------|----------------------:|--------------------:|--------------------:|-----------------------:|---------------------:|--------------------------:|------------------:|---------------------:|--------------------:|
| TP      |             0.0687881 |           0.0960391 |            0.152968 |               0.508962 |            0.0529452 |                 0.223785  |          0.570804 |             591048   |            0.958159 |
| FN      |             0.0819727 |           0.121595  |            0.124164 |               0.492245 |            0.0548418 |                 0.192752  |          0.439009 |             106808   |            0.868687 |
| FP      |             0.071329  |           0.0997388 |            0.126724 |               0.526903 |            0.047144  |                 0.214516  |          0.386704 |             331329   |            0.974781 |
| TN      |             0.0283106 |           0.0414624 |            0.0434   |               0.17801  |            0.0213483 |                 0.0753675 |          0.190234 |              51289.9 |            0.305917 |

### 深入觀察結論

- **Clean Arousal 區分力不足**：TP (0.0688) 與 FP (0.0713) 差異不大，需檢查 FP 是否為高品質的非爆品。


## BERT Semantic Feature Analysis

我們已導入 BERT Zero-Shot Classification 取代舊的 Regex 關鍵字。以下分析新特徵的效果：

- **clean_arousal_score**: `Arousal * (1 - Negative) * (1 - Advertisement)`。排除負評與廣告後的純淨驚豔分數。
- **bert_negative_mean**: BERT 判定的負面抱怨機率。
- **bert_advertisement_mean**: BERT 判定的廣告業配機率。
- **bert_novelty_mean**: BERT 判定的新奇感機率。

## Review Kinematics Analysis (New)

引入「評論動力學」特徵，捕捉評論流量的物理變化：

- **Velocity ($v$)**: `kin_v_1` (近7天), `kin_v_2` (前7-14天)。
- **Acceleration ($a$)**: `kin_acc_abs` ($v_1 - v_2$)。正值代表評論量加速增長。
- **Jerk ($j$)**: `kin_jerk_abs` ($v_1 - 2v_2 + v_3$)。加速度的變化率，預期能捕捉爆發瞬間。

### 特徵平均值比較 (Mean Values by Group)

| group   |   clean_arousal_score |   bert_arousal_mean |   bert_novelty_mean |   bert_repurchase_mean |   bert_negative_mean |   bert_advertisement_mean |   intensity_score |   validated_velocity |   is_mature_product |   kin_v_1 |   kin_v_2 |   kin_v_3 |   kin_acc_abs |   kin_acc_rel |   kin_jerk_abs |
|:--------|----------------------:|--------------------:|--------------------:|-----------------------:|---------------------:|--------------------------:|------------------:|---------------------:|--------------------:|----------:|----------:|----------:|--------------:|--------------:|---------------:|
| TP      |             0.0709092 |           0.0990718 |           0.151348  |               0.512549 |            0.0528661 |                 0.222633  |          0.568076 |             477114   |            0.955414 |  1.21656  |  1.04777  |  0.834395 |     0.16879   |     0.151613  |     -0.044586  |
| FN      |             0.0845973 |           0.129436  |           0.110736  |               0.472895 |            0.0562001 |                 0.176769  |          0.365609 |             102394   |            0.821138 |  0.756098 |  0.739837 |  0.536585 |     0.0162602 |     0.0382928 |     -0.186992  |
| FP      |             0.0740895 |           0.104819  |           0.122896  |               0.521943 |            0.0517217 |                 0.209994  |          0.405582 |             286356   |            0.956851 |  0.890992 |  0.75246  |  0.708554 |     0.138531  |     0.139034  |      0.0946253 |
| TN      |             0.0244053 |           0.0358462 |           0.0380641 |               0.152979 |            0.0182967 |                 0.0660022 |          0.170875 |              41154.5 |            0.259974 |  0.194705 |  0.129803 |  0.150947 |     0.0649016 |     0.0203472 |      0.0860452 |

### 深入觀察結論

- **Clean Arousal 區分力不足**：TP (0.0709) 與 FP (0.0741) 差異不大，需檢查 FP 是否為高品質的非爆品。
- **加速度 (Acceleration) 有效**：TP 的加速度 (0.1688) 顯著高於 FN (0.0163)，顯示爆品具有更強的評論增長動能。
- **急動度 (Jerk) 有效**：TP 的 Jerk (-0.0446) 高於 FN (-0.1870)，成功捕捉到爆發瞬間的非線性增長。


## BERT Semantic Feature Analysis

我們已導入 BERT Zero-Shot Classification 取代舊的 Regex 關鍵字。以下分析新特徵的效果：

- **clean_arousal_score**: `Arousal * (1 - Negative) * (1 - Advertisement)`。排除負評與廣告後的純淨驚豔分數。
- **bert_negative_mean**: BERT 判定的負面抱怨機率。
- **bert_advertisement_mean**: BERT 判定的廣告業配機率。
- **bert_novelty_mean**: BERT 判定的新奇感機率。

## Review Kinematics Analysis (New)

引入「評論動力學」特徵，捕捉評論流量的物理變化：

- **Velocity ($v$)**: `kin_v_1` (近7天), `kin_v_2` (前7-14天)。
- **Acceleration ($a$)**: `kin_acc_abs` ($v_1 - v_2$)。正值代表評論量加速增長。
- **Jerk ($j$)**: `kin_jerk_abs` ($v_1 - 2v_2 + v_3$)。加速度的變化率，預期能捕捉爆發瞬間。

### 特徵平均值比較 (Mean Values by Group)

| group   |   clean_arousal_score |   bert_arousal_mean |   bert_novelty_mean |   bert_repurchase_mean |   bert_negative_mean |   bert_advertisement_mean |   intensity_score |   validated_velocity |   is_mature_product |   kin_v_1 |   kin_v_2 |   kin_v_3 |   kin_acc_abs |   kin_acc_rel |   kin_jerk_abs |   early_bird_momentum |
|:--------|----------------------:|--------------------:|--------------------:|-----------------------:|---------------------:|--------------------------:|------------------:|---------------------:|--------------------:|----------:|----------:|----------:|--------------:|--------------:|---------------:|----------------------:|
| TP      |             0.0692417 |           0.0961254 |           0.157196  |               0.508495 |            0.0556599 |                  0.22354  |          0.600195 |             304166   |            0.955285 |  1.21138  |  1.06098  |  0.731707 |     0.150407  |     0.153565  |     -0.178862  |             0.0367483 |
| FN      |             0.0818717 |           0.12242   |           0.117663  |               0.492235 |            0.0514148 |                  0.19193  |          0.396324 |             458553   |            0.86911  |  0.926702 |  0.832461 |  0.774869 |     0.0942408 |     0.0761227 |      0.0366492 |             0.0333372 |
| FP      |             0.0714892 |           0.100882  |           0.124471  |               0.525981 |            0.0460242 |                  0.213204 |          0.388114 |             338099   |            0.972648 |  0.917943 |  0.80744  |  0.763676 |     0.110503  |     0.126106  |      0.0667396 |             0.0320741 |
| TN      |             0.0282709 |           0.0412638 |           0.0437238 |               0.178035 |            0.0215145 |                  0.075525 |          0.189947 |              50135.7 |            0.306021 |  0.238967 |  0.164557 |  0.18115  |     0.0744099 |     0.0306313 |      0.0910024 |             0.0204866 |

### 深入觀察結論

- **Clean Arousal 區分力不足**：TP (0.0692) 與 FP (0.0715) 差異不大，需檢查 FP 是否為高品質的非爆品。
- **加速度 (Acceleration) 有效**：TP 的加速度 (0.1504) 顯著高於 FN (0.0942)，顯示爆品具有更強的評論增長動能。
