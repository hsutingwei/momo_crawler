# Binary Explosive Error Analysis

**Total Samples:** 7197
**Best Threshold Used:** 0.6500

## 1. Category Distribution

| Category | Count | Percentage |
| --- | --- | --- |
| TP | 62 | 0.86% |
| FP | 308 | 4.28% |
| FN | 266 | 3.70% |
| TN | 6561 | 91.16% |

## 2. Feature Comparison (Mean / Median)

### price

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 1603.7097 | 1279.5000 | 1784.7195 |
| FP | 1283.4448 | 828.5000 | 1396.3085 |
| FN | 1622.5940 | 1030.0000 | 1987.4339 |
| TN | 1850.6472 | 1188.0000 | 2477.9476 |

### comment_count_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 6.1774 | 3.0000 | 9.5085 |
| FP | 5.3994 | 3.0000 | 7.0285 |
| FN | 18.0263 | 6.0000 | 42.7242 |
| TN | 15.6191 | 1.0000 | 65.8478 |

### score_mean

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 4.8664 | 5.0000 | 0.2602 |
| FP | 4.7483 | 5.0000 | 0.7770 |
| FN | 4.6677 | 4.8824 | 0.8351 |
| TN | 2.9102 | 4.6667 | 2.3555 |

### like_count_sum

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0806 | 0.0000 | 0.3289 |
| FP | 0.2175 | 0.0000 | 0.7834 |
| FN | 0.5714 | 0.0000 | 1.9553 |
| TN | 0.7523 | 0.0000 | 5.2459 |

### had_any_change_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0000 | 0.0000 | 0.0000 |
| FP | 0.0227 | 0.0000 | 0.1493 |
| FN | 0.0150 | 0.0000 | 0.1219 |
| TN | 0.0090 | 0.0000 | 0.0944 |

### num_increases_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0000 | 0.0000 | 0.0000 |
| FP | 0.0227 | 0.0000 | 0.1493 |
| FN | 0.0150 | 0.0000 | 0.1219 |
| TN | 0.0090 | 0.0000 | 0.0944 |

### has_image_urls

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.2097 | 0.0000 | 0.4104 |
| FP | 0.1818 | 0.0000 | 0.3863 |
| FN | 0.4023 | 0.0000 | 0.4913 |
| TN | 0.2637 | 0.0000 | 0.4407 |

### has_video_url

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.1129 | 0.0000 | 0.3191 |
| FP | 0.0584 | 0.0000 | 0.2350 |
| FN | 0.1805 | 0.0000 | 0.3853 |
| TN | 0.1259 | 0.0000 | 0.3318 |

### has_reply_content

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.2419 | 0.0000 | 0.4318 |
| FP | 0.2597 | 0.0000 | 0.4392 |
| FN | 0.2519 | 0.0000 | 0.4349 |
| TN | 0.1376 | 0.0000 | 0.3445 |

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
| TP | 6.1774 | 3.0000 | 9.5085 |
| FP | 5.3994 | 3.0000 | 7.0285 |
| FN | 18.0263 | 6.0000 | 42.7242 |
| TN | 15.6191 | 1.0000 | 65.8478 |

### days_since_last_comment

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 36.9911 | 39.4686 | 5.4483 |
| FP | 46.0818 | 39.4686 | 49.1765 |
| FN | 51.0207 | 40.4823 | 52.3651 |
| TN | 169.8390 | 51.1781 | 156.5967 |

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
| TP | 4.8664 | 5.0000 | 0.2602 |
| FP | 4.8165 | 5.0000 | 0.3932 |
| FN | 4.7467 | 4.8824 | 0.4349 |
| TN | 4.0839 | 4.6667 | 0.9271 |

### neg_ratio_recent

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0016 | 0.0000 | 0.0127 |
| FP | 0.0023 | 0.0000 | 0.0245 |
| FN | 0.0049 | 0.0000 | 0.0254 |
| TN | 0.0046 | 0.0000 | 0.0353 |

### promo_ratio_recent

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0080 | 0.0000 | 0.0257 |
| FP | 0.0162 | 0.0000 | 0.0670 |
| FN | 0.0362 | 0.0000 | 0.0826 |
| TN | 0.0208 | 0.0000 | 0.0621 |

### repurchase_ratio_recent

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0770 | 0.0000 | 0.1284 |
| FP | 0.1086 | 0.0000 | 0.1529 |
| FN | 0.1736 | 0.1613 | 0.1652 |
| TN | 0.1122 | 0.0000 | 0.1599 |

## 3. Key Observations

### TP vs FN (Missed Opportunities)
- **Price**: TP median (1279.5) vs FN median (1030.0).
- **Comments**: TP median (3.0) vs FN median (6.0).

### FP vs TP (False Alarms)
- **Price**: FP median (828.5) vs TP median (1279.5).
- **Comments**: FP median (3.0) vs TP median (3.0).
