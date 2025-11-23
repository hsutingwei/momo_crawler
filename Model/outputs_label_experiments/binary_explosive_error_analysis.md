# Binary Explosive Error Analysis

**Total Samples:** 7197
**Best Threshold Used:** 0.5000

## 1. Category Distribution

| Category | Count | Percentage |
| --- | --- | --- |
| TP | 138 | 1.92% |
| FP | 1222 | 16.98% |
| FN | 190 | 2.64% |
| TN | 5647 | 78.46% |

## 2. Feature Comparison (Mean / Median)

### price

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 1430.3551 | 1113.0000 | 1586.4061 |
| FP | 1391.8969 | 890.5000 | 1682.4221 |
| FN | 1756.0579 | 1180.0000 | 2167.3046 |
| TN | 1918.9834 | 1214.0000 | 2568.1596 |

### comment_count_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 7.1957 | 4.0000 | 8.6251 |
| FP | 6.1980 | 3.0000 | 10.1881 |
| FN | 22.0263 | 8.0000 | 49.7889 |
| TN | 17.1004 | 1.0000 | 70.7276 |

### score_mean

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 4.8414 | 5.0000 | 0.2773 |
| FP | 4.7309 | 5.0000 | 0.7811 |
| FN | 4.6064 | 4.8750 | 0.9649 |
| TN | 2.6164 | 4.4333 | 2.3933 |

### like_count_sum

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.1304 | 0.0000 | 0.4493 |
| FP | 0.1678 | 0.0000 | 0.6221 |
| FN | 0.7316 | 0.0000 | 2.2716 |
| TN | 0.8497 | 0.0000 | 5.6442 |

### had_any_change_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0072 | 0.0000 | 0.0851 |
| FP | 0.0131 | 0.0000 | 0.1137 |
| FN | 0.0158 | 0.0000 | 0.1250 |
| TN | 0.0089 | 0.0000 | 0.0937 |

### num_increases_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0072 | 0.0000 | 0.0851 |
| FP | 0.0131 | 0.0000 | 0.1137 |
| FN | 0.0158 | 0.0000 | 0.1250 |
| TN | 0.0089 | 0.0000 | 0.0937 |

### has_image_urls

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.2174 | 0.0000 | 0.4140 |
| FP | 0.1956 | 0.0000 | 0.3968 |
| FN | 0.4737 | 0.0000 | 0.5006 |
| TN | 0.2740 | 0.0000 | 0.4460 |

### has_video_url

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0725 | 0.0000 | 0.2602 |
| FP | 0.0704 | 0.0000 | 0.2559 |
| FN | 0.2368 | 0.0000 | 0.4263 |
| TN | 0.1342 | 0.0000 | 0.3409 |

### has_reply_content

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.2536 | 0.0000 | 0.4367 |
| FP | 0.1980 | 0.0000 | 0.3987 |
| FN | 0.2474 | 0.0000 | 0.4326 |
| TN | 0.1312 | 0.0000 | 0.3377 |

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

### days_since_last_comment

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 38.7399 | 39.4686 | 6.7662 |
| FP | 50.0950 | 40.3367 | 48.8111 |
| FN | 55.3623 | 40.4823 | 61.2914 |
| TN | 189.0014 | 57.3830 | 159.6040 |

### comment_7d_ratio

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0000 | 0.0000 | 0.0000 |
| FP | 0.0000 | 0.0000 | 0.0000 |
| FN | 0.0000 | 0.0000 | 0.0000 |
| TN | 0.0000 | 0.0000 | 0.0000 |

## 3. Key Observations

### TP vs FN (Missed Opportunities)
- **Price**: TP median (1113.0) vs FN median (1180.0).
- **Comments**: TP median (4.0) vs FN median (8.0).

### FP vs TP (False Alarms)
- **Price**: FP median (890.5) vs TP median (1113.0).
- **Comments**: FP median (3.0) vs TP median (4.0).
