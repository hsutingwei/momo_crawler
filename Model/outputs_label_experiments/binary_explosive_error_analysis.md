# Binary Explosive Error Analysis

**Total Samples:** 7197
**Best Threshold Used:** 0.1500

## 1. Category Distribution

| Category | Count | Percentage |
| --- | --- | --- |
| TP | 287 | 3.99% |
| FP | 3635 | 50.51% |
| FN | 41 | 0.57% |
| TN | 3234 | 44.94% |

## 2. Feature Comparison (Mean / Median)

### price

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 1657.5087 | 1127.0000 | 2020.1216 |
| FP | 1643.2239 | 1090.0000 | 2121.3739 |
| FN | 1349.6341 | 1199.0000 | 1327.6200 |
| TN | 2029.7706 | 1280.0000 | 2745.1573 |

### comment_count_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 11.9268 | 5.0000 | 20.8377 |
| FP | 11.6624 | 4.0000 | 33.7492 |
| FN | 42.8049 | 12.0000 | 91.9109 |
| TN | 19.0931 | 0.0000 | 86.6177 |

### score_mean

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 4.8203 | 4.9125 | 0.2792 |
| FP | 4.7118 | 4.9286 | 0.7564 |
| FN | 3.9000 | 4.8182 | 1.8592 |
| TN | 1.0602 | 0.0000 | 1.9803 |

### like_count_sum

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.3171 | 0.0000 | 0.9048 |
| FP | 0.3620 | 0.0000 | 1.3779 |
| FN | 1.6098 | 0.0000 | 4.2946 |
| TN | 1.1401 | 0.0000 | 7.3123 |

### had_any_change_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0105 | 0.0000 | 0.1019 |
| FP | 0.0116 | 0.0000 | 0.1069 |
| FN | 0.0244 | 0.0000 | 0.1562 |
| TN | 0.0074 | 0.0000 | 0.0858 |

### num_increases_pre

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.0105 | 0.0000 | 0.1019 |
| FP | 0.0116 | 0.0000 | 0.1069 |
| FN | 0.0244 | 0.0000 | 0.1562 |
| TN | 0.0074 | 0.0000 | 0.0858 |

### has_image_urls

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.3449 | 0.0000 | 0.4762 |
| FP | 0.3406 | 0.0000 | 0.4740 |
| FN | 0.5122 | 1.0000 | 0.5061 |
| TN | 0.1694 | 0.0000 | 0.3752 |

### has_video_url

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.1463 | 0.0000 | 0.3541 |
| FP | 0.1395 | 0.0000 | 0.3465 |
| FN | 0.3171 | 0.0000 | 0.4711 |
| TN | 0.1042 | 0.0000 | 0.3056 |

### has_reply_content

| Category | Mean | Median | Std |
| --- | --- | --- | --- |
| TP | 0.2474 | 0.0000 | 0.4322 |
| FP | 0.1882 | 0.0000 | 0.3909 |
| FN | 0.2683 | 0.0000 | 0.4486 |
| TN | 0.0925 | 0.0000 | 0.2897 |

## 3. Key Observations

### TP vs FN (Missed Opportunities)
- **Price**: TP median (1127.0) vs FN median (1199.0).
- **Comments**: TP median (5.0) vs FN median (12.0).

### FP vs TP (False Alarms)
- **Price**: FP median (1090.0) vs TP median (1127.0).
- **Comments**: FP median (4.0) vs TP median (5.0).
