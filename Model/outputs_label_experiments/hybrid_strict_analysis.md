# Hybrid Strict Dropped Samples Analysis

## Overview
- **Baseline Positives:** 464
- **Dropped by Strict (Ratio<0.3):** 27
- **Drop Rate:** 5.82%

## Statistics Comparison

### All Baseline Positives (y=1)
| Metric | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_raw_delta | 464.0000 | 9175.6466 | 37324.0231 | 50.0000 | 50.0000 | 500.0000 | 2000.0000 | 500000.0000 |
| max_raw_ratio | 419.0000 | 1.5243 | 1.4665 | 0.0000 | 0.6667 | 1.0000 | 2.0000 | 9.0000 |

### Dropped Samples (y_base=1, y_strict=0)
| Metric | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| max_raw_delta | 27.0000 | 9259.2593 | 26154.2651 | 2000.0000 | 2000.0000 | 2000.0000 | 2000.0000 | 100000.0000 |
| max_raw_ratio | 27.0000 | 0.2500 | 0.0000 | 0.2500 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |

## Conclusion
The dropped samples have an average max growth ratio of **0.2500** and average delta of **9259.26**.
These represent products that met the absolute delta threshold (10) but failed to meet the 30% relative growth requirement.
This indicates they likely had a high base sales volume, making a delta of 10+ insignificant in relative terms.
