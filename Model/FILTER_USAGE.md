# Filter System 使用指南

## 概述

新的过滤系统允许您基于不同准则（评论数、文本质量等）过滤产品，并将过滤清单存储在 `ml_data_filters` 表中，实现版本控制和可重现的实验。

---

## 1. 生成过滤清单

### 1.1 最低评论数过滤 (min_comments)

过滤掉评论数少于阈值的产品：

```bash
python Model/build_filters.py \
  --mode min_comments \
  --date-cutoff 2025-06-25 \
  --min-comments 5
```

**生成的 version_tag**: `v4_min_comments_prod__cutoff_20250625__min5`

**示例输出**:
```
🔍 Building min_comments filter (cutoff=2025-06-25, min=5)
  version_tag: v4_min_comments_prod__cutoff_20250625__min5
  Found 3582 products with comment_count < 5
  ✅ Inserted 3582 records into ml_data_filters
  💾 Saved to: Model/filters/excluded_products_v4_min_comments_prod__cutoff_20250625__min5.parquet

📊 Summary:
  Excluded products: 3582
  Total products: 7197
  Exclusion ratio: 49.77%
```

---

### 1.2 空文本过滤 (empty_doc)

过滤掉 TF-IDF `doc_text` 为空的产品（即使有评论，但分词/清洗后为空）：

```bash
python Model/build_filters.py \
  --mode empty_doc \
  --date-cutoff 2025-06-25
```

**生成的 version_tag**: `v5_empty_doc_prod__cutoff_20250625`

**示例输出**:
```
🔍 Building empty_doc filter (cutoff=2025-06-25)
  version_tag: v5_empty_doc_prod__cutoff_20250625
  Loaded 7197 products
  Found 2410 products with empty doc_text (33.48%)
  ✅ Inserted 2410 records into ml_data_filters
  💾 Saved to: Model/filters/excluded_products_v5_empty_doc_prod__cutoff_20250625.parquet
```

---

## 2. 在训练中套用过滤器

### 2.1 单个过滤器

```bash
python Model/train_v2.py \
  --date-cutoff 2025-06-25 \
  --feature-set baseline \
  --run-id baseline_filter_min5 \
  --group-id ablation_filters \
  --n-folds 5 \
  --random-seed 42 \
  --tfidf-dim 500 \
  --apply-filter-tag v4_min_comments_prod__cutoff_20250625__min5 \
  --label-strategy absolute \
  --hyperparameter-mode tuning \
  --threshold-mode tuned \
  --paper-mode \
  --output-dir runs
```

### 2.2 多个过滤器（逻辑 OR）

同时套用最低评论数 + 空文本过滤：

```bash
python Model/train_v2.py \
  --date-cutoff 2025-06-25 \
  --feature-set baseline \
  --run-id baseline_filter_combo \
  --group-id ablation_filters \
  --apply-filter-tag "v4_min_comments_prod__cutoff_20250625__min5,v5_empty_doc_prod__cutoff_20250625" \
  ...
```

**训练时输出示例**:
```
🗑️  套用 ml_data_filters 過濾清單
  Filter tags: v4_min_comments_prod__cutoff_20250625__min5,v5_empty_doc_prod__cutoff_20250625

  📋 Loading filters from ml_data_filters...
    v4_min_comments_prod__cutoff_20250625__min5: 3582 products
    v5_empty_doc_prod__cutoff_20250625: 2410 products
  ✅ Total excluded products: 4216

  Before: 7197 products
  After:  2981 products
  ✅ Filtered: 4216 products (58.57%)
```

---

## 3. 版本控制

### 3.1 命名规则

| 过滤器类型 | version_tag 格式 | 示例 |
|-----------|-----------------|------|
| min_comments | `v4_min_comments_prod__cutoff_{YYYYMMDD}__min{N}` | `v4_min_comments_prod__cutoff_20250625__min10` |
| empty_doc | `v5_empty_doc_prod__cutoff_{YYYYMMDD}` | `v5_empty_doc_prod__cutoff_20250625` |
| v2_error_prod (旧) | `v2_error_prod` | `v2_error_prod` |

### 3.2 重跑安全性

所有 INSERT 都使用 `ON CONFLICT DO NOTHING`，重复运行 `build_filters.py` 不会产生重复记录。

---

## 4. 查询过滤清单

### 4.1 查看已有的过滤版本

```sql
SELECT 
    version_tag, 
    COUNT(*) AS excluded_count,
    MIN(created_at) AS created_at
FROM ml_data_filters
WHERE filter_level = 'product_id'
GROUP BY version_tag
ORDER BY created_at DESC;
```

### 4.2 查看某个过滤器的详细信息

```sql
SELECT 
    filter_value AS product_id,
    reason,
    score
FROM ml_data_filters
WHERE version_tag = 'v4_min_comments_prod__cutoff_20250625__min5'
  AND filter_level = 'product_id'
ORDER BY score ASC
LIMIT 20;
```

---

## 5. 实验建议

### 5.1 A/B 测试不同阈值

```bash
# Baseline (无过滤)
python Model/train_v2.py --run-id baseline_no_filter --group-id ablation_min_comments ...

# min_comments >= 5
python Model/train_v2.py --run-id baseline_min5 --group-id ablation_min_comments \
  --apply-filter-tag v4_min_comments_prod__cutoff_20250625__min5 ...

# min_comments >= 10
python Model/train_v2.py --run-id baseline_min10 --group-id ablation_min_comments \
  --apply-filter-tag v4_min_comments_prod__cutoff_20250625__min10 ...
```

对比 AUC 变化，选择最佳阈值。

### 5.2 空文本过滤的适用场景

- **使用 TF-IDF 特征时**：强烈建议套用 `empty_doc` 过滤器
- **只用 dense features**：可以不用（但过滤掉可能提升质量）

---

## 6. 本地 Parquet 文件

每次运行 `build_filters.py` 都会生成本地备份：

```
Model/filters/excluded_products_<version_tag>.parquet
```

可用于离线分析或调试：

```python
import pandas as pd

df = pd.read_parquet('Model/filters/excluded_products_v4_min_comments_prod__cutoff_20250625__min5.parquet')
print(df.head())
print(df['score'].describe())
```

---

## 7. 常见问题

### Q1: 我要测试不同的 cutoff，需要怎么做？

重新运行 `build_filters.py` 并指定新的 `--date-cutoff`：

```bash
python Model/build_filters.py --mode min_comments --date-cutoff 2025-07-01 --min-comments 5
# version_tag 会自动变成: v4_min_comments_prod__cutoff_20250701__min5
```

### Q2: 如何删除某个过滤版本？

```sql
DELETE FROM ml_data_filters
WHERE version_tag = 'v4_min_comments_prod__cutoff_20250625__min5';
```

### Q3: 过滤器会影响 test set 吗？

会。过滤是在 split 前进行的，所以 train_pool 和 test 都会被过滤。

### Q4: 能否只过滤 train，保留 test？

目前不支持。过滤是在 dataset 级别，split 之前。如需此功能，请修改 `train_v2.py` 在 split 之后再过滤。

---

## 8. 技术细节

### 8.1 empty_doc 判断逻辑

与 `data_loader.py` 完全一致：

1. 优先使用 `doc_text_tokenized`（分词结果）
2. 如果 tokenized 为空，fallback 到 `aggregated_comments`（原始文本）
3. 两者都为空或清洗后为空，判定为 empty

### 8.2 min_comments 计算

```sql
COUNT(DISTINCT pc.comment_id)
WHERE pc.capture_time <= cutoff
```

只计算 cutoff 前捕获的评论（与 TF-IDF 的 cutoff 一致）。
