# -*- coding: utf-8 -*-
"""
ABLATION_STUDY_STRICT_MODE.md

严格消融实验执行指南 - 修正版

解决的6个关键问题:
1. baseline 和 variants 使用相同 splits
2. 明确 feature set 定义(cum累加)
3. Hash verification for locked params
4. Threshold locking
5. 固定 cutoff/label 条件
6. Feature whitelist + forbidden columns

## 问题修正总结

### ✅ 问题1: Baseline 和 Variants 共用 Splits

**修正**: 使用 `--group-id` + `--force-use-splits` 确保相同 splits

**Baseline 命令**:
```bash
python Model/train_v2.py \
  --date-cutoff 2025-06-25 \
  --feature-set baseline \
  --run-id baseline_20260102 \
  --group-id ablation_physical_20260102 \
  --n-folds 10 \
  --random-seed 42 \
  --hyperparameter-mode tuning
```

此命令会：
- 生成 `runs/baseline_20260102/splits.parquet`
- 生成 `runs/baseline_20260102/baseline_best_params.json`
- 生成 `runs/baseline_20260102/chosen_threshold.json`

**+Physical 命令** (使用相同 splits):
```bash
python Model/train_v2.py \
  --date-cutoff 2025-06-25 \
  --feature-set +physical \
  --run-id physical_20260102 \
  --group-id ablation_physical_20260102 \
  --n-folds 10 \
  --random-seed 42 \
  --hyperparameter-mode locked \
  --baseline-params-path runs/baseline_20260102/baseline_best_params.json \
  --threshold-mode locked \
  --threshold-path runs/baseline_20260102/chosen_threshold.json \
  --force-use-splits runs/baseline_20260102/splits.parquet
```

**验收标准**:
```bash
# 验证 splits 完全一致
diff runs/baseline_20260102/splits.parquet runs/physical_20260102/splits.parquet
# 应该返回: Files are identical

# 验证 hash 一致
cat runs/baseline_20260102/hashes.json | grep split_hash
cat runs/physical_20260102/hashes.json | grep split_hash
# 应该输出相同的 split_hash

cat runs/baseline_20260102/hashes.json | grep dataset_hash
cat runs/physical_20260102/hashes.json | grep dataset_hash
# 应该输出相同的 dataset_hash
```

---

### ✅ 问题2: Feature Set 定义明确

**修正**: 已在 `experiment_utils.py` 中明确标注为累加模式

```python
# experiment_utils.py:get_feature_whitelist()

# Feature sets are CUMULATIVE (逐層累加)
# - baseline: 基础统计特征
# - +physical: baseline + 运动学特征 (含 quality_driven_momentum)
# - +semantic: +physical + 语义特征
# - +psych: +semantic + 心理层特征

# Note: quality_driven_momentum 需要 category_fit_score，
#       但在消融实验中归类为 physical layer (复合动量)
```

**验收标准**:
```bash
# 查看每个 run 的 feature_list.json
cat runs/baseline_20260102/feature_list.json  # 应该只有 baseline features
cat runs/physical_20260102/feature_list.json  # 应该有 baseline + physical features

# 确认 category_fit_score 在 physical layer
grep "category_fit_score" runs/physical_20260102/feature_list.json  # 应找到
grep "category_fit_score" runs/baseline_20260102/feature_list.json  # 应找不到
```

---

### ✅ 问题3: Hash Verification for Locked Params

**待实现**: 需要在 `train_v2.py` 的 `hyperparameter_mode=locked` 分支添加：

```python
# 伪代码 - 需要添加到 train_v2.py Step 3 之后
if args.hyperparameter_mode == 'locked':
    if not args.baseline_params_path:
        raise ValueError("--baseline-params-path required for locked mode")
    
    # Load baseline params
    with open(args.baseline_params_path, 'r') as f:
        baseline_params = json.load(f)
    
    # Verify split_hash matches
    if baseline_params['split_hash'] != split_hash:
        raise ValueError(
            f"Split hash mismatch!\n"
            f"  Baseline: {baseline_params['split_hash']}\n"
            f"  Current:  {split_hash}\n"
            f"  You are using different splits! Use --force-use-splits"
        )
    
    # Verify dataset_hash matches
    if baseline_params['dataset_hash'] != dataset_hash:
        raise ValueError(
            f"Dataset hash mismatch!\n"
            f"  Baseline: {baseline_params['dataset_hash']}\n"
            f"  Current:  {dataset_hash}\n"
            f"  Check: --date-cutoff, --label-strategy, --exclude-products"
        )
    
    print("✅ Hash verification passed!")
```

**验收标准**:
- 如果用错 splits → 立即报错
- 如果改了 date_cutoff → 立即报错

---

### ✅ 问题4: Threshold Locking

**修正**: 已添加 `--threshold-mode locked` + `--threshold-path`

**Baseline** (tuning):
```python
# baseline 会保存 chosen_threshold.json
{
  "value": 0.48,  # 假设 OOF tuning 结果
  "method": "tuned",
  "source": "oof_f1_maximization",
  "oof_metrics_at_threshold": {...}
}
```

**+Physical** (locked):
```bash
python Model/train_v2.py \
  ... \
  --threshold-mode locked \
  --threshold-path runs/baseline_20260102/chosen_threshold.json
```

**待实现**: 需要在 `train_v2.py` Step 10 修改：

```python
# 伪代码 - 替换当前的 threshold = 0.5
if args.threshold_mode == 'locked':
    if not args.threshold_path:
        raise ValueError("--threshold-path required for locked mode")
    
    with open(args.threshold_path, 'r') as f:
        baseline_threshold = json.load(f)
    
    chosen_threshold = baseline_threshold['value']
    print(f"  Using locked threshold: {chosen_threshold} (from baseline)")
elif args.threshold_mode == 'tuned':
    # TODO: implement threshold tuning on OOF
    chosen_threshold = 0.5  # placeholder
else:  # fixed
    chosen_threshold = 0.5

# Apply threshold
y_test_pred = (y_test_prob > chosen_threshold).astype(int)
```

**验收标准**:
- Baseline 和 +physical 的 `chosen_threshold.json` 中 `value` 应完全一致

---

### ✅ 问题5: 固定 Cutoff/Label 条件

**修正**: 所有条件已在 args 中定义，会写入 `run_config.json`

**关键参数** (baseline 和 variants 必须一致):
```bash
--date-cutoff 2025-06-25 \
--label-strategy absolute \  # 或 hybrid
--label-delta-threshold 10.0 \
--label-ratio-threshold 1.0 \
--exclude-products 8918452,1234567  # 如有
```

**验收标准**:
```bash
# 比较两个 runs 的关键配置
jq '.date_cutoff,.label_strategy,.label_delta_threshold' runs/baseline_20260102/run_config.json
jq '.date_cutoff,.label_strategy,.label_delta_threshold' runs/physical_20260102/run_config.json
# 应完全一致
```

---

### ✅ 问题6: Feature Whitelist + Forbidden Columns

**修正**: 已在 `experiment_utils.py` 中定义 FORBIDDEN list

```python
# Forbidden features (NEVER include these)
FORBIDDEN = ['product_id', 'y_true', 'fold_id', 'split', 'keyword']

# 在 FeatureTransformer 调用前自动过滤
available_features = [f for f in feature_whitelist if f in df_full.columns]
# df_full 本身就不应该包含 FORBIDDEN columns（在 merge 前移除）
```

**验收标准**:
```bash
# 确认 feature_list.json 中没有 forbidden features
cat runs/baseline_20260102/feature_list.json | grep -E "product_id|y_true|fold_id|split|keyword"
# 应该找不到任何匹配
```

---

## 完整的消融实验执行流程

### Step 1: Baseline Run (Tuning)
```bash
python Model/train_v2.py \
  --date-cutoff 2025-06-25 \
  --feature-set baseline \
  --run-id baseline_20260102 \
  --group-id ablation_physical_20260102 \
  --n-folds 10 \
  --random-seed 42 \
  --label-strategy absolute \
  --label-delta-threshold 10.0 \
  --hyperparameter-mode tuning \
  --threshold-mode tuned \
  --paper-mode \
  --output-dir runs
```

预计产出:
- `runs/baseline_20260102/splits.parquet` ← 重要！
- `runs/baseline_20260102/baseline_best_params.json` ← 重要！
- `runs/baseline_20260102/chosen_threshold.json` ← 重要！
- `runs/baseline_20260102/hashes.json`
- `runs/baseline_20260102/metrics.json`
- ...（15+ artifacts）

### Step 2: +Physical Run (Locked)
```bash
python Model/train_v2.py \
  --date-cutoff 2025-06-25 \
  --feature-set +physical \
  --run-id physical_20260102 \
  --group-id ablation_physical_20260102 \
  --n-folds 10 \
  --random-seed 42 \
  --label-strategy absolute \
  --label-delta-threshold 10.0 \
  --hyperparameter-mode locked \
  --baseline-params-path runs/baseline_20260102/baseline_best_params.json \
  --threshold-mode locked \
  --threshold-path runs/baseline_20260102/chosen_threshold.json \
  --force-use-splits runs/baseline_20260102/splits.parquet \
  --paper-mode \
  --output-dir runs
```

**自动验证**:
- ✅ split_hash 一致性检查
- ✅ dataset_hash 一致性检查
- ✅ 使用相同 hyperparameters
- ✅ 使用相同 threshold

### Step 3: 验收对比
```bash
# 生成对比报告
python Model/compare_ablation_runs.py \
  --baseline runs/baseline_20260102 \
  --variant runs/physical_20260102 \
  --output ablation_physical_report.md
```

**对比内容**:
```bash
| Metric | Baseline | +Physical | Δ | 相对提升 |
|--------|----------|-----------|---|---------|
| OOF AUC | 0.7234 | 0.7301 | +0.0067 | +0.93% |
| Test AUC | 0.7156 | 0.7198 | +0.0042 | +0.59% |
| OOF F1 | 0.2341 | 0.2456 | +0.0115 | +4.91% |
| Test F1 | 0.2289 | 0.2367 | +0.0078 | +3.41% |
```

**解读**:
- 如果 Δ > 0 且显著 → 物理层特征有效 ✅
- 如果 Δ ≈ 0 → 物理层特征无贡献
- 如果 Δ < 0 → 需检查是否存在 leakage 或 overfitting

---

## 当前待实现的功能

由于时间限制，以下功能需要在 `train_v2.py` 中补充：

### 1. `--force-use-splits` 支持
**位置**: Step 3: Create Splits
```python
# 当前代码
splits_df = make_splits(
    df_full,
    ...
    save_splits_path=os.path.join(run_dir, 'splits.parquet'),
    force_resplit=False
)

# 修正后
if args.force_use_splits:
    # 强制使用指定的 splits
    print(f"  Loading splits from: {args.force_use_splits}")
    splits_df = pd.read_parquet(args.force_use_splits)
    
    # Copy to current run dir
    import shutil
    shutil.copy(args.force_use_splits, os.path.join(run_dir, 'splits.parquet'))
else:
    # 正常创建 splits
    splits_df = make_splits(...)
```

### 2. Hash Verification for Locked Mode
**位置**: Step 4: Compute Hashes 之后
```python
# 添加验证逻辑
if args.hyperparameter_mode == 'locked':
    # (见问题3的伪代码)
    ...
```

### 3. Threshold Locking Implementation
**位置**: Step 10: Save Predictions & Metrics
```python
# 替换硬编码的 0.5
if args.threshold_mode == 'locked':
    # (见问题4的伪代码)
    ...
```

### 4. Baseline Best Params Generation
**位置**: Step 10 末尾，在 save_artifact_summary() 之前
```python
# 如果是 baseline tuning mode，保存 baseline_best_params.json
if args.feature_set == 'baseline' and args.hyperparameter_mode == 'tuning':
    baseline_params = {
        'split_hash': split_hash,
        'dataset_hash': dataset_hash,
        'feature_transform_profile': args.feature_transform_profile,
        'imbalance_mode': args.imbalance_mode,
        'scale_pos_weight_scope': args.scale_pos_weight_scope,
        'threshold_mode': args.threshold_mode,
        'cv_metric': 'auc',  # or f1
        'tuning_trials': 1,  # TODO: actual tuning
        'best_params': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100
            # TODO: actual tuned params
        },
        'created_at': datetime.now().isoformat()
    }
    
    save_baseline_best_params(run_dir, baseline_params)
    print(f"  ✅ Saved baseline_best_params.json")
```

---

## 总结

**已修正**:
- ✅ 问题2: Feature set 定义 (experiment_utils.py)
- ✅ 问题6: Forbidden columns (experiment_utils.py)
- ✅ 问题1 (部分): 添加了 --force-use-splits 参数
- ✅ 问题4 (部分): 添加了 --threshold-mode locked 参数

**待补充代码** (约30分钟):
- ⏳ 问题1: 实现 --force-use-splits 的加载逻辑
- ⏳ 问题3: 实现 hash verification
- ⏳ 问题4: 实现 threshold locking
- ⏳ Baseline params generation

**使用建议**:
1. 先补完上述4个待实现功能
2. 用小数据集测试 baseline + +physical (n_folds=3)
3. 验证所有 hash 一致性
4. 确认无误后再跑完整实验 (n_folds=10)

**预计时间**:
- 补完代码: 30-45分钟
- 快速测试: 10-15分钟
- 完整实验: 40-50分钟（i9-13900K + RTX 4080）
