# train_v2.py 快速開始指南

## 🎯 train_v2.py vs train.py

### train.py (原版)
- ❌ 使用 date cutoff split (每次不同)
- ❌ 無 hash tracking
- ❌ 無 leakage checks
- ❌ 無 PostgreSQL logging
- ✅ 支援多算法 (XGBoost/SVM/LightGBM)

### train_v2.py (新版) ✨
- ✅ 固定 8:2 split + K-fold CV (可復現)
- ✅ Hash tracking (dataset/split/feature)
- ✅ 5 個自動化 leakage checks
- ✅ PostgreSQL logging
- ✅ 15+ artifacts per run
- ✅ Ablation study 支援
- ⏳ 目前只支援 XGBoost

---

## 🚀 基本用法

### 最簡單的執行方式

```bash
python Model/train_v2.py \
  --date-cutoff 2025-06-25 \
  --feature-set baseline
```

這會：
- 使用真實數據 (`load_product_level_training_set`)
- 創建固定 8:2 split (隨機種子=42)
- K-fold CV (10 folds)
- 產生 15+ artifacts
- 寫入 PostgreSQL

---

## 📋 完整參數範例

### Baseline 實驗

```bash
python Model/train_v2.py \
  --date-cutoff 2025-06-25 \
  --feature-set baseline \
  --feature-transform-profile phaseA \
  --n-folds 10 \
  --random-seed 42 \
  --run-id baseline_001 \
  --output-dir runs \
  --paper-mode
```

### Ablation Study - 添加 Physical Features

```bash
# 1. 先跑 baseline 並保存參數
python Model/train_v2.py \
  --feature-set baseline \
  --run-id baseline_001 \
  --hyperparameter-mode tuning

# 2. 鎖定參數，測試新特徵
python Model/train_v2.py \
  --feature-set +physical \
  --run-id ablation_physical_001 \
  --group-id ablation_group_001 \
  --hyperparameter-mode locked \
  --baseline-params-path runs/baseline_001/baseline_best_params.json
```

---

## 🔍 輸出結構

每個 run 會產生以下 artifacts：

```
runs/
└── baseline_001/
    ├── run_config.json          # 完整配置
    ├── splits.parquet           # 固定 8:2 splits
    ├── label.parquet            # 標籤
    ├── hashes.json              # 3 種 hash
    ├── samples_metadata.json    # 樣本統計
    ├── predictions_oof.parquet  # OOF 預測 (is_oof=1)
    ├── predictions_test.parquet # Test 預測 (is_oof=0)
    ├── metrics.json             # OOF + Test metrics
    ├── chosen_threshold.json    # Threshold 選擇
    ├── feature_list.json        # 活躍特徵
    ├── feature_hash_context.json # Feature hash context
    ├── imbalance_report.json    # scale_pos_weight per fold
    ├── leakage_checks.json      # Leakage 檢查結果
    ├── transformer/             # FeatureTransformer state
    └── artifact_summary.json    # Artifacts 摘要
```

---

## ⚖️ 類不平衡處理

### fold_train scope (預設，論文推薦)

```bash
python Model/train_v2.py \
  --imbalance-mode scale_pos_weight \
  --scale-pos-weight-scope fold_train
```

每個 fold 分別計算 `scale_pos_weight`（train fold only，不含 val）

### train_pool scope (所有 fold 一致)

```bash
python Model/train_v2.py \
  --imbalance-mode scale_pos_weight \
  --scale-pos-weight-scope train_pool
```

所有 fold 使用相同的 `scale_pos_weight`（整個 80% train pool）

---

## 🔐 Leakage Prevention

預設啟用 5 個自動化檢查：

1. ✅ Test 不混入 train fold
2. ✅ TF-IDF vocab 只來自 train
3. ✅ Clip bounds 只來自 train
4. ✅ Imputation values 只來自 train
5. ✅ Scaler 只在 train 上 fit

發現 leakage → 立即拋出 `ValueError` → 實驗中止

若要關閉 fail-fast：
```bash
python Model/train_v2.py --fail-on-leakage=False
```

---

## 📊 與 train.py 並行測試

### Step 1: 使用相同數據跑兩個版本

```bash
# train.py (原版)
python Model/train.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --algorithms xgboost \
  --cv 10

# train_v2.py (新版)
python Model/train_v2.py \
  --date-cutoff 2025-06-25 \
  --feature-set baseline \
  --n-folds 10
```

### Step 2: 比較結果

查看：
- Metrics 是否相近 (AUC, F1)
- 正負樣本比例
- Feature importance

### Step 3: 驗證可復現性

```bash
# 執行兩次 train_v2.py (相同參數)
python Model/train_v2.py --run-id test_run1 --random-seed 42
python Model/train_v2.py --run-id test_run2 --random-seed 42

# 比較 hashes (應完全一致)
cat runs/test_run1/hashes.json
cat runs/test_run2/hashes.json
```

---

## 💾 PostgreSQL 查詢

### 查看所有實驗

```sql
SELECT run_id, status, created_at, 
       feature_set, model_type, 
       metrics_json->'test'->>'auc' as test_auc
FROM experiment_runs
ORDER BY created_at DESC
LIMIT 10;
```

### 查看某個 run 的預測

```sql
SELECT product_id, y_true, y_prob, y_pred, split, fold
FROM experiment_predictions
WHERE run_id = 'baseline_001'
AND split = 'test'
ORDER BY y_prob DESC
LIMIT 20;
```

### 比較兩個 runs

```sql
SELECT 
    r.run_id,
    r.feature_set,
    r.metrics_json->'test'->>'auc' as test_auc,
    r.split_hash
FROM experiment_runs r
WHERE r.group_id = 'ablation_group_001'
ORDER BY r.created_at;
```

---

## 🐛 常見問題

### Q1: 如何使用自己的 splits？

```bash
# 第一次執行會生成 splits
python Model/train_v2.py --run-id exp1

# 後續執行會自動讀取相同的 splits (若 hash 一致)
python Model/train_v2.py --run-id exp2
# 會讀取 runs/exp1/splits.parquet
```

強制重新生成：修改 `--random-seed`

### Q2: 如何排除某些商品？

```bash
python Model/train_v2.py \
  --exclude-products 8918452,1234567
```

### Q3: 如何關閉 DB logging？

```bash
python Model/train_v2.py --enable-db-logging=False
```

### Q4: 出現 leakage error 怎麼辦？

檢查 `runs/{run_id}/leakage_checks.json`:
```json
{
  "all_passed": false,
  "failed_checks": 1,
  "checks": [
    {
      "check_name": "tfidf_vocab_isolation",
      "passed": false,
      "leakage_count": 5,
      "error": "❌ LEAKAGE: 5 test products in TF-IDF vocab source!"
    }
  ]
}
```

→ 修正數據載入流程，確保 TF-IDF 只在 train fold 上建立

---

## ✅ Next Steps

1. **測試 train_v2.py** - 跑一個小實驗
2. **比較結果** - 與 train.py 對比
3. **驗證可復現性** - 相同參數跑兩次
4. **Ablation study** - 測試 feature sets
5. **生產部署** - 確認無誤後替換 train.py

---

## 📚 相關文檔

- `ML_PIPELINE_SUMMARY.md` - 完整 pipeline 說明
- `implementation_plan.md` - 實作規格
- `test_end_to_end_pipeline.py` - E2E 測試

---

**有問題？** 查看 `Model/train_integration_example.py` 了解更多範例！
