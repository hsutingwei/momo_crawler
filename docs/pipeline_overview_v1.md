# momo_crawler 自動化研究流程 v1 說明

> 目的：記錄「資料來源 ➜ 特徵/標記 ➜ 訓練與評估 ➜ model_run」完整資料流，方便研究者對照程式碼並延伸新的實驗。

---

## 1. 資料流：來源到訓練輸入

| 步驟 | 說明 | 相關檔案/程式 |
| --- | --- | --- |
| 1 | **資料來源**：`products`, `product_comments`, `sales_snapshots`（PostgreSQL）。 | DB schema |
| 2 | **對齊評論批次與銷售快照**：<br>1) 取每個商品的 `comment_batches`（distinct `capture_time`）<br>2) 對每批次找「最近且不晚於」的 `sales_snapshots` (`batch_repr`)<br>3) 建 `seq`，含 `sales_count` 與 `prev_sales`。 | `Model/data_loader.py` (`sql_y`, `batch_repr`, `seq`) |
| 3 | **標記 y**：依 CLI 參數決定 y=1 條件：<br>・`sales_count - prev_sales >= label_delta_threshold`<br>・若有 ratio 門檻則 `(sales_count/prev_sales - 1) >= label_ratio_threshold`<br>・下一個 snapshot 與該批次間隔 ≤ `label_max_gap_days`<br>・且 `batch_time > date_cutoff`（只看 cutoff 之後）。 | `Model/data_loader.py` |
| 4 | **Dense 特徵**：在 cutoff 之前聚合評論媒體/互動（`media_agg`）、歷史 sales 狀態（`pre_seq`）。 | `Model/data_loader.py` |
| 5 | **Pandas 篩選**：<br>・`keyword_whitelist` / `keyword_blacklist`<br>・`min_comments`<br>結果為 `df`（含 y）。 | `Model/data_loader.py` |
| 6 | **TF vocab 與稀疏矩陣**：以 cutoff 前評論計算 top-N tokens (`fetch_top_terms`)，並在 `tfidf_scores` 抽出 (product, token) 建 `X_tfidf`。 | `Model/data_loader.py` |
| 7 | **輸出給訓練**：<br>・`X_dense_df`: 媒體/互動/歷史 summary<br>・`X_tfidf`: token presence<br>・`y`: 0/1<br>・`meta`: `product_id/name/keyword`（後續會附上實驗資訊）<br>這些同時在 `save_dataset_artifacts()` 中被存成 CSV/NPZ 並生成 manifest（含 labeling/sampling config 與 class distribution）。 | `Model/train.py` |

---

## 2. `Model/data_loader.py` 的參數與流程

### 2.1 新增參數

| 參數 | 用途 |
| --- | --- |
| `label_delta_threshold` | 下個批次銷售需增加的絕對值（≥此值才算 y=1）。 |
| `label_ratio_threshold` | 相對增幅門檻（可為 None）。 |
| `label_max_gap_days` | 允許的 snapshot 與評論批次最大時間差。 |
| `min_comments` | 保留在 cutoff 前累積至少 N 筆評論的商品。 |
| `keyword_whitelist` / `keyword_blacklist` | keyword 白/黑名單。 |

### 2.2 單一商品的處理示意（pseudo-step）

1. `comment_batches`：列出此商品所有 `capture_time`。  
2. `sales_snapshots` ⇒ `batch_repr`：對每個 batch 找「最近且不晚於」的 snapshot，保留 `snapshot_time`。  
3. `seq`：為每個 batch 計算 `prev_sales`（上一批）。  
4. y=1 條件（以 v1 為例）：
   ```sql
   CASE WHEN batch_time > cutoff
          AND prev_sales IS NOT NULL
          AND (sales_count - prev_sales) >= 10
          AND (
                :ratio_threshold IS NULL OR prev_sales = 0
                OR ((sales_count::float / NULLIF(prev_sales, 0)) - 1) >= :ratio_threshold
              )
          AND (
                :max_gap_seconds IS NULL
                OR EXTRACT(EPOCH FROM (snapshot_time - batch_time)) <= :max_gap_seconds
              )
        THEN 1 ELSE 0
   ```
5. merge 到 dense table → pandas 篩選 `min_comments`、keyword 名單。  
6. `fetch_top_terms` + `tfidf_scores` 生成 `X_tfidf`。  
7. 回傳 `(X_dense_df, X_tfidf, y, meta, vocab)`。

---

## 3. `Model/train.py` → `run_one_setting` 流程

1. **載入資料**  
   ```python
   X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(...)
   dataset_art = save_dataset_artifacts(..., labeling_config, sampling_config)
   ```
2. **特徵組裝**  
   - Dense → `std_scaler_to_sparse`  
   - `X_all = hstack([X_dense, X_tfidf])`  
   - `feature_names = dense_cols + [f"tfidf::{token}"]`
3. **模型訓練與 K-fold**  
   ```python
   model, needs_dense = build_model(alg_name, args)
   skf = StratifiedKFold(...)
   for fold, (tr_idx, va_idx) in skf.split(...):
       # optional FS, oversample
       model.fit(...)
       y_pred, y_score = model.predict(...)
       fold_metrics = eval_metrics(...)
       record per-sample predictions
   ```
4. **整體評估**  
   - `df_predictions = concat(all_predictions)`  
   - `build_metrics_v1(...)`  
     - `binary_overall`: ROC-AUC, PR-AUC, accuracy, Brier  
     - `by_class`: precision/recall/F1 (0/1)  
     - `topk`: precision/recall@k（由 `--topk` 決定）  
     - `threshold_search`: 在 `--threshold-grid` 掃描；若有 `--threshold-min-recall`，先過濾，再用 `--threshold-target`（預設 F1_1）挑最佳 threshold。  
   - 重新 fit 全資料一次 → `extract_feature_importance`（top 20），`collect_error_examples`（FP/FN top 5）。  
5. **輸出**  
   - Fold metrics / summary / predictions CSV。  
   - Child manifest (`run_*_manifest.json`) 附帶：dataset manifest、label/sampling config、target_focus、metrics_v1、feature importance、error_analysis。  
   - 若 metrics 存在 → `run_*_model_run_v1.json`（`build_model_run_payload`）。  
   - Run-level manifest (`*_RUN_manifest.json`) 彙整 dataset 與所有子實驗。

---

## 4. 給研究者的摘要 & 後續改動位置

### 4.1 v1 pipeline 在做的事
- 以時間序尊重的方式，從 DB 對齊評論批次與銷售快照。  
- 使用 CLI 參數決定 y 的定義與樣本篩選（絕對增量/相對增幅/時間窗/keyword/min comments）。  
- 產生 dense + TF features，保存 dataset manifest（含 class distribution、label/sampling config）。  
- 進行 K-fold 訓練，紀錄每筆 validation prediction，輸出包含：  
  - Binary metrics（PR-AUC、by-class precision/recall/F1、Brier）  
  - Top-K precision/recall  
  - Threshold sweep 結果  
  - Feature importance + error samples  
  - `model_run_v1.json` 方便後台/DB 採集。

### 4.2 之後要調整的定位

| 想做的事 | 檔案/段落 |
| --- | --- |
| 改 y 定義（delta/ratio/time window）或樣本篩選（keyword/min_comments） | `Model/train.py` CLI 參數 + `load_product_level_training_set`（SQL + pandas） |
| 增/改 dense 特徵 | `Model/data_loader.py` SQL 聚合段（`sql_dense`）與 `dense_cols`。 |
| 文本 vocab 規則或 embedding | `fetch_top_terms` / `train_with_embeddings.py`（若使用 embedding pipeline）。 |
| 調整模型/評估（cv、Top-K、threshold grid） | `Model/train.py` (`build_model`, `run_one_setting`, `build_metrics_v1`)。 |
| 新實驗（Exp A/B/C） | 導入新的資料 loader 或模型模組，並記得更新此文檔以維持 traceability。 |

---

> **維護提醒**：未來若有流程/參數變動，請同步更新此檔，以確保所有研究者能快速理解「資料如何被標記、篩選、訓練與評估」。  
