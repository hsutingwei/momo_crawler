# momo_crawler 自動化研究流程 v1 說明

> 目標：記錄「資料從哪裡來、如何被標記與過濾、如何訓練與評估、輸出哪些成果」，讓研究者能快速對照程式碼並延伸實驗。

---

## 1. 資料流全貌（來源 ➜ 樣本 ➜ 特徵）

| 步驟 | 說明 | 主要檔案/程式 |
| --- | --- | --- |
| 1 | **來源資料**：PostgreSQL 中的 `products`、`product_comments`、`sales_snapshots`。 | DB schema |
| 2 | **對齊評論批次與銷售快照**：<br>1) 以 `comment_batches` 收集每個商品的 `capture_time`。<br>2) 針對每個批次找「最近且不晚於該批次」的 sales snapshot (`batch_repr`)。<br>3) 形成 `seq` 表，提供 `sales_count` 與 `prev_sales`。 | `Model/data_loader.py` (`sql_y` 區塊) |
| 3 | **標記 y**：依 CLI 參數決定 y=1 條件；v1 預設為 `sales_count - prev_sales >= label_delta_threshold`，若有 `label_ratio_threshold` 則再檢查相對增幅；同時限制 snapshot 與批次間隔 ≤ `label_max_gap_days`，並只看 `batch_time > date_cutoff` 的下一批。 | 同上 |
| 4 | **Dense 特徵**：在 cutoff 之前彙整評論媒體資訊、互動數與歷史銷售狀態（`media_agg`, `pre_seq`）。 | `Model/data_loader.py` (`sql_dense`) |
| 5 | **Pandas 篩選**：<br>・`keyword_whitelist` / `keyword_blacklist`<br>・`min_comments`（cutoff 前至少 N 筆評論）<br>將無法滿足條件的商品剔除。 | `Model/data_loader.py` |
| 6 | **TF vocab & 稀疏矩陣**：使用 cutoff 前評論的 TF-IDF 統計 (`fetch_top_terms`)，並在 `tfidf_scores` 取出 (product, token) 配對，建立 `X_tfidf`。 | `Model/data_loader.py` |
| 7 | **輸出給訓練程式**：回傳 `X_dense_df`（媒體/互動特徵）、`X_tfidf`、`y`、`meta`、`vocab`。`Model/train.py` 會把這些存成 CSV/NPZ，並寫 dataset manifest（含 class distribution、labeling/sampling config）。 | `Model/train.py` (`save_dataset_artifacts`) |

---

## 2. `Model/data_loader.py`：參數與流程細節

### 2.1 新增參數用途

| 參數 | 功能 |
| --- | --- |
| `label_delta_threshold` | 下一個批次銷售量需增加的絕對值門檻。 |
| `label_ratio_threshold` | 若設定，需同時達到指定百分比的相對增幅。 |
| `label_max_gap_days` | 限制 snapshot 與評論批次的最大允許時間差。 |
| `min_comments` | 保留在 cutoff 前至少 N 筆評論的商品。 |
| `keyword_whitelist` / `keyword_blacklist` | 只使用（或排除）特定 keyword 的商品群。 |

### 2.2 單一商品的標記流程（pseudo）

1. 取得該商品的所有 `comment_batches`（distinct `capture_time`）。
2. 找出每個批次對應的 `sales_snapshots`：只取 snapshot_time ≤ 該批次時間，並選擇最近的一筆（`batch_repr`）。
3. 在 `seq` 中計算 `prev_sales`（上一批的銷售級距）。
4. y=1 條件（以 v1 預設為例）：
   ```sql
   CASE WHEN batch_time > cutoff
          AND prev_sales IS NOT NULL
          AND (sales_count - prev_sales) >= :label_delta_threshold
          AND (
                :label_ratio_threshold IS NULL OR prev_sales = 0
                OR ((sales_count::float / NULLIF(prev_sales, 0)) - 1) >= :label_ratio_threshold
              )
          AND (
                :label_max_gap_days IS NULL
                OR EXTRACT(EPOCH FROM (snapshot_time - batch_time)) <= :label_max_gap_days * 86400
              )
        THEN 1 ELSE 0
   ```
5. 將結果 merge 回 dense 表；在 pandas 端依 `keyword_*`/`min_comments` 做最後篩選。
6. 以 `fetch_top_terms` 取得 vocab，並從 `tfidf_scores` 建立 `X_tfidf`。
7. 回傳 `(X_dense_df, X_tfidf, y, meta, vocab)` 給訓練流程。

---

## 3. `Model/train.py`：`run_one_setting` 流程重點

1. **載入資料**
   ```python
   X_dense_df, X_tfidf, y, meta, vocab = load_product_level_training_set(...)
   dataset_art = save_dataset_artifacts(..., labeling_config, sampling_config)
   ```
2. **組裝特徵**：Dense ➜ `std_scaler_to_sparse`，再與 `X_tfidf` hstack，並建立 feature 名稱（dense 欄位 + `tfidf::<token>`）。
3. **模型與交叉驗證**：
   - `build_model` 依 `alg_name` 建立 XGBoost / LightGBM / SVM（此 sweep 僅用 XGBoost）。
   - `StratifiedKFold` 分成 `cv` 折；每折可選擇做 `lgbm_fs`、oversampling（random/SMOTE/scale_pos_weight）。
   - 訓練後記錄 fold metrics (`eval_metrics`) 與逐筆 prediction（`product_id`, `keyword`, `y_true`, `y_pred`, `y_score`）。
4. **整體評估**：
   - 將所有驗證預測合併為 `df_predictions`。
   - `build_metrics_v1` 輸出：
     - `binary_overall`: ROC-AUC、PR-AUC、accuracy、Brier。
     - `by_class`: y=0 / y=1 的 precision / recall / F1。
     - `topk`: 依 `--topk` 列出 precision@k / recall@k。
     - `threshold_search`: 走訪 `--threshold-grid`，若 `--threshold-min-recall` > 0 先過濾，再用 `--threshold-target`（預設 F1_1）挑選最佳 threshold。
   - 重新用整份資料 fit 一次模型，以取得 feature importance（前 20 名）並挑出 top FP/FN。 
5. **輸出**：產生 fold/summary/predictions CSV、child manifest（記錄 dataset manifest、label/sampling config、metrics_v1、feature importance、error analysis），若有 metrics 會同步寫 `model_run_v1.json`，方便後續匯入 DB/Dashboard。

---

## 4. 研究者摘要與後續調整點

### 4.1 v1 Pipeline 的核心任務
- 以時間順序尊重的方式對齊評論批次與銷售快照，並依設定標記「下一批是否顯著增加」。
- 透過 CLI 參數控制 y 定義與樣本篩選，確保資料語意符合研究需要。
- 建立 dense + TF 特徵並儲存 dataset manifest，以便重現與紀錄資料策略。
- 交叉驗證 + 全量評估，輸出含 PR-AUC、Top-K、threshold sweep、feature importance、錯誤樣本等資訊的 `model_run_v1.json`。

### 4.2 想調整哪些部分？

| 需求 | 位置 |
| --- | --- |
| 修改 y 定義或樣本篩選 | `Model/train.py` CLI 參數（`label_*`, `min_comments`, `keyword_*`）+ `Model/data_loader.py`（SQL 與 pandas 篩選）。 |
| 增/改 dense 特徵 | `Model/data_loader.py` 的 `sql_dense` 與 `dense_cols`。 |
| 調整 vocab 或引入 embedding | `fetch_top_terms` / `Model/embeddings/*`。 |
| 改模型/評估流程（cv、threshold grid、Top-K） | `Model/train.py`：`build_model`, `run_one_setting`, `build_metrics_v1`。 |
| 新實驗（Exp A/B/C） | 依需求新增 loader/模型模組，並記得更新本文。 |

### 4.3 Label/Sampling Sweep 腳本
- `experiments/label_sweep.py` 集中維護多組 label/sampling 組合，並以 v1 baseline 的 XGBoost 訓練 pipeline 執行。輸出固定落在 `Model/outputs_v1_sweep/<config_name>`。
- 目前內建：`delta8_gap21`, `delta15_health`, `delta12_ratio20_min10`, `delta5_gap10_all`, `delta20_gap21_highvalue`。可依需要調整 `CONFIGS` 或另行新增。
- 執行：
  ```bash
  python experiments/label_sweep.py
  ```
  每個設定會產生自己的 `model_run_v1.json`，可直接比較 pos_rate、threshold 指標、Top-K precision/recall，挑選最合適的資料定義進入下一階段。

### 4.4 Sweep 結果摘要（v1_main / v1_alt）

| Config | label_delta | ratio | max_gap | min_comments | keyword | 商業語意 | pos_rate | PR-AUC | best θ (P/R/F1) | P@50 / R@50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `delta8_gap21` | 8 | – | 21 | 5 | blacklist=口罩 | 中度補貨、排除口罩 | 7.21% | 0.103 | 0.05 → 0.131 / 0.232 / 0.167 | 0.16 / 0.058 |
| `delta15_health` | 15 | – | 14 | 5 | whitelist=膠原蛋白/益生菌/維他命/葉黃素 | 保健品明顯補貨 | 6.35% | 0.094 | 0.05 → 0.101 / 0.180 / 0.129 | 0.16 / 0.072 |
| **`delta12_ratio20_min10` (v1_main)** | 12 | 20% | 14 | 10 | blacklist=口罩 | 高活躍且加速成長 | **7.74%** | **0.109** | **0.05 → 0.109 / 0.204 / 0.142** | **0.16 / 0.078** |
| **`delta5_gap10_all` (v1_alt)** | 5 | – | 10 | 3 | 全品類 | 輕度補貨、廣覆蓋 | 5.23% | **0.112** | **0.05 → 0.145 / 0.214 / 0.173** | **0.20 / 0.060** |
| `delta20_gap21_highvalue` | 20 | – | 21 | 8 | whitelist=膠原蛋白/維他命/雞精/寵物 | 高價商品大跳躍 | 7.44% | 0.063 | 0.05 → 0.039 / 0.077 / 0.052 | 0.00 / 0.00 |

- **v1_main** = `delta12_ratio20_min10`：兼具語意（真正熱度商品）與可學性（PR-AUC、F1、Top-K 均優於 baseline），後續主要分析 / Dashboard 以此定義為核心。  
- **v1_alt** = `delta5_gap10_all`：低門檻全品類，雖噪音較多但可觀察模型在「輕度補貨」情境的檢出力，適合作為對照組。

---

> **維護提醒**：往後只要調整資料定義、特徵或訓練流程，請同步更新此文件（尤其是參數/流程說明與 sweep 設定），確保研究者能清楚追蹤每次實驗的語意與流程。
