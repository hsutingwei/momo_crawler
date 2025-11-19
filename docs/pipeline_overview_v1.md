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

## 2. `Model/data_loader.py`：標記模式與流程細節

### 2.1 標記模式（`label_mode`）

- `next_batch`（v1 預設）：以「評論批次」為時間軸，對齊最近且不晚於該批次的 sales snapshot，並檢查「下一個批次」是否上升。為了模擬「cutoff 之後的未來」，`batch_time > cutoff` 的批次才會計算 y=1。  
- `fixed_window`（Phase 2 追加）：不再依賴「下一個批次」，改為每個批次都用固定時間窗 `[t, t + label_window_days]` 搜尋 sales snapshot，只要在這個窗口內找到滿足 delta/ratio 的上升就標記為 y=1。這種做法讓不同商品的時間尺度一致，也不必強制依靠 `batch_time > cutoff` 來定義「未來」。

### 2.2 新增參數用途

| 參數 | 功能 |
| --- | --- |
| `label_delta_threshold` | 下一個批次銷售量需增加的絕對值門檻。 |
| `label_ratio_threshold` | 若設定，需同時達到指定百分比的相對增幅。 |
| `label_mode` | `next_batch` / `fixed_window`。 |
| `label_window_days` | `fixed_window` 模式下，往後搜尋 snapshot 的窗口長度（天）。 |
| `label_max_gap_days` | `next_batch` 模式下，允許 snapshot 與批次的最大時間差；`fixed_window` 模式用於限制「過舊」的參考 snapshot。 |
| `align_max_gap_days` | 對齊上一筆 snapshot 時可允許的最大差距；可避免把太久以前的 snapshot 當作參考值。 |
| `min_comments` | 保留在 cutoff 前至少 N 筆評論的商品（fixed_window 模式下不受 cutoff 限制）。 |
| `keyword_whitelist` / `keyword_blacklist` | 只使用（或排除）特定 keyword 的商品群。 |
| `train_end_date` / `val_end_date` | Time-based split 的切分點（僅在 fixed_window 模式下使用）。當提供這兩個參數時，會使用時間切分取代 StratifiedKFold。訓練集：`representative_batch_time <= train_end_date`；驗證集：`train_end_date < representative_batch_time <= val_end_date`；測試集：`representative_batch_time > val_end_date`。 |

### 2.3 單一商品的標記流程（pseudo）

1. 取得該商品的所有 `comment_batches`（distinct `capture_time`）。
2. 找出每個批次對應的 `sales_snapshots`：只取 snapshot_time ≤ 該批次時間，並選擇最近的一筆（`batch_repr`）。
3. 在 `seq` 中計算 `prev_sales`（上一批的銷售級距）。
4. y=1 條件：
   - `next_batch`（v1 預設）：
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
   - `fixed_window`（Phase 2+）：對每個 `batch_time = t`，先取得最近的 `prev_sales`（若與 t 差距大於 `align_max_gap_days` 則捨棄），再搜尋 `t < snapshot_time ≤ t + label_window_days` 的快照，只要在窗口內存在 `(sales_count - prev_sales) ≥ delta`（及 ratio 條件）即標記 1，否則 0。**Phase 3 更新**：
     - 取消 `cutoff` 過濾：`comment_batches`、`pre_comments`、TF-IDF pairs 的 SQL 都不再限制 `pc.capture_time <= cutoff`。
     - 新增 `representative_batch_time`：在 `y_post` CTE 中使用 `MIN(cb.batch_time)` 取得每個商品最早的 batch_time，並在 meta DataFrame 中包含此欄位。
     - 支援 time-based split：當提供 `--train-end-date` 和 `--val-end-date` 時，使用時間切分取代 StratifiedKFold。
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
3. **模型與資料切分**：
   - `build_model` 依 `alg_name` 建立 XGBoost / LightGBM / SVM（此 sweep 僅用 XGBoost）。
   - **資料切分策略**：
     - **預設（next_batch 模式或未提供時間參數）**：使用 `StratifiedKFold` 分成 `cv` 折，進行交叉驗證。
     - **Time-based split（fixed_window 模式 + 提供時間參數）**：
       - 當 `label_mode == "fixed_window"` 且提供 `--train-end-date` 和 `--val-end-date` 時，自動使用時間切分。
       - 切分規則：
         - 訓練集：`representative_batch_time <= train_end_date`
         - 驗證集：`train_end_date < representative_batch_time <= val_end_date`
         - 測試集：`representative_batch_time > val_end_date`
       - 此方式真正模擬「預測未來」的情境，避免 data leakage。
       - 使用 `representative_batch_time`（每個商品最早的 batch_time）作為切分依據，確保同一個商品不會同時出現在訓練集和測試集中。
   - 每折/每個 split 可選擇做 `lgbm_fs`、oversampling（random/SMOTE/scale_pos_weight）。
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
- **資料切分策略**：
  - **next_batch 模式**：使用 `StratifiedKFold` 進行交叉驗證。特徵生成時使用 `cutoff` 限制，避免 data leakage。
  - **fixed_window 模式（Phase 3 更新）**：
    - 預設：使用 `StratifiedKFold` 進行交叉驗證（向後兼容）。
    - 當提供 `--train-end-date` 和 `--val-end-date` 時：使用 time-based split（`train≤t1 / val≤t2 / test>t2`），真正模擬預測未來的情境。
    - 取消 cutoff 過濾，所有資料都參與特徵生成，但透過時間切分確保訓練時不會看到未來的資訊。
    - 使用 `representative_batch_time`（每個商品最早的 batch_time）作為切分依據。
- 全量評估，輸出含 PR-AUC、Top-K、threshold sweep、feature importance、錯誤樣本等資訊的 `model_run_v1.json`。

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

## 5. 實驗結果比較與 QA 檢查實例（Phase 3+）

### 5.1 問題案例：eval set 沒有正類

**舊 run (`run_20250625_global_top100_xgboost_no_fs_20251119-222709`)**：
- 問題：eval set 只剩 1 筆 y=0，完全沒有 y=1
- 結果：accuracy=1.0, f1_macro=1.0 看似很高，但其實實驗無效
- 原因分析：
  - 可能使用了過度過濾或時間區間太窄
  - 或使用了不當的時間切分點，導致 eval set 沒有正類樣本

**新 run (`fixed_window v2`, `run_20250625_global_top100_xgboost_no_fs_20251119-225342`)**：
- 配置：fixed_window 模式 + time-based split，min_comments=5（較寬鬆）
- 結果：
  - 訓練集：n_train_0=2814, n_train_1=24, pos_rate=0.84%
  - 測試集：n_test_0=9, n_test_1=0, pos_rate=0%
  - **QA 檢查失敗**：`valid_for_evaluation = False`
- 問題分析：
  - 時間切分點設置不當：val_end_date=2025-06-25 之後的數據太少（只有 9 個樣本）
  - 數據時間分布不均：大部分數據集中在 2025-04-28 到 2025-05-16，2025-06-25 之後的數據極少
  - 2025-06-25 之後的樣本全部是 y=0，沒有正類
- 改善建議：
  - 調整時間切分點，使用更早的切分點（例如 train_end_date=2025-05-10, val_end_date=2025-05-20）
  - 或改用 StratifiedKFold 進行交叉驗證，但保留 QA 檢查機制

### 5.2 QA 檢查機制驗證

新 run 成功觸發了 QA 檢查機制：
- 自動檢測到 test set 沒有正類樣本（n_test_1=0 < 10）
- 標記 `valid_for_evaluation = False`
- 將 metrics 設為 null，避免誤認為有效結果
- 在 log 中明確輸出警告訊息，包含詳細的 class 分布資訊

這證明了 QA 檢查機制的有效性，能夠及時發現並標記無效實驗。

### 5.3 與舊 run 的對比

| 項目 | 舊 run (next_batch, StratifiedKFold) | 新 run v1 (fixed_window, time-based) | 新 run v2 (fixed_window, StratifiedKFold) | 新 run v3 (next_batch, StratifiedKFold + QA) |
| --- | --- | --- | --- | --- |
| label_mode | next_batch | fixed_window | fixed_window | next_batch |
| 切分方式 | StratifiedKFold (10 折) | Time-based split | StratifiedKFold (10 折) | StratifiedKFold (10 折) |
| 總樣本數 | 約 6773 | 2847 | 2847 | 約 6773 |
| 正類總數 | 約 423 | 24 | 24 | 約 423 |
| 正類比例 | 約 6.2% | 0.84% | 0.84% | 約 6.2% |
| 訓練集樣本 | 約 6096 (平均每折: 609.6) | 2838 | 約 2562 (平均每折) | 約 6096 (平均每折) |
| 驗證集樣本 | 約 677 (平均每折) | 0 (val set 為空) | 約 285 (平均每折) | 約 677 (平均每折) |
| 測試集樣本 | 約 677 (平均每折) | 9 | 約 285 (平均每折) | 約 677 (平均每折) |
| 正類樣本 (eval) | 約 42.3 (平均每折) | 0 | 約 2.4 (平均每折) | 約 14.1 (平均每折) |
| QA 檢查 | ❌ 無 | ✅ 有 | ✅ 有 | ✅ 有 |
| valid_for_evaluation | ❌ 無此欄位 | ✅ False (正確標記) | ✅ False (正確標記) | ✅ True (通過檢查) |
| 問題發現 | ❌ 事後才發現 | ✅ 訓練時即發現 | ✅ 訓練時即發現 | ✅ 無問題 |
| 主要 metrics | PR-AUC≈0.109, F1≈0.142 | 無效（metrics=null） | 無效（metrics=null） | PR-AUC≈0.568, F1≈0.495 |

**關鍵發現**：

1. **舊 run (next_batch + StratifiedKFold)**：
   - 正類比例較高（約 6.2%），每個 fold 有約 42.3 個正類樣本
   - 沒有 QA 檢查機制，無法及時發現問題
   - 如果某個 fold 剛好沒有正類，會導致 accuracy=1.0 的假神模型

2. **新 run v1 (fixed_window + time-based split)**：
   - 時間切分點設置不當（val_end_date=2025-06-25），導致 test set 沒有正類（0 個）
   - 數據時間分布不均：大部分數據集中在 2025-04-28 到 2025-05-16，2025-06-25 之後的數據極少
   - QA 檢查成功發現問題並標記為無效

3. **新 run v2 (fixed_window + StratifiedKFold)**：
   - 即使使用 StratifiedKFold，每個 fold 也只有約 2.4 個正類樣本，低於 QA 檢查門檻（10 個）
   - QA 檢查成功發現問題並標記為無效
   - 這證明了 QA 檢查機制的必要性

**根本原因分析**：

1. **label_mode 的差異**：
   - `next_batch` 模式：正類比例約 6.2-7.7%，較容易滿足 QA 檢查要求
   - `fixed_window` 模式：正類比例極低（24/2847 = 0.84%），遠低於 next_batch 模式
   - 原因：fixed_window 模式的 label 定義更嚴格（需要 7 天內有顯著成長），導致正類樣本極少

2. **切分方式的影響**：
   - `StratifiedKFold`：可以保證每個 fold 的正類比例與整體一致，但當整體正類比例極低時，每個 fold 的正類樣本數仍可能不足
   - `time-based split`：需要仔細選擇切分點，如果數據時間分布不均或切分點設置不當，可能導致某些 split 沒有正類

3. **QA 檢查的重要性**：
   - 當正類比例極低時，即使使用 StratifiedKFold，每個 fold 的正類樣本數也可能不足
   - QA 檢查機制能夠及時發現並標記這類問題，避免將無效實驗誤認為有效結果

**改善建議**：
1. 對於 fixed_window 模式，可以考慮降低 QA 檢查門檻（例如 `--min-eval-pos-samples 3`），因為其正類比例本身就極低
2. 或調整 label 定義，使用較寬鬆的條件（例如降低 `label_delta_threshold` 或 `label_ratio_threshold`），以提高正類比例
3. 或使用 next_batch 模式，其正類比例較高（約 7.7%），更容易滿足 QA 檢查要求
4. 在選擇切分方式時，需要考慮數據的時間分布和正類比例，選擇最適合的切分策略

### 5.4 實驗配置建議

基於以上分析，建議的實驗配置：

**配置 A：next_batch 模式 + StratifiedKFold + QA 檢查（推薦用於 baseline）**
- 正類比例較高（約 6.2-7.7%），容易滿足 QA 檢查
- 使用 StratifiedKFold 進行交叉驗證
- 啟用 QA 檢查機制，確保實驗有效性
- 適合作為 baseline 實驗
- **實測結果**：新 run v3 通過 QA 檢查（`valid_for_evaluation = True`），每個 fold 有約 14.1 個正類樣本（超過門檻 10 個），實驗有效

**配置 B：fixed_window 模式 + StratifiedKFold + 降低 QA 門檻**
- 使用 fixed_window 模式確保時間尺度一致
- 使用 StratifiedKFold 進行交叉驗證
- 降低 QA 檢查門檻（例如 `--min-eval-pos-samples 3`）以適應低正類比例

**配置 C：fixed_window 模式 + time-based split（需謹慎選擇切分點）**
- 使用 fixed_window 模式 + time-based split 真正模擬預測未來
- 需要仔細選擇切分點，確保每個 split 都有足夠的正負類樣本
- 建議先分析數據的時間分布，選擇合適的切分點

---

> **維護提醒**：往後只要調整資料定義、特徵或訓練流程，請同步更新此文件（尤其是參數/流程說明與 sweep 設定），確保研究者能清楚追蹤每次實驗的語意與流程。
