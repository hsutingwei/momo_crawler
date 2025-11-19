# momo_crawler 方法說明 v1（截至 Phase 2 固定視窗實驗）

## 1. y 定義的動機與版本

### 1.1 v1_main（next_batch 模式）
- **定義**：針對每個商品，每次評論批次 `t` 皆對齊「最近且不晚於 t 的 sales snapshot」，再比較「下一個評論批次」的 sales_count 是否成長。若 `(next_sales - cur_sales) ≥ delta (=12)` 且 `next_sales / cur_sales ≥ 1.2`，則標記 `y=1`。
- **理由**：貼近「下一次爬蟲時是否會看到補貨」的實務問題；高活躍商品（評論頻率高）能精細觀察補貨波段。
- **參數**：`label_delta_threshold=12`, `label_ratio_threshold=0.2`, `label_max_gap_days=14`, `min_comments=10`, `keyword_blacklist=口罩`。
- **現象**：正類比例約 7.7%，PR-AUC ≈ 0.109；最佳 threshold (0.05) 時 precision ≈0.11、recall ≈0.20、F1 ≈0.14。

### 1.2 v1_alt（next_batch，低門檻對照）
- **定義**：`delta=5`, `ratio=None`, `max_gap=10`, `min_comments=3`, 無 keyword 限制。
- **理由**：觀察「輕度補貨」情境；作為 v1_main 的對照。
- **現象**：pos_rate ≈5.2%，PR-AUC ≈0.112；最佳 threshold 時 precision ≈0.145、recall ≈0.214、F1 ≈0.173（recall 高但噪音較多）。

### 1.3 新增的 fixed_window 模式
- **問題**：next_batch 會受評論頻率影響；「下一次」的時間長度不一致，且需 `batch_time > cutoff` 來界定「未來」。
- **改法**：改成固定視窗 `[t, t+W]`，只要在此視窗內有任一 snapshot 使 `sales - prev_sales ≥ delta (12)` 且 ratio ≥0.2，就判定 `y=1`；`align_max_gap_days=1` 限制 “上一筆可用 snapshot 與批次 t 的距離”。
- **效果**：在 v1_main 參數下（W=7, align=1），pos_rate ≈0.98%，但 PR-AUC ≈0.876，最佳 threshold 0.45 時 precision 0.75 / recall 0.92 / F1 0.83；P@50 0.24 / R@50 0.92。顯示 y=1 更集中，模型更容易檢出，但也代表樣本只剩小量高置信度事件，適合作為「高確度補貨信號」。

## 2. 資料集 SQL 的關鍵條件

1. **comment_batches**：取得產品每個 `capture_time`；限制在 cutoff 之前，以免特徵洩漏。  
2. **snap_mapped / prev_snap**：使用 LATERAL 找「最近且不晚於批次」的 sales snapshot，並用 `align_max_gap_days` 控制時間差，避免把太久之前的銷售當作現狀。  
3. **future_snap / y_post**：  
   - `next_batch`：比較「下一個批次」的 snapshot；`batch_time > cutoff` 用來定義「未來」。  
   - `fixed_window`：比較 `[t, t+W]` 內的 snapshot；不依賴「下一批」，而是用固定視窗判斷。  
4. **pre_comments / media_agg / pre_seq**：在 DB 層彙整媒體、互動、歷史銷售趨勢，避免 pandas 做大量 groupby；可減少記憶體並確保同一套邏輯在 DB/Dashboard 的一致性。  
5. **TF vocab**：在 DB 端先取得 top-N tokens，再用 comment_id/token 建 sparse matrix。

## 3. 取樣與過濾

- `keyword_blacklist` / `keyword_whitelist`：排除口罩、聚焦保健品等是為了降低噪音（口罩補貨太頻繁且不具代表性）。  
- `min_comments`: 確保模型看到的商品至少有 N 筆評論；資料不足的商品無法提供穩定訊號，容易被少數事件主導。  
- 這些條件會造成偏誤（偏向熱門商品），但換來更穩定的訓練樣本；必要時可針對冷門商品另設一組實驗。

## 4. 設計取捨與替代方案

- **next_batch vs fixed_window**：  
  - next_batch 語意貼近「下一輪爬蟲」，但不同商品的時間尺度不一致；  
  - fixed_window 保證時間尺度一致，但需要再設法做 time-based split 來維持預測語意。  
  - Phase 2 先同時保留兩種（透過 `--label-mode`）。
- **cutoff**：  
  - 目前仍用 cutoff 來限制特徵生成（避免看未來）；  
  - fixed_window 的 y 不再依賴 cutoff，未來可考慮以 rolling split 完全替代。
- **其他未採用方案**：  
  1. 直接用每日對齊（每日 snapshot vs 每日評論）：樣本量倍增，但資料缺漏（有時無快照）會導致大量補值。  
  2. 將所有商品合併成全局時間序列：跨商品可能會互相影響（不同 keyword 的補貨節奏不同），暫時不採用。

## 5. 實驗結果摘要（本輪）

| setting | label_mode | window/gap | delta/ratio | keyword/min_comments | pos_rate | PR-AUC | best θ (P/R/F1) | P@50/R@50 | 語意 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1_main | next_batch | gap ≤14天 | 12 & 20% | blacklist=口罩, min_comments=10 | 0.077 | 0.109 | 0.05 → 0.109 / 0.204 / 0.142 | 0.16 / 0.078 | 高活躍且成長的商品。 |
| v1_alt | next_batch | gap ≤10天 | 5 & – | 全品類, min_comments=3 | 0.052 | 0.112 | 0.05 → 0.145 / 0.214 / 0.173 | 0.20 / 0.060 | 輕度補貨，廣覆蓋。 |
| v1_main_fixedwindow | fixed_window | W=7天, align=1天 | 12 & 20% | 同 v1_main | 0.0098 | 0.876 | 0.45 → 0.75 / 0.923 / 0.828 | 0.24 / 0.923 | 高確度補貨信號（樣本極少）。 |

## 6. Time-based Split 實作（Phase 3）

### 6.1 動機與設計
- **問題**：fixed_window 模式雖然解決了時間尺度不一致的問題，但原本仍依賴 `cutoff` 來限制特徵生成，且使用 `StratifiedKFold` 隨機切分，無法真正模擬「預測未來」的情境。隨機切分會讓訓練集看到未來的資料，造成 data leakage，無法真實反映模型的預測能力。
- **解決方案**：在 fixed_window 模式下實作 time-based split，取消 cutoff 過濾，改為基於 `representative_batch_time` 的時間切分：
  - **訓練集**：`representative_batch_time <= train_end_date`
  - **驗證集**：`train_end_date < representative_batch_time <= val_end_date`
  - **測試集**：`representative_batch_time > val_end_date`
- **實作細節**：
  - **data_loader.py 修改**：
    - fixed_window 模式的 SQL 中，`comment_batches` CTE 不再限制 `pc.capture_time <= cutoff`，改為返回所有資料。
    - `sql_dense` 中的 `pre_comments` CTE 也取消 cutoff 過濾（使用條件判斷 `cutoff_filter`）。
    - `pre_seq` 中的 `had_any_change_pre` 和 `num_increases_pre` 計算時，在 fixed_window 模式下改為 `TRUE` 條件（不再限制 batch_time）。
    - TF-IDF pairs 的 SQL 也取消 cutoff 過濾。
    - `y_post` CTE 中新增 `MIN(cb.batch_time) AS representative_batch_time`，取每個商品最早的 batch_time 作為代表時間。
    - 在 meta DataFrame 中合併 `representative_batch_time`，並確保時間格式為 UTC-aware。
  - **train.py 修改**：
    - 新增參數 `--train-end-date` 和 `--val-end-date`。
    - 在 `run_one_setting` 中，當 `label_mode == "fixed_window"` 且提供時間參數時，自動使用 time-based split 取代 `StratifiedKFold`。
    - 使用 `representative_batch_time` 進行時間切分，確保訓練時不會看到未來的資訊。
    - 在 `model_run_v1.json` 的 `data_spec.split_strategy` 中記錄使用的切分策略。
- **參數**：
  - `--train-end-date`：訓練集結束日期（格式：YYYY-MM-DD）
  - `--val-end-date`：驗證集結束日期（格式：YYYY-MM-DD）
  - 僅在 `--label-mode fixed_window` 時生效
  - `--date-cutoff` 在 fixed_window 模式下仍可提供，但不會用於資料過濾（僅作為 metadata）

### 6.2 與原有設計的差異
- **next_batch 模式**：仍使用 `cutoff` 限制特徵生成（`pc.capture_time <= cutoff`），並使用 `StratifiedKFold` 進行交叉驗證。這種方式雖然避免了特徵洩漏，但隨機切分仍可能讓訓練集看到未來的樣本。
- **fixed_window + time-based split**：
  - 取消 cutoff 過濾，所有資料都參與特徵生成（包括未來的資料），但透過時間切分確保訓練時不會看到未來的資訊。
  - 使用 `representative_batch_time`（每個商品最早的 batch_time）作為切分依據，確保時間順序的一致性。
  - 真正模擬「預測未來」的情境，訓練集只能看到過去的資料，驗證集和測試集代表未來的資料。

### 6.3 設計理由與取捨
- **為什麼使用 `representative_batch_time`（最早的 batch_time）**：
  - 每個商品可能有多個 batch_time，選擇最早的可以確保該商品的所有歷史資訊都在訓練集中，避免同一個商品同時出現在訓練集和測試集。
  - 如果使用最新的 batch_time，可能會導致商品在訓練集和測試集中重複出現，造成 data leakage。
- **為什麼在 fixed_window 模式下取消 cutoff 過濾**：
  - fixed_window 模式的 y 標記不再依賴 cutoff（而是使用固定時間窗），因此可以安全地使用所有資料來生成特徵。
  - 透過 time-based split 確保訓練時不會看到未來的資訊，同時最大化資料利用率。
- **與 next_batch 模式的對比**：
  - next_batch 模式依賴 `batch_time > cutoff` 來定義「未來」，因此必須在 SQL 層面限制特徵生成。
  - fixed_window 模式不依賴 cutoff，可以在 Python 層面進行時間切分，更靈活且更符合預測情境。

### 6.4 使用範例
```bash
python Model/train.py \
  --mode product_level \
  --label-mode fixed_window \
  --label-delta-threshold 12 \
  --label-ratio-threshold 0.2 \
  --label-window-days 7 \
  --align-max-gap-days 1 \
  --min-comments 10 \
  --keyword-blacklist 口罩 \
  --train-end-date 2025-06-20 \
  --val-end-date 2025-06-25 \
  --date-cutoff 2025-06-25 \
  --algorithms xgboost \
  --top-n 100
```

### 6.5 注意事項
- 確保 `train_end_date < val_end_date`，否則會導致驗證集為空。
- `representative_batch_time` 為 NULL 的商品會被排除在切分之外（但這種情況應該很少見）。
- 如果沒有提供 `--train-end-date` 和 `--val-end-date`，fixed_window 模式仍會使用 `StratifiedKFold`（保持向後兼容）。

## 7. 後續建議

1. **特徵擴充**：加入 rolling sales、評論密度、情緒分數等，觀察對 v1_main/v1_alt 的 F1/P@K 是否有持續提升。  
2. **SOP 產出**：每次實驗需更新 `model_run_v1.json` 與本方法說明，讓 Dashboard/論文皆能追蹤到「標記與資料策略」。

## 8. QA 檢查與實驗有效性驗證（Phase 3+）

### 8.1 問題案例：假神模型

在實驗過程中，我們曾經遇到「eval set 沒有任何 y=1，導致 accuracy=1.0 看似很高但其實實驗無效」的案例（例如 `run_20250625_global_top100_xgboost_no_fs_20251119-222709`）。這種情況可能發生在：

- **time-based split 時時間區間太窄**：如果 `train_end_date` 和 `val_end_date` 太接近，或 `val_end_date` 之後的時間區間剛好沒有 y=1 的商品，會導致 eval set 沒有正類。
- **過度過濾**：如果使用了過嚴格的過濾條件（例如 `min_comments` 太高、keyword 限制太嚴格），可能導致樣本太少，某些時間區間完全沒有正類。
- **label 定義太嚴格**：如果 label 定義太嚴格（例如 `delta=12` 對低級距商品太嚴格），可能導致某些時間區間完全沒有 y=1。

### 8.2 改善措施

為防止這類問題，我們在 pipeline 中新增了以下機制：

1. **Class 分布 QA 檢查**：在 train/val/test 切分完成後，自動檢查每個 eval set 的 class 分布。預設要求至少 10 個正類和 10 個負類樣本（可透過 `--min-eval-pos-samples` 和 `--min-eval-neg-samples` 調整）。

2. **`valid_for_evaluation` 標記**：如果 QA 檢查失敗，會在 summary CSV 和 `model_run_v1.json` 中標記 `valid_for_evaluation = false`，並將主要 metrics 設為 null，避免將無效實驗誤認為有效結果。

3. **新的實驗設定**：設計了 `fixed_window v2` 實驗配置（見 `experiments/run_fixedwindow_v2.py`），使用較寬鬆的過濾條件和足夠大的時間區間，確保 eval set 有足夠的正負類樣本。

這些措施確保了實驗結果的有效性和可重現性，避免了「假神模型」的問題。
