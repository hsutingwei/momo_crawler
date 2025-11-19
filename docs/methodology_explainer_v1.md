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

## 6. 後續建議

1. **Time-based split**：在 fixed_window 模式下取消 cutoff 過濾、改為 `train≤t1 / val≤t2 / test>t2` 的時間切分，以真正模擬「未來」。  
2. **特徵擴充**：加入 rolling sales、評論密度、情緒分數等，觀察對 v1_main/v1_alt 的 F1/P@K 是否有持續提升。  
3. **SOP 產出**：每次實驗需更新 `model_run_v1.json` 與本方法說明，讓 Dashboard/論文皆能追蹤到「標記與資料策略」。
