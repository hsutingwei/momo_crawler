# momo_crawler 方法說明 v2

## 1. 研究任務與背景
momo_crawler 的研究任務是針對各商品「評論語意／互動訊號」與「銷售級距變化」的關聯進行實驗化驗證。我們必須同時：
- 在資料層面建出可持續迭代的標籤邏輯，讓模型可以真正預測「下一次補貨」。
- 在模型層面維持可重現的 model_run JSON，提供 Dashboard 與論文撰寫使用。
- 在流程層面確保 MLOps discipline：所有程式、特徵與評估必須有版本紀錄、可追溯。

## 2. y 值定義的版本與動機
| 版本 | 條件 | 商業語意 | 觀察到的現象（Pos rate / PR-AUC / best F1_1） |
| --- | --- | --- | --- |
| **v1_main（next_batch）** | delta≥12, 
atio≥20%, max_gap≤14天, min_comments≥10, 排除口罩 | 高活躍商品的「下一次補貨是否顯著上升」 | pos_rate ≈ 0.077；PR-AUC ≈ 0.11；最佳 θ=0.05 時 precision≈0.11 / recall≈0.20 / F1≈0.14 |
| **v1_alt（next_batch 對照）** | delta≥5, 
atio不限, max_gap≤10天, min_comments≥3, 全品類 | 輕度補貨、廣覆蓋的偵測任務 | pos_rate ≈ 0.052；PR-AUC ≈ 0.112；最佳 θ=0.05 時 precision≈0.145 / recall≈0.214 / F1≈0.173 |
| **fixed_window（高精度 alert）** | label_mode=fixed_window, window=7天, align_gap≤1天, delta≥12, 
atio≥20% | 將每個批次的補貨定義在固定視窗內，做「高確度補貨警示」 | pos_rate ≈ 0.01；PR-AUC ≈ 0.88；最佳 θ≈0.45 時 precision≈0.75 / recall≈0.92 / F1≈0.83 |

## 3. 數據集（SQL）為什麼要這樣取得
1. **comment_batches + sales_snapshots 對齊**：利用 LATERAL join 找「最近且不晚於評論批次」的 sales snapshot，確保評論語意與當下銷售一致；若拿掉就會混用非常久以前的數值。
2. **cutoff / window 設計**：next_batch 需要 batch_time > cutoff 來定義「未來」；fixed_window 則透過 [t, t+W] 視窗判斷補貨，並用 align_max_gap_days 限制上一筆 snapshot 與批次時間差。
3. **DB-side 聚合**：pre_comments, media_agg, pre_seq 在 SQL 層做 groupby，維持計算一致、避免 pandas 記憶體爆炸，也讓 Dashboard 可復用。
4. **風險**：若移除 cutoff 節點，fixed_window 也會看到未來評論；若拿掉 align_max_gap_days，冷門商品可能引用一年前的銷售數據當 baseline。

## 4. 取樣與過濾的設計理由
- keyword_blacklist（口罩）避免被噪音極大的商品干擾；keyword_whitelist 用於專題分析（如保健品）。
- min_comments 確保樣本對模型有足夠訊號，但也會偏向熱門商品。
- 時間範圍（cutoff 或窗口）避免資料洩漏或引用過舊資料。

## 5. 特徵設計與未來擴充
- 現況：10 個 dense（價格、媒體、互動、歷史 sales）＋ global top-N TF tokens。
- 未來：加入 rolling sales delta、評論密度、情緒/互動比率、客服回覆延遲，以及 per-keyword 的客製特徵。

## 6. 資料切分與評估策略
- next_batch 目前用 StratifiedKFold，cutoff 來界定特徵/標記的時間順序。
- fixed_window 需要 time-based split；接下來會在 train.py 裡加入指定的時間分區，以避免窗口跨折。
- QA 機制：model_run_v1.json 的 qa_check 紀錄 train/test 的正負樣本數，一旦不足會標記 valid_for_evaluation=false。

## 7. 代表性實驗結果摘要
| run | label_mode | pos_rate | PR-AUC | best F1_1 (θ) | P@50 / P@100 | 備註 |
| --- | --- | --- | --- | --- | --- | --- |
| 20250823-130024 | v1_main / next_batch | 0.077 | 0.109 | 0.142 (θ=0.05) | 0.16 / 0.13 | 主線任務。 |
| 20250823-130501 | v1_alt / next_batch | 0.052 | 0.112 | 0.173 (θ=0.05) | 0.20 / 0.11 | 低門檻對照。 |
| 20251119-225842 | fixed_window (7d) | 0.010 | 0.876 | 0.828 (θ=0.45) | 0.24 / 0.14 | 高精度 alert。 |

## 8. 設計取捨與未來方向
- 為何 next_batch 暫時為主：語意直接對應下一次補貨且樣本量充足；fixed_window 暫擔任延伸警示。
- delta=12 在低級距限制大，未來要依級距調整 delta。
- 接下來會加入 time-based split、rolling features、keyword-specific 分析等。

## 9. Changelog (v1 → v2)
1. 新增 label_mode，支援 fixed_window 與 align_max_gap_days，打造高精度 alert 流程。
2. model_run 增加 QA 機制欄位。
3. 推出 per-keyword performance 報表（例如 keyword=益生菌）。
