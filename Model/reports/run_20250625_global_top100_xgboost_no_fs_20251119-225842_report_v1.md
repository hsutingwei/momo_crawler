# Run Report: run_20250625_global_top100_xgboost_no_fs_20251119-225842

## 1. Run 基本資訊
- 
un_id: 0e331c5c-3ee6-413b-ac39-6ddf762a0be2
- label_mode: next_batch（v1_main）
- date_cutoff: 2025-06-25
- outdir: Model/outputs_fixedwindow_v2
- created_at: 2025-11-19 14:58:43Z

## 2. 資料摘要
- 
_samples: 1,756（pos=141, neg=1,615，pos_rate=0.080）
- 篩選條件：delta≥12, 
atio≥20%, max_gap≤14d, min_comments≥10, keyword_blacklist={口罩}。
- QA：
_train_0=1453, 
_train_1=127, 
_test_0=161, 
_test_1=14, valid_for_evaluation=True。

## 3. 模型與切分
- 模型：XGBoost (	ree_method=gpu_hist, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, 
eg_lambda=1.0)，未啟用 oversampling。
- 特徵：10 個 dense（媒體/互動/歷史 sales）＋ 100 個 TF tokens（global vocab）。
- 切分：StratifiedKFold (5 folds)；同一套 dataset 也在 QA 段落確認 train/test 仍有正負樣本。

## 4. 整體指標
- ROC-AUC：0.568
- PR-AUC：0.110
- Best threshold（依 F1_1）：θ=0.05  
  - precision_1=0.127  
  - recall_1=0.163  
  - F1_1=0.143  
  - specificity=0.902
- Top-K（validation predictions）：
  - P@20=0.150（R@20≈0.021）
  - P@50=0.180（R@50≈0.064）
  - P@100=0.140（R@100≈0.099）

## 5. 錯誤分析摘錄
- **False Positives**：集中於保健品（膠原蛋白、維他命）與寵物商品；多半帶有「物流快」「包裝好」等通用詞，顯示 TF-IDF 容易被通用好評吸引。
- **False Negatives**：出現在高價但評論較少的產品，上次補貨後銷售級距拉升時間落在 cutoff 後，但模型預估概率極低。
- **Feature Importance**：Top tokens包括「出貨」「快速」「回購」等物流／體驗字詞，dense features 中以 has_reply_content, comment_count_pre 以及 had_any_change_pre 排前。

## 6. 這個 run 告訴我的事
- v1_main 的 delta=12 對 PR-AUC 與 F1 仍偏嚴，模型大部分時間只學到「誰不會補貨」。
- 通用好評詞會誤判為補貨訊號，說明目前 TF-IDF 還沒掌握真實補貨語意。
- min_comments=10 讓資料較乾淨，但也讓 QA test fold 只剩 14 個正類，統計波動大。
- keyword=益生菌 在 θ=0.05 時 precision_1≈0.184 / recall_1≈0.259 / F1≈0.215，比全域平均好，顯示該類別較易學。
- 現有 dense 特徵對於「缺貨或超熱賣」的長期趨勢區分有限，需要更細的時間特徵。

## 7. 下一步實驗建議
1. **資料層**：導入固定視窗（W=7天）的標記 + time-based split，檢查是否能得到更穩定的高 recall；同時在 next_batch 版本調整 delta（依級距調整）。
2. **特徵層**：加入 rolling sales delta、評論密度與情緒特徵，並針對 keyword=益生菌 建立 per-keyword vocab，觀察 P@50 是否能破 0.2。
3. **模型層**：比較 LightGBM（GPU）與 class-weighted XGBoost，確認是否能在不 oversample 的情況下提高正類 recall。
4. **QA 與監控**：針對 test fold 正類少的問題，建立自動警示（若 
_test_1 < 20 即標記為需重跑），避免過度依賴不足的統計樣本。
