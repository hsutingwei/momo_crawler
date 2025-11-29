# Feature Engineering Plan for Growth & Explosive Tasks

## 1. 研究主角與 baseline
- **成長任務 (Baseline)**：`hybrid_strict_balanced` (delta>=10, ratio>=0.3)
- **爆品任務 (Target)**：`binary_explosive` (delta>=10, ratio>=1.0)
- **共用設定**：Class Weight + Threshold Tuning (Standardized in Phase 2)

## 2. Temporal 特徵優化方向（優先級高）
根據錯誤分析，`days_since_last_comment` 與 `comment_count_90d` 已證明有效。為了更早偵測爆發跡象，我們需要捕捉「加速度」與「趨勢變化」。

### 2.1 趨勢與加速度 (Trend & Acceleration)
- **概念**：爆品在爆發前通常會有評論數的「爬升期」。
- **擬新增特徵**：
  - 以週為單位建立近 4 週評論數序列：`[count_w1, count_w2, count_w3, count_w4]`
  - `slope_4w`：近 4 週評論數的線性回歸斜率。
  - `acc_4w`：斜率的變化率（加速度）。
  - `volatility_4w`：評論數的標準差，衡量熱度穩定性。

### 2.2 分段比較 (Segmented Ratios)
- **概念**：比較「最近一個月」與「前兩個月」的熱度差異。
- **擬新增特徵**：
  - `comment_1st_30d` (最近 30 天)
  - `comment_2nd_30d` (31-60 天前)
  - `comment_3rd_30d` (61-90 天前)
  - `ratio_recent30_to_prev60` = `comment_1st_30d` / (`comment_2nd_30d` + `comment_3rd_30d` + 1e-6)

## 3. Content / Keyword 特徵細緻化（優先級中）
目前的 `sentiment_mean_recent` 與 `repurchase_ratio_recent` 有效，但顆粒度較粗。錯誤分析顯示 FN 樣本中常出現特定類型的正面評價。

### 3.1 關鍵字類別細分 (Granular Keyword Categories)
- **概念**：不同類型的爆品有不同的評價模式（如「便宜好用」vs「效果驚人」）。
- **擬新增特徵**：
  - **功效／效果詞 (Efficacy)**：`有感|改善|超好用|有效|推薦`
  - **價格／CP 值詞 (Price/Value)**：`便宜|划算|CP值|特價|優惠`
  - **物流／包裝詞 (Logistics)**：`出貨快|包裝完整|快速`
  - **回購／囤貨詞 (Repurchase)**：`回購|囤貨|再買|買爆|忠實` (延伸既有集合)
- **計算方式**：
  - 對每一類別計算近 90 天的 `count` 與 `ratio`。
  - 例如：`efficacy_ratio_recent`, `price_ratio_recent`。

### 3.2 負評細分 (Granular Negative Sentiment)
- **概念**：區分「產品爛」與「物流慢」的負評。物流慢可能不影響產品本身的爆發潛力，但產品爛會。
- **擬新增特徵**：
  - `neg_product_ratio`：包含「難用|無效|失望」的負評比例。
  - `neg_logistics_ratio`：包含「慢|破损|態度差」的負評比例。

## 4. Embedding / Topic 方向（延伸研究）
若上述特徵仍有瓶頸，可引入更深層的語意特徵。

### 4.1 Sentence Embeddings
- **方法**：使用輕量級預訓練模型 (e.g., `paraphrase-multilingual-MiniLM-L12-v2`)。
- **特徵**：
  - 對近 90 天所有評論的 embedding 取平均 (Mean Pooling)。
  - 使用 PCA 降維至 16~32 維，作為 Dense Features 輸入 XGBoost。

### 4.2 Topic Modeling (Offline)
- **方法**：使用 BERTopic 或 LDA 對全站評論進行分群。
- **特徵**：
  - `top_topic_id`：該商品評論最常出現的主題 ID。
  - `topic_distribution`：各主題的佔比向量。

## 5. 實作規劃 (Implementation Roadmap)
1.  **Phase 2.1**: 實作 Temporal Trend 特徵 (Slope, Acceleration)。
2.  **Phase 2.2**: 實作細分關鍵字特徵 (Efficacy, Price, Logistics)。
3.  **Phase 2.3**: 重新執行 `binary_explosive` 與 `hybrid_strict_balanced` 實驗，評估新特徵對 F1 的貢獻。
