# 特徵維度說明書 (Feature Definitions)

本文檔詳細說明 `train.py` 與 `data_loader.py` 中使用的特徵計算邏輯，特別是針對「物理運動學 (Kinematics)」相關的特徵。

## 1. 物理運動學特徵 (Kinematics Features)
這些特徵是基於物理學中的「運動」概念，將評論數量的變化視為物體的位移，從而推導出速度、加速度與衝擊度 (Jerk)。目的是捕捉「爆品」在爆發初期的動態特徵。

### 1.1 速度 (Velocity)
將時間切分為週 (7天)，計算每週的評論數量作為「速度」。
*   **`kin_v_1`** (Current Speed): 最近一週 [cutoff-7d, cutoff] 的評論數。
*   **`kin_v_2`** (Previous Speed): 上一週 [cutoff-14d, cutoff-7d] 的評論數。
*   **`kin_v_3`** (Baseline Speed): 上上週 [cutoff-21d, cutoff-14d] 的評論數。

### 1.2 加速度 (Acceleration)
描述討論熱度「增長」的快慢。
*   **`kin_acc_abs` (絕對加速度)**: 
    *   **公式**: `kin_v_1 - kin_v_2`
    *   **意義**: 直接反映本週比上週多了多少評論。正值代表加速，負值代表減速。
*   **`kin_acc_rel` (相對加速度)**:
    *   **公式**: `(log1p(v1) - log1p(v2)) / (log1p(v2) + 1)`
    *   **意義**: 對數尺度的增長率。解決了基數小的時候變異過大的問題（例如從 1 到 2 與從 100 到 200 的區別）。

### 1.3 衝擊度 (Jerk)
描述加速度的變化率（加速度的加速度）。
*   **`kin_jerk_abs` (物理衝擊度)**:
    *   **公式**: `kin_v_1 - 2*kin_v_2 + kin_v_3`
    *   **意義**: 偵測趨勢的「轉折」。
        *   若 `v1, v2, v3` 為 `10, 5, 0` (線性增長)，Jerk = 0。
        *   若 `v1, v2, v3` 為 `20, 5, 0` (指數爆發)，Jerk > 0。
        *   這能幫助模型抓住「突然爆紅」的瞬間，而非穩定成長的商品。

---

### 1.4 複合動量特徵 (derived Momentum)
結合物理特徵與其他維度（總量、語意品質），以過濾掉「虛假」的加速度。

#### 1.4.1 早鳥動量 (Early Bird Momentum)
*   **公式**: `kin_acc_abs * (1 / (log1p(comment_count_90d) + 1))`
*   **意義**: **獎勵「新」的加速度**。
    *   同樣增加 10 則評論，對於一個累積只有 5 則的新品 (分母小) 來說，此分數極高。
    *   對於已經有 1000 則評論的老品 (分母大)，此分數會被稀釋。
    *   用途：專門抓那種「橫空出世」的新爆品。

#### 1.4.2 品質驅動動量 (Quality Driven Momentum)
*   **公式**: `kin_acc_abs * category_fit_score`
*   **意義**: **需經過「語意合群度」驗證的加速度**。
    *   `category_fit_score` (類別適配度) 衡量該商品的評論內容是否與該品類的「典型討論」相符。
    *   若某商品因為「負評灌水」或「無關廣告」導致評論暴增 (`acc` 高)，但內容與品類核心無關 (`fit` 低)，此特徵會被打折。
    *   用途：過濾掉廣告操作或炎上導致的虛假熱度。

---

### 1.5 其他基礎特徵
*   **`comment_count_pre`**: 訓練截止日前累積總評論數。
*   **`days_since_last_comment`**: 距離最後一則評論的天數 (Recency)。
*   **`sentiment_mean_recent`**: 近期評論的平均情感分數。

---

## 2 演算法層特徵 (Algorithm Layer)

### 2.1 類別適配度 (Category Fit Score)
**研究主題**: 以「品類共識」為基準，衡量該商品的評論內容是否為「典型代表」。

#### 目標
判斷商品是否符合該品類的「主流討論風格」。假設是：**爆品通常是「原型」(Prototypical)，非爆品則是「異類」(Outlier)**。

#### 計算邏輯
1.  **品類分群**: 依 `keyword` (關鍵詞/品類) 將商品分組。
2.  **文字向量化**: 
    *   對該品類所有商品的 `aggregated_comments` (所有評論內容的合併) 進行 **TF-IDF 向量化** (2000維)。
    *   得到每個商品的「語意向量」 `X_text[i]`。
3.  **計算品類中心點 (Centroid)**:
    ```python
    centroid = mean(X_text)  # 所有商品向量的平均
    ```
4.  **計算相似度 (Cosine Similarity)**:
    *   對每個商品，計算其向量與中心點的 **餘弦距離 (Cosine Distance)**。
    *   轉換為相似度分數：
    ```python
    category_fit_score = 1 - cosine_distance(X_text[i], centroid)
    ```
    *   分數範圍 [0, 1]。越接近 1 代表越「標準/典型」。

#### 洞察 (Insight)
*   **高分 (0.8+)**: 商品討論內容與品類主流高度一致 (例如「口罩」品類中提到「防護、過濾、舒適」)。
*   **低分 (0.3-)**: 商品評論內容偏離品類共識 (例如「口罩」品類中卻都在討論「退貨、客服、品質差」)。

**應用價值**:
*   結合 `kin_acc_abs` (加速度) 使用，過濾掉「評論暴增但內容與品類無關」的異常商品 (可能是炎上或操作)。
*   例如：`quality_driven_momentum = kin_acc_abs * category_fit_score`。

---

## 3. 心理層特徵 (Psychology Layer / Info-Theoretic Features)

**研究主題**: 使用資訊理論與心理學概念，衡量評論的「真實性」與「有機性」，用於區分「真實的有機討論」與「刷評/機器人行為」。

### 3.1 語意熵 (Semantic Entropy)
**目標**: 測量評論的**主題多樣性**，使用 Shannon 熵來量化評論內容的語意聚類分布。

#### 計算邏輯
1.  **資料來源**: 使用該商品最近 **90 天內的最多 100 條評論**（從 `recent_comments_json` 欄位）。
2.  **文字向量化** (A/B 測試)：
    *   **`feat_entropy_tfidf`** (Baseline): 使用 TF-IDF 向量化 (100 維)
    *   **`feat_entropy_emb`** (Challenger): 使用 SBERT 語意嵌入 (`all-MiniLM-L6-v2`)
3.  **K-Means 聚類**:
    ```python
    n_clusters = min(len(texts), 5)  # 最多 5 個主題群
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    ```
4.  **計算 Shannon 熵**:
    ```python
    counts = np.bincount(labels)
    probs = counts / len(labels)
    entropy = scipy.stats.entropy(probs, base=2)
    ```

#### 語意解釋

| 熵值範圍 | 意義 | 實際案例 |
|---------|------|---------|
| **高熵 (1.5+)** | 評論主題**多元分散** | 真實用戶從多種角度討論（品質、外觀、價格、配送等） |
| **低熵 (< 0.5)** | 評論主題**高度一致** | 可疑的「範本化刷評」或「一致性回覆」 |

#### 應用
*   **Spam Risk Score**: `risk_mask = (comment_count_90d > 5) & (entropy < 0.5)`
*   **Challenger Momentum**: `momentum_emb = kin_acc_abs × category_fit_score × (entropy + 0.5)`

---

### 3.2 時間突發性 (Temporal Burstiness)
**目標**: 測量評論的**時間分布模式**，區分「自然的討論節奏」與「集中刷評」。

#### 計算邏輯
1.  **資料來源**: 使用該商品最近 **90 天內的評論**及其 `comment_date`（至少需要 3 條評論）。
2.  **計算評論間隔時間 (IAT)**:
    ```python
    dates_sorted = sorted(dates)
    iats = [(dates_sorted[i+1] - dates_sorted[i]).total_seconds() 
            for i in range(len(dates_sorted)-1)]
    ```
3.  **計算 Burstiness 指標**:
    ```python
    mean_iat = np.mean(iats)
    std_iat = np.std(iats)
    r = std_iat / mean_iat  # 變異係數
    burstiness = (r - 1) / (r + 1)  # 標準化到 [-1, 1]
    ```

#### 語意解釋

| 值範圍 | 意義 | 實際案例 |
|-------|------|---------|
| **接近 1** | **突發性高** (Bursty) | 評論集中在短時間內爆發，然後長時間沉寂（**典型爆品模式**） |
| **接近 0** | **穩定均勻** (Regular) | 評論均勻分布（穩定銷售的成熟商品） |
| **接近 -1** | **異常規律** (Periodic) | 評論以極其規律的間隔出現（**可疑的機器人刷評**） |

#### 洞察
*   **爆品檢測**: 高 burstiness (> 0.5) + 高加速度 → 真實的爆發式討論
*   **刷評檢測**: 低 burstiness (< -0.5) + 高評論量 → 可疑的規律性刷評

---

### 3.3 詞彙多樣性 (Lexical Diversity)
**目標**: 測量評論的**用詞豐富程度**，使用 **Guiraud's R 指標**量化詞彙重複率。

#### 計算邏輯
1.  **資料來源**: 合併該商品最近 **90 天內的所有評論文字**。
2.  **分詞統計**:
    ```python
    tokens = all_text.split()
    N = len(tokens)  # 總詞數
    V = len(set(tokens))  # 不同詞彙數
    V1 = sum(1 for count in Counter(tokens).values() if count == 1)  # 單次詞數
    ```
3.  **Guiraud's R 指標**:
    ```python
    denominator = 1 - (V1 / V)
    R = (100 * log(N)) / denominator
    ```

#### 語意解釋

| R 值 | 意義 | 實際案例 |
|-----|------|---------|
| **高 R (> 100)** | **詞彙多樣** | 真實用戶用不同方式描述（「超讚」「好用」「推薦」「划算」） |
| **低 R (< 50)** | **詞彙貧乏** | 重複使用相同詞彙（「好好好好好」「必買必買必買」） |

#### 應用
*   **真實性驗證**: 高詞彙多樣性通常代表真實的多元討論
*   **範本化檢測**: 極低的詞彙多樣性可能代表「複製貼上」或「範本回覆」

---

### 3.4 特徵組合應用

#### Challenger Momentum (挑戰者動量)
```python
organic_factor_emb = df['feat_entropy_emb'].fillna(0) + 0.5
df['momentum_emb'] = df['kin_acc_abs'] * quality_factor * organic_factor_emb
```
**邏輯**: 加速度 × 品類適配度 × 語意多樣性 → 過濾掉「虛假的熱度增長」

#### Spam Risk Score (垃圾風險分數)
```python
risk_entropy = df['feat_entropy_emb']
risk_mask = (df['comment_count_90d'] > 5) & (risk_entropy < 0.5)
df['spam_risk_score'] = df.loc[risk_mask, 'kin_acc_abs']
```
**邏輯**: 高評論量 + 低語意熵 + 高加速度 → 可疑的刷榜行為

---

### 3.5 研究價值與洞察

#### 為什麼需要心理層特徵？

| 特徵類型 | 優勢 | 盲點 |
|---------|------|------|
| **物理特徵** (`kin_acc_abs`) | 快速捕捉「量」的變化 | 無法區分真假，刷評也會產生高加速度 |
| **心理特徵** (Entropy/Burstiness/Diversity) | 捕捉「質」的特徵，能識別真實性 | 計算成本較高 |

#### 特徵組合的威力

| 情境 | 物理特徵 | 心理特徵 | 判斷結果 |
|-----|---------|---------|---------|
| **真實爆品** | 高加速度 | 高熵 + 高突發 + 高多樣性 | ✅ 正類 |
| **刷榜商品** | 高加速度 | 低熵 + 低突發 + 低多樣性 | ❌ 負類 (spam_risk_score 標記) |
| **穩定老品** | 低加速度 | 中等熵 + 低突發 + 中多樣性 | ❌ 負類 |

**核心洞察**: 心理層特徵形成了一個「**真實性驗證系統**」，與物理特徵結合後能有效過濾掉「虛假的熱度增長」，只保留「真實的爆品訊號」。

