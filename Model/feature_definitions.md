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

---

## 4. 融合層特徵 (Fusion Layer)

**研究主題**: 整合物理層、演算法層與心理層的特徵，形成「**多維驗證系統**」，最大化爆品檢測的準確性並最小化誤報（刷評/炎上）。

### 4.1 真實動量 (Authentic Momentum)
**目標**: 結合三層特徵，生成「**經過多重驗證的加速度訊號**」。

#### 計算邏輯

融合層有兩種實作方式（A/B 測試）：

##### 4.1.1 Baseline Momentum (基於 TF-IDF)
```python
quality_factor = df['category_fit_score'].fillna(0) + 0.5
organic_factor_tfidf = df['feat_entropy_tfidf'].fillna(0) + 0.5
df['momentum_tfidf'] = df['kin_acc_abs'] * quality_factor * organic_factor_tfidf
```

##### 4.1.2 Challenger Momentum (基於 SBERT) - 即 `authentic_momentum`
```python
quality_factor = df['category_fit_score'].fillna(0) + 0.5
organic_factor_emb = df['feat_entropy_emb'].fillna(0) + 0.5
df['momentum_emb'] = df['kin_acc_abs'] * quality_factor * organic_factor_emb
```

**別名**: `authentic_momentum` = `momentum_emb`（在部分實驗中使用此名稱）

#### 公式拆解

```
Authentic Momentum = 物理層 × 演算法層 × 心理層
                   = kin_acc_abs × (category_fit_score + 0.5) × (feat_entropy_emb + 0.5)
```

| 組成部分 | 來源層 | 意義 | 檢測目的 |
|---------|-------|------|---------|
| `kin_acc_abs` | 物理層 | 評論增長速度 | 捕捉「量」的變化 |
| `category_fit_score + 0.5` | 演算法層 | 品類適配度 | 驗證「語意合群性」 |
| `feat_entropy_emb + 0.5` | 心理層 | 語意多樣性 | 驗證「有機真實性」 |

**為什麼 +0.5？**
- 防止任一因子為 0 時導致整體分數歸零
- 即使某個維度分數較低，仍保留其他維度的訊號
- 類似於「平滑因子」或「先驗信心」

#### 語意解釋

| Authentic Momentum 值 | 意義 | 典型情境 |
|---------------------|------|---------|
| **高分 (> 5)** | 三層驗證皆通過 | ✅ **真實爆品**：高加速度 + 符合品類 + 主題多元 |
| **中分 (1-5)** | 部分驗證通過 | ⚠️ **潛在爆品** 或 **穩定成長商品** |
| **低分 (< 1)** | 至少一層驗證失敗 | ❌ **刷評商品** 或 **炎上事件** 或 **無討論商品** |

#### 失敗案例分析

| 情境 | 物理層 | 演算法層 | 心理層 | Authentic Momentum | 判斷 |
|-----|-------|---------|-------|-------------------|------|
| **範本刷評** | 高加速度 (10) | 高適配度 (0.8) | **低熵 (0.1)** | 10 × 1.3 × 0.6 = **7.8** → 被心理層打折 | ❌ 刷評 |
| **負評炎上** | 高加速度 (10) | **低適配度 (0.2)** | 高熵 (1.5) | 10 × 0.7 × 2.0 = **14** → 被演算法層打折 | ❌ 炎上 |
| **無關廣告** | 高加速度 (10) | **低適配度 (0.1)** | **低熵 (0.2)** | 10 × 0.6 × 0.7 = **4.2** → 被雙重打折 | ❌ 廣告 |
| **真實爆品** | 高加速度 (10) | 高適配度 (0.9) | 高熵 (1.8) | 10 × 1.4 × 2.3 = **32.2** ✅ | ✅ 爆品 |

---

### 4.2 垃圾風險分數 (Spam Risk Score)
**目標**: 識別「**高評論量 + 低語意熵 + 高加速度**」的可疑刷評行為。

#### 計算邏輯

```python
# 選擇熵指標（優先使用 SBERT，否則使用 TF-IDF）
risk_entropy = df['feat_entropy_emb'] if sbert_model else df['feat_entropy_tfidf']

# 定義風險遮罩
risk_mask = (df['comment_count_90d'] > 5) & (risk_entropy < 0.5)

# 計算風險分數
df['spam_risk_score'] = 0.0
df.loc[risk_mask, 'spam_risk_score'] = df.loc[risk_mask, 'kin_acc_abs']
```

#### 觸發條件

| 條件 | 閾值 | 意義 |
|-----|------|------|
| `comment_count_90d > 5` | 至少 6 條評論 | 排除「樣本太少」的情況 |
| `entropy < 0.5` | 語意熵極低 | 評論主題高度一致（可疑） |
| **兩者同時滿足** | - | 觸發風險標記 |

#### 風險分數解讀

```python
if spam_risk_score > 0:
    # 風險分數 = 該商品的加速度
    # 分數越高，表示「可疑的加速度」越大
    風險等級 = "高" if spam_risk_score > 10 else "中"
else:
    風險等級 = "低（無風險）"
```

#### 語意解釋

| Spam Risk Score | 意義 | 建議動作 |
|----------------|------|---------|
| **0** | 未觸發風險條件 | ✅ 安全，繼續正常評估 |
| **1-5** | 低風險 | ⚠️ 關注，可能是小規模刷評 |
| **5-10** | 中風險 | 🔍 人工審查，檢查評論內容 |
| **> 10** | 高風險 | 🚨 高度可疑，建議排除或降權 |

#### 實際案例

##### 案例 1: 真實商品（無風險）
```
comment_count_90d = 50
feat_entropy_emb = 1.8  (主題多元)
kin_acc_abs = 10

✅ entropy (1.8) > 0.5 → 不觸發 risk_mask
✅ spam_risk_score = 0
```

##### 案例 2: 可疑刷評（高風險）
```
comment_count_90d = 30
feat_entropy_emb = 0.2  (主題極度一致，可能是範本)
kin_acc_abs = 15

❌ comment_count > 5 AND entropy < 0.5 → 觸發 risk_mask
❌ spam_risk_score = 15 (高風險！)
```

##### 案例 3: 樣本不足（無法判斷）
```
comment_count_90d = 3   (樣本太少)
feat_entropy_emb = 0.1
kin_acc_abs = 8

✅ comment_count ≤ 5 → 不觸發 risk_mask
✅ spam_risk_score = 0 (保守估計，不誤判)
```

---

### 4.3 融合層的設計哲學

#### 多層防禦機制

```
第一層防禦 (物理層): 捕捉「量」的異常
    ↓
第二層防禦 (演算法層): 驗證「語意合群性」
    ↓
第三層防禦 (心理層): 驗證「真實性」
    ↓
融合層輸出:
  • Authentic Momentum: 三層皆通過 → 高分
  • Spam Risk Score: 任一層失敗 → 標記
```

#### 優勢

| 優勢 | 說明 |
|-----|------|
| **高精確度** | 三層驗證減少誤報（不會把真實爆品誤判為刷評） |
| **高召回率** | Spam Risk Score 主動標記異常（不會漏掉刷評） |
| **可解釋性** | 每一層失敗都有明確的理由（演算法層失敗 = 不符品類；心理層失敗 = 主題單一） |
| **魯棒性** | 即使某一層特徵缺失（如 SBERT 未載入），仍可退回到 TF-IDF |

#### 與單層特徵的對比

| 方法 | 準確率 | 召回率 | 問題 |
|-----|-------|-------|------|
| **僅物理層** (`kin_acc_abs`) | 低 | 高 | 刷評也會產生高加速度，誤報率極高 |
| **僅演算法層** (`category_fit_score`) | 中 | 低 | 無法捕捉時間動態，漏掉新爆品 |
| **僅心理層** (`feat_entropy_emb`) | 中 | 中 | 無法處理「穩定但主題多元」的老品 |
| **融合層** (`authentic_momentum` + `spam_risk_score`) | **高** | **高** | ✅ 多層驗證，最佳平衡 |

---

### 4.4 關鍵洞察

> **核心理念**: 真實的爆品應該同時滿足「量的異常」（物理層）、「質的符合」（演算法層）和「有機的多元」（心理層）。任一層失敗，都可能是虛假訊號。

**Authentic Momentum** 和 **Spam Risk Score** 是「一體兩面」的設計：
- **Authentic Momentum**: 正向指標，尋找「三層皆優秀」的商品
- **Spam Risk Score**: 負向指標，標記「心理層失敗」的可疑商品

兩者結合，形成完整的「真實性驗證系統」。

---

## 5. 交互作用層 (Interaction Features) - 輔助過濾器

**研究主題**: 透過「**跨特徵組合**」來捕捉單一特徵無法表達的複雜模式，並創建「**輔助過濾器**」來減少誤報和提升訊號品質。

### 設計理念

> **核心概念**: 真實世界的爆品訊號往往是「多個條件同時滿足」的結果，而非單一特徵的極值。交互特徵通過**乘法組合**或**條件判斷**來模擬這種「AND」邏輯。

---

### 5.1 品質驅動動量 (Quality Driven Momentum)
**已在 1.4.2 節說明**，屬於物理層與演算法層的交互。

```python
df["quality_driven_momentum"] = df["kin_acc_abs"] * df["category_fit_score"]
```

**意義**: 加速度需經過「語意合群性」驗證，過濾掉因負評灌水或廣告導致的虛假熱度。

---

### 5.2 早鳥動量 (Early Bird Momentum)
**已在 1.4.1 節說明**，屬於物理層與總量特徵的交互。

```python
df["early_bird_momentum"] = df["kin_acc_abs"] * (1 / (np.log1p(df["comment_count_90d"]) + 1))
```

**意義**: 獎勵「新」的加速度，專門抓「橫空出世」的新爆品。

---

### 5.3 驗證速度 (Validated Velocity)
**目標**: 過濾掉「樣本太少導致的不穩定加速度」。

#### 計算邏輯
```python
df["validated_velocity"] = df["ratio_recent30_to_prev60"] * np.log1p(df["comment_3rd_30d"])
```

**公式拆解**:
- `ratio_recent30_to_prev60`: 近 30 天評論數 / 前 60 天評論數（加速度比例）
- `log1p(comment_3rd_30d)`: 近 30 天評論數的對數（音量驗證）

#### 為什麼需要這個特徵？

| 情境 | `ratio_recent30_to_prev60` | `comment_3rd_30d` | `validated_velocity` | 判斷 |
|-----|---------------------------|------------------|---------------------|------|
| **真實爆品** | 5.0 (500%) | 100 | 5.0 × log(101) ≈ **23** | ✅ 高音量 + 高加速 |
| **噪音訊號** | 5.0 (500%) | 2 | 5.0 × log(3) ≈ **5.5** | ❌ 樣本太少，不可信 |
| **穩定老品** | 1.0 (100%) | 100 | 1.0 × log(101) ≈ **4.6** | ⚠️ 高音量但無加速 |

**洞察**: 
- 如果只看比例（`ratio`），「從 1 到 5」和「從 100 到 500」看起來一樣（都是 5 倍）
- 但前者可能只是隨機波動，後者才是真實趨勢
- **對數音量驗證**確保加速度建立在「足夠的樣本基礎」上

---

### 5.4 新奇動量 (Novelty Momentum)
**目標**: 捕捉「由**新用戶**驅動的加速度」，而非老客戶的回購。

#### 計算邏輯
```python
df["novelty_momentum"] = df["ratio_recent30_to_prev60"] * (1 - df["repurchase_ratio_recent"])
```

**公式拆解**:
- `ratio_recent30_to_prev60`: 加速度比例
- `1 - repurchase_ratio_recent`: **非回購比例**（新客佔比）

#### 為什麼需要這個特徵？

爆品有兩種類型：
1. **新客爆品**: 大量新用戶湧入（高 novelty_momentum）
2. **回購爆品**: 老客戶持續回購（低 novelty_momentum）

| 情境 | 加速度 | 回購比例 | Novelty Momentum | 類型 |
|-----|-------|---------|-----------------|------|
| **網紅推薦** | 5.0 | 0.1 (90% 新客) | 5.0 × 0.9 = **4.5** | ✅ 新客爆品 |
| **老客回購** | 5.0 | 0.8 (20% 新客) | 5.0 × 0.2 = **1.0** | ⚠️ 回購驅動 |
| **無加速** | 1.0 | 0.1 | 1.0 × 0.9 = **0.9** | ❌ 無趨勢 |

**洞察**:
- 高 `novelty_momentum` = 「破圈效應」（新市場、新族群）
- 低 `novelty_momentum` = 「忠誠度效應」（穩定但無擴散）
- 模型可以根據業務目標選擇重視哪一種

---

### 5.5 價格加權情緒 (Price Weighted Arousal)
**目標**: 捕捉「高價商品的情緒訊號更強」的模式。

#### 計算邏輯
```python
df["price_weighted_arousal"] = df["bert_arousal_mean"] * np.log1p(df["price"])
```

**公式拆解**:
- `bert_arousal_mean`: BERT 偵測的「興奮/激動」情緒機率
- `log1p(price)`: 價格的對數（避免極值影響）

#### 研究發現（2025-12-08 更新）

> **重要發現**: 原本使用 `clean_arousal_score`（扣除負面情緒），但後來發現「**負面情緒本身也是強訊號**」（炎上也是一種爆紅），因此改用 **RAW arousal**。

| 商品類型 | 價格 | Arousal | `price_weighted_arousal` | 爆品機率 |
|---------|-----|---------|-------------------------|---------|
| **高價 3C** | $10,000 | 0.8 | 0.8 × log(10001) ≈ **7.4** | 高 ✅ |
| **平價日用** | $100 | 0.8 | 0.8 × log(101) ≈ **3.7** | 中 ⚠️ |
| **高價低情緒** | $10,000 | 0.2 | 0.2 × log(10001) ≈ **1.8** | 低 ❌ |

**洞察**:
- 高價商品的討論「情緒密度」通常更高（投資大 → 情緒重）
- 同樣的 arousal 分數，在高價商品上更有意義

**延伸特徵 (2025-12-08 新增)**:
```python
df["price_weighted_novelty"] = df["bert_novelty_mean"] * np.log1p(df["price"])
```
**研究發現**: Novelty 與高價的關聯性**甚至比 Arousal 更強**。

---

### 5.6 成熟商品標記 (Is Mature Product)
**目標**: 幫助樹模型區分「新爆品」與「老爆品」。

#### 計算邏輯
```python
df["is_mature_product"] = ((df["comment_count_pre"] > 50) | (df["repurchase_ratio_recent"] > 0.2)).astype(int)
```

**觸發條件（OR 邏輯）**:
- 累積評論數 > 50 條
- **OR** 回購比例 > 20%

#### 為什麼需要這個特徵？

| 商品狀態 | 累積評論 | 回購比例 | `is_mature_product` | 策略 |
|---------|---------|---------|-------------------|------|
| **新爆品** | 10 | 0.05 | 0 | 重點關注，高成長潛力 |
| **成熟爆品** | 200 | 0.35 | 1 | 已確立地位，穩定收益 |
| **快速轉回購** | 30 | 0.25 | 1 | 雖然新但已有忠誠度 |

**洞察**:
- **樹模型的優勢**: 可以用此旗標做「第一層分裂」
  - `is_mature_product == 0` → 新品分支 → 關注 `early_bird_momentum`
  - `is_mature_product == 1` → 老品分支 → 關注 `repurchase_ratio`
- **避免混淆**: 不同生命週期的商品，爆品定義不同

---

### 5.7 交互特徵的設計模式總結

#### 模式 1: **音量驗證** (Volume Validation)
**代表**: `validated_velocity`
```
pattern = 相對變化 × log1p(絕對音量)
目的 = 過濾「樣本太少的不穩定訊號」
```

#### 模式 2: **條件過濾** (Conditional Filtering)
**代表**: `quality_driven_momentum`, `early_bird_momentum`
```
pattern = 核心訊號 × 過濾因子
目的 = 只保留「通過驗證」的訊號
```

#### 模式 3: **語意分層** (Semantic Segmentation)
**代表**: `novelty_momentum`, `price_weighted_arousal`
```
pattern = 情緒特徵 × 結構特徵
目的 = 捕捉「特定語意邏輯」的模式
```

#### 模式 4: **顯式分群** (Explicit Grouping)
**代表**: `is_mature_product`
```
pattern = boolean flag (0/1)
目的 = 幫助樹模型「先分群，再建模」
```

---

### 5.8 與其他層的關係

```
物理層 (Kinematics)
    ↓ 提供「量」的訊號
演算法層 (Algorithm)
    ↓ 提供「質」的驗證
心理層 (Psychology)
    ↓ 提供「真實性」的驗證
融合層 (Fusion)
    ↓ 整合三層形成「多維驗證」
交互作用層 (Interaction) ← 【本層】
    ↓ 創建「輔助過濾器」
    ↓ 處理「AND 邏輯」和「分群邏輯」
    ↓
最終特徵集 → 模型訓練
```

**定位**: 交互作用層是「**工具箱**」，針對特定問題設計專用過濾器。

---

### 5.9 關鍵洞察

> **設計哲學**: 不要期待模型自己學會「AND 邏輯」，主動透過乘法組合告訴模型「這兩個條件要同時滿足」。

#### 為什麼需要手動交互？

**反例**: 如果只給模型 `kin_acc_abs` 和 `comment_count_90d` 兩個特徵：
- 線性模型：無法學會「樣本少時要打折」
- 樹模型：可以學，但需要很多樣本，且容易過擬合

**正例**: 直接給 `early_bird_momentum = kin_acc_abs / (log1p(comment_count_90d) + 1)`：
- 線性模型：一個係數就能捕捉此模式
- 樹模型：更容易找到最佳分裂點

**結論**: 交互特徵 = **特徵工程的「語法糖」**，讓模型更容易學習你的領域知識。

---

## 6. BERT 語意層 (Deep Semantic) - 情緒與意圖

**研究主題**: 使用 **BERT Zero-Shot 分類**來偵測評論中的「**情緒狀態**」與「**用戶意圖**」，取代傳統的關鍵詞匹配或情感詞典方法。

### 設計理念

> **核心概念**: 傳統的 Regex 或詞典方法只能捕捉「表面詞彙」，無法理解「語境」和「隱含意圖」。BERT Zero-Shot 可以理解「我超愛這個」和「愛死了會回購」都表達「arousal + repurchase」，即使用詞完全不同。

---

### 6.1 BERT Zero-Shot 分類架構

#### 資料來源
- **表格**: `comment_semantic_scores`
- **欄位**: `score_arousal`, `score_novelty`, `score_repurchase`, `score_negative`, `score_advertisement`
- **時間窗口**: 最近 **90 天**的評論
- **聚合方式**: 對每個商品計算**平均機率分數**

#### 標籤定義 (Label Definitions)
**來源檔案**: [`compute_bert_features.py`](file:///c:/yves/momo/momo_crawler/Model/compute_bert_features.py#L106-L112)

BERT Zero-Shot 分類使用以下**中文標籤字串**來定義每個語意維度：

```python
label_map = {
    "High_Arousal": "驚豔、激動、太神了",
    "High_Novelty": "新奇、初次體驗、相見恨晚",
    "High_Repurchase_Intent": "回購意願高、忠實粉絲",
    "Negative_Complaint": "憤怒、失望、反推",
    "Advertisement": "業配、廣告、湊字數"
}
```

**運作原理**:
1. **模型**: 使用 `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`（多語言 Zero-Shot 模型）
2. **輸入**: 評論文字（如「這個超好用會再買」）
3. **候選標籤**: 上述 5 個中文標籤字串
4. **輸出**: 每個標籤的機率分數（0-1）
5. **儲存**: 機率分數寫入 `comment_semantic_scores` 表

**範例推理過程**:
```python
評論: "這個超好用會再買"

BERT 分類器判斷：
  - "驚豔、激動、太神了" (High_Arousal) → 0.75 (高興奮)
  - "新奇、初次體驗、相見恨晚" (High_Novelty) → 0.30 (中等新奇)
  - "回購意願高、忠實粉絲" (High_Repurchase_Intent) → 0.90 (強回購意圖！)
  - "憤怒、失望、反推" (Negative_Complaint) → 0.05 (低負面)
  - "業配、廣告、湊字數" (Advertisement) → 0.10 (低廣告嫌疑)
```

**為什麼使用中文標籤？**
- mDeBERTa-v3-base-mnli-xnli 是多語言模型，支援中英文
- 中文標籤能更精確匹配中文評論的語意
- 標籤字串的選擇影響分類準確度（「驚豔」vs「高興」效果不同）


#### SQL 查詢邏輯
```sql
-- Semantic Mean Scores (Recent 90 Days)
AVG(score_arousal) FILTER (WHERE comment_date >= cutoff - INTERVAL '90 days') AS bert_arousal_mean,
AVG(score_novelty) FILTER (WHERE comment_date >= cutoff - INTERVAL '90 days') AS bert_novelty_mean,
AVG(score_repurchase) FILTER (WHERE comment_date >= cutoff - INTERVAL '90 days') AS bert_repurchase_mean,
AVG(score_negative) FILTER (WHERE comment_date >= cutoff - INTERVAL '90 days') AS bert_negative_mean,
AVG(score_advertisement) FILTER (WHERE comment_date >= cutoff - INTERVAL '90 days') AS bert_advertisement_mean
```

**設計選擇**: 使用**平均機率**而非二元標籤（>0.8），因為：
- 機率值包含更豐富的信息（0.9 vs 0.5 都是正類，但信心不同）
- 避免硬閾值導致的邊界問題
- 可以捕捉「整體氛圍」而非「極端案例」

---

### 6.2 五個核心語意維度

#### 6.2.1 興奮度 (Arousal) - `bert_arousal_mean`
**偵測目標**: 評論者的**情緒激動程度**（正向或負向皆可）。

**典型表達**:
- ✅ 正向興奮: "太好用了！"、"超讚"、"驚艷"、"愛死了"
- ✅ 負向興奮: "太爛了！"、"超失望"、"氣死"、"不能接受"
- ❌ 低興奮: "還可以"、"普通"、"沒什麼特別"

**意義**:
- 高 arousal (> 0.6) = 商品引發**強烈情緒反應**（好或壞）
- 低 arousal (< 0.3) = 商品**平淡無奇**，缺乏討論價值

**爆品關聯**:
- 爆品通常引發強烈情緒（高 arousal）
- 但需結合 `bert_negative_mean` 判斷是正向爆紅還是負面炎上

---

#### 6.2.2 新奇度 (Novelty) - `bert_novelty_mean`
**偵測目標**: 商品是否具有**創新性、獨特性、話題性**。

**典型表達**:
- ✅ 高新奇: "第一次看到"、"好特別"、"超酷的設計"、"創新"、"驚喜"
- ❌ 低新奇: "跟別家一樣"、"常見款"、"沒什麼特別"

**意義**:
- 高 novelty (> 0.6) = 商品有**差異化**，容易引發討論和分享
- 低 novelty (< 0.3) = 同質化商品，難以脫穎而出

**爆品關聯**:
- **與高價商品高度相關**（研究發現：novelty × price 是最強訊號之一）
- 新奇度是「破圈傳播」的關鍵（引發好奇 → 分享 → 擴散）

---

#### 6.2.3 回購意圖 (Repurchase) - `bert_repurchase_mean`
**偵測目標**: 用戶是否表達**再次購買或推薦他人**的意願。

**典型表達**:
- ✅ 高回購: "會再買"、"推薦給朋友"、"回購第三次"、"囤貨"、"必買清單"
- ❌ 低回購: "買一次就夠了"、"不會再買"、"踩雷"

**意義**:
- 高 repurchase (> 0.6) = 用戶忠誠度高，**長期價值**
- 低 repurchase (< 0.3) = 一次性購買，或使用體驗不佳

**爆品關聯的兩面性**:
- **新客爆品**: 低 repurchase（大量新用戶湧入，還沒回購機會）
- **回購爆品**: 高 repurchase（老客戶持續回購）
- 需結合 `novelty_momentum` 判斷是哪一種

---

#### 6.2.4 負面情緒 (Negative) - `bert_negative_mean`
**偵測目標**: 評論中的**不滿、憤怒、失望**等負面情緒。

**典型表達**:
- ✅ 高負面: "很失望"、"品質差"、"客服態度惡劣"、"退貨"、"不推薦"
- ❌ 低負面: "滿意"、"符合預期"、"很好"

**意義**:
- 高 negative (> 0.6) = 商品存在問題，可能是**炎上事件**
- 低 negative (< 0.2) = 用戶滿意度高

**重要發現 (2025-12-08)**:
> **負面情緒本身也是強訊號**！炎上也是一種「爆紅」（高討論度）。
> 因此在 `price_weighted_arousal` 中改用 **RAW arousal**，不再扣除 negative。

---

#### 6.2.5 廣告嫌疑 (Advertisement) - `bert_advertisement_mean`
**偵測目標**: 評論是否像**商業廣告或業配文**。

**典型表達**:
- ✅ 高廣告: "官方推薦"、"限時優惠"、"點擊連結"、"私訊我"、"團購開跑"
- ❌ 低廣告: "我自己買的"、"真心推薦"、"使用心得"

**意義**:
- 高 advertisement (> 0.6) = 可能是**假評論或業配**
- 低 advertisement (< 0.2) = 真實用戶分享

**應用**:
- 用於計算 `clean_arousal_score`（扣除廣告成分）
- 但後來發現過度懲罰，因此調整策略

---

### 6.3 衍生特徵

#### 6.3.1 乾淨興奮度 (Clean Arousal Score)
**目標**: 過濾掉「負面炎上」和「假評論廣告」的興奮度。

```python
df["clean_arousal_score"] = df["bert_arousal_mean"] * (1 - df["bert_negative_mean"]) * (1 - df["bert_advertisement_mean"])
```

**邏輯**:
- `bert_arousal_mean`: 原始興奮度
- `× (1 - bert_negative_mean)`: 扣除負面成分
- `× (1 - bert_advertisement_mean)`: 扣除廣告成分

**案例對比**:

| 情境 | Arousal | Negative | Advertisement | Clean Arousal | 解讀 |
|-----|---------|----------|---------------|---------------|------|
| **真實好評** | 0.8 | 0.1 | 0.1 | 0.8 × 0.9 × 0.9 = **0.65** | ✅ 真實興奮 |
| **負評炎上** | 0.8 | 0.7 | 0.1 | 0.8 × 0.3 × 0.9 = **0.22** | ❌ 被負面打折 |
| **業配廣告** | 0.8 | 0.1 | 0.8 | 0.8 × 0.9 × 0.2 = **0.14** | ❌ 被廣告打折 |
| **平淡評論** | 0.3 | 0.1 | 0.1 | 0.3 × 0.9 × 0.9 = **0.24** | ⚠️ 原本就低 |

**限制**:
- 後來發現「負面炎上」本身也是強訊號（高討論度）
- 因此在部分特徵中改用 RAW arousal

---

#### 6.3.2 強度分數 (Intensity Score)
**目標**: 衡量商品的「**話題爆發力 vs 穩定回購力**」比例。

```python
df["intensity_score"] = (df["clean_arousal_score"] + df["bert_novelty_mean"]) / (df["bert_repurchase_mean"] + 0.1)
```

**公式拆解**:
- **分子**: `clean_arousal + novelty` = 「話題性」（興奮 + 新奇）
- **分母**: `repurchase + 0.1` = 「穩定性」（回購意圖）
- **+0.1**: 避免除以 0

**語意解釋**:

| Intensity Score | 意義 | 典型商品 |
|----------------|------|---------|
| **高 (> 5)** | **話題性 >> 回購性** | 網紅推薦、限量聯名、話題商品（短期爆發） |
| **中 (2-5)** | **話題與回購平衡** | 優質新品（既有話題又有品質） |
| **低 (< 2)** | **回購性 >> 話題性** | 日用品、回購商品（穩定但平淡） |

**案例對比**:

| 商品類型 | Clean Arousal | Novelty | Repurchase | Intensity Score | 類型 |
|---------|--------------|---------|------------|----------------|------|
| **網紅聯名** | 0.7 | 0.8 | 0.1 | (0.7+0.8) / 0.2 = **7.5** | ✅ 話題爆品 |
| **優質新品** | 0.6 | 0.5 | 0.4 | (0.6+0.5) / 0.5 = **2.2** | ✅ 平衡型 |
| **日用品** | 0.3 | 0.2 | 0.6 | (0.3+0.2) / 0.7 = **0.7** | ⚠️ 穩定但無爆發 |

**業務應用**:
- 高 intensity = 適合「短期促銷、流量獲取」
- 低 intensity = 適合「長期經營、會員回購」

---

#### 6.3.3 比例別名 (Ratio Aliases)
為了與舊程式碼兼容，創建了別名：

```python
df["arousal_ratio"] = df["bert_arousal_mean"]
df["novelty_ratio"] = df["bert_novelty_mean"]
df["repurchase_ratio_recent"] = df["bert_repurchase_mean"]
```

**說明**: 這些「ratio」並非真正的比例，而是 BERT 的**機率分數**（0-1 之間）。命名為 `ratio` 是歷史遺留，實際上應理解為「該維度的平均強度」。

---

### 6.4 BERT vs 傳統方法的對比

#### 傳統 Regex 方法
```python
# 舊方法：關鍵詞匹配
arousal_count = COUNT(WHERE comment_text ~ '超讚|太好|愛死|驚艷')
```

**問題**:
- ❌ 無法理解語境（"不是很讚" 也會匹配）
- ❌ 無法捕捉同義詞（"amazing" 無法匹配中文詞典）
- ❌ 無法量化程度（"好" vs "超級好" 都算一次）

#### BERT Zero-Shot 方法
```python
# 新方法：語意理解
bert_arousal_mean = AVG(BERT分類器(comment_text, label="興奮激動"))
```

**優勢**:
- ✅ 理解語境（"不是很讚" → 低分；"超級讚" → 高分）
- ✅ 泛化能力（"amazing"、"fantastic" 也能識別）
- ✅ 機率分數（0.9 vs 0.5，量化信心程度）
- ✅ 隱含意圖（"會再買" → 高 repurchase，即使沒說「回購」）

---

### 6.5 與其他層的關係

```
BERT 語意層 (Deep Semantic)
    ↓ 提供「情緒」和「意圖」的深度理解
    ↓
物理層 (Kinematics)
    ↓ 提供「量」的變化
    ↓
交互作用層 (Interaction)
    ↓ 組合形成 price_weighted_arousal, novelty_momentum
    ↓
心理層 (Psychology)
    ↓ 驗證「真實性」
    ↓
融合層 (Fusion)
    ↓ 整合形成最終訊號
```

**定位**: BERT 語意層是「**深度特徵提取器**」，將「非結構化文本」轉換為「結構化情緒/意圖分數」。

---

### 6.6 關鍵洞察

#### 洞察 1: 情緒 ≠ 情感極性
**傳統觀念**: 情感分析 = 正面/負面二分類

**BERT 語意層**: 
- `arousal` = 情緒激動程度（正負皆可）
- `negative` = 負面情緒強度
- 兩者獨立，可以同時高（例如「氣炸了！」= 高 arousal + 高 negative）

#### 洞察 2: 負面也是訊號
**早期假設**: 負面評論應該過濾掉

**研究發現 (2025-12-08)**: 
- 負面炎上也會帶來高討論度和銷售
- 因此在某些特徵中保留負面訊號（如 `price_weighted_arousal` 使用 RAW arousal）

#### 洞察 3: 多維度優於單一情感分數
**傳統**: sentiment_score (1-5) 

**BERT 語意層**: 5 個獨立維度
- 更豐富的信息
- 可以捕捉複雜情緒組合（例如「有瑕疵但會回購」= 中 negative + 高 repurchase）

#### 洞察 4: 平均值 vs 極值
**設計選擇**: 使用 **AVG** 而非 MAX/COUNT

**理由**:
- AVG 捕捉「整體氛圍」
- MAX 容易被極端案例影響
- COUNT 無法量化程度

**結論**: `bert_arousal_mean = 0.7` 表示「整體而言，這個商品的評論帶有中高程度的興奮情緒」，比「有 10 條超興奮評論」更有價值。

---

## 7. 基礎與統計層 (Basic & Statistical) - 控制變數

**研究主題**: 提供商品的**基本屬性**和**統計特徵**，作為其他高階特徵的「**控制變數**」或「**基準線**」。

### 設計理念

> **核心概念**: 在評估「爆品訊號」之前，需要先了解商品的「基本面」。例如，一個只有 5 條評論的商品即使加速度很高，也可能只是噪音；而一個有 1000 條評論的商品，加速度相同時意義完全不同。

---

### 7.1 靜態屬性 (Static Attributes)

這些特徵在商品生命週期中**基本不變**或**變化緩慢**。

#### 7.1.1 價格 (Price)
```sql
p.price::float AS price
```

**意義**: 商品售價（元）

**爆品關聯**:
- 高價商品的「情緒密度」通常更高（已在 `price_weighted_arousal` 中應用）
- 價格可能影響討論熱度（奢侈品 vs 日用品）

**統計特性**:
- **範圍**: 通常 10 - 50,000 元
- **分布**: 右偏（大多數商品便宜，少數商品昂貴）
- **處理**: 使用 `log1p(price)` 來平滑分布

---

#### 7.1.2 媒體豐富度特徵
```sql
has_image_urls     -- 是否有圖片（0/1）
has_video_url      -- 是否有影片（0/1）
has_reply_content  -- 是否有賣家回覆（0/1）
```

**意義**: 評論的「多媒體豐富程度」

**假設**:
- 有圖片/影片的評論 → 用戶投入度高 → 商品品質可能較好
- 有賣家回覆 → 客服積極 → 用戶體驗較好

**限制**:
- 這些是**二元特徵**（有/無），無法量化程度
- 可能受平台政策影響（例如某些品類鼓勵上傳圖片）

---

### 7.2 累積數據 (Cumulative Data)

這些特徵代表商品「**從上架至今**」的累積表現。

#### 7.2.1 累積評論數 (Comment Count Pre)
```sql
COUNT(*) AS comment_count_pre
```

**意義**: 訓練截止日前的**總評論數**

**重要性**: ⭐⭐⭐⭐⭐（最重要的控制變數之一）

**用途**:
1. **音量驗證**: 區分「真實趨勢」vs「隨機波動」
2. **成熟度判斷**: 結合 `is_mature_product` 區分新品/老品
3. **打折因子**: 在 `early_bird_momentum` 中用於獎勵新品

**典型值**:
- 新品: 0-10 條
- 正常商品: 10-100 條
- 熱門商品: 100-1000 條
- 超級爆品: 1000+ 條

---

#### 7.2.2 平均評分 (Score Mean)
```sql
AVG(score::float) AS score_mean
```

**意義**: 所有評論的平均星級評分（1-5）

**假設**: 高評分商品更可能成為爆品

**限制**:
- **評分膨脹**: 大多數商品評分集中在 4-5 分（左偏分布）
- **刷評問題**: 評分容易被操控
- **更新**: 已被 `sentiment_mean_recent`（近期情感）和 BERT 語意特徵取代

---

#### 7.2.3 累積按讚數 (Like Count Sum)
```sql
SUM(like_count::int) AS like_count_sum
```

**意義**: 所有評論獲得的按讚總數

**假設**: 高按讚數 → 評論有用/有趣 → 商品值得關注

**限制**:
- 與 `comment_count_pre` 高度相關（評論多 → 按讚多）
- 可能不是獨立訊號

---

### 7.3 近期統計 (Recency Statistics)

這些特徵捕捉**不同時間窗口**的評論數量。

#### 時間窗口定義
```sql
comment_count_7d   -- 最近 7 天
comment_count_30d  -- 最近 30 天
comment_count_90d  -- 最近 90 天
```

**用途**:
1. **活躍度指標**: 近期評論多 = 商品仍在活躍討論中
2. **衰退偵測**: `comment_count_7d` 遠小於 `comment_count_90d` → 熱度衰退
3. **比例計算**: 作為分母計算各種「近期比例」特徵

---

### 7.4 時間特徵 (Temporal Features)

#### 7.4.1 距離最後評論天數 (Days Since Last Comment)
```sql
EXTRACT(EPOCH FROM (cutoff::timestamp - MAX(comment_date))) / 86400.0 AS days_since_last_comment
```

**意義**: 從最後一條評論到現在經過了**多少天**

**重要性**: ⭐⭐⭐⭐⭐（最重要的活躍指標）

**語意解釋**:

| 天數範圍 | 意義 | 爆品機率 |
|---------|------|---------|
| **0-7 天** | 商品仍在活躍討論中 | 高 ✅ |
| **8-30 天** | 討論減緩但未停止 | 中 ⚠️ |
| **31-90 天** | 討論稀疏，可能已過熱度高峰 | 低 ❌ |
| **> 90 天** | 商品基本「死亡」，無人討論 | 極低 🚫 |

**特殊處理**:
```python
df.loc[df["comment_count_pre"] == 0, "days_since_last_comment"] = 365.0
```
**原因**: 沒有評論的商品，設為 365 天（最大懲罰）

---

### 7.5 趨勢分段 (Trend Segments)

將近 90 天切分為 **3 個 30 天區段**，用於捕捉「加速/減速」趨勢。

```sql
-- 最近 30 天 (0-30 days before cutoff)
comment_3rd_30d = COUNT(*) FILTER (WHERE comment_date >= cutoff - INTERVAL '30 days')

-- 中間 30 天 (31-60 days before cutoff)
comment_2nd_30d = COUNT(*) FILTER (WHERE comment_date >= cutoff - INTERVAL '60 days' 
                                     AND comment_date < cutoff - INTERVAL '30 days')

-- 早期 30 天 (61-90 days before cutoff)
comment_1st_30d = COUNT(*) FILTER (WHERE comment_date >= cutoff - INTERVAL '90 days' 
                                     AND comment_date < cutoff - INTERVAL '60 days')
```

**用途**: 計算 `ratio_recent30_to_prev60`（加速度比例）

#### 衍生特徵: 近期 vs 前期比例
```python
df["ratio_recent30_to_prev60"] = df["comment_3rd_30d"] / (df["comment_1st_30d"] + df["comment_2nd_30d"] + 1e-6)
```

**意義**: 最近 30 天的評論量 / 前 60 天的評論量

**語意解釋**:

| 比例值 | 意義 | 趨勢 |
|-------|------|------|
| **> 2.0** | 近期評論量是前期的 2 倍以上 | 🔥 強加速 |
| **1.0-2.0** | 近期評論量與前期相當或略高 | ⚠️ 穩定或微加速 |
| **0.5-1.0** | 近期評論量低於前期 | 📉 減速 |
| **< 0.5** | 近期評論量不到前期一半 | 🚫 快速衰退 |

**應用**: 在 `validated_velocity` 中作為加速度訊號

---

### 7.6 關鍵字比例 (Keyword Ratios)

基於**近期 90 天評論**計算各種內容特徵的比例。

#### 7.6.1 情感平均分 (Sentiment Mean Recent)
```sql
AVG(score::float) FILTER (WHERE comment_date >= cutoff - INTERVAL '90 days') AS sentiment_mean_recent
```

**意義**: 近 90 天評論的平均星級評分（1-5）

**vs `score_mean`**: 
- `score_mean`: 全部評論的平均（包含很久以前的）
- `sentiment_mean_recent`: **只看近期**，更能反映當前品質

---

#### 7.6.2 負評比例 (Negative Ratio Recent)
```sql
-- SQL 計算負評數
neg_count_recent = COUNT(*) FILTER (WHERE comment_date >= cutoff - INTERVAL '90 days' AND score <= 2)

-- Python 計算比例
df["neg_ratio_recent"] = df["neg_count_recent"] / (df["comment_count_90d"] + 1)
```

**意義**: 近 90 天中，低分評論（≤2 星）的比例

**假設**: 高負評比例 → 商品品質問題 → 不太可能成為好的爆品

**限制**:
- **粗糙**: 只用星級判斷，無法理解評論內容
- **已被取代**: `bert_negative_mean` 提供更精確的負面情緒偵測

---

#### 7.6.3 促銷比例 (Promo Ratio Recent)
```sql
-- SQL 計算促銷關鍵字出現次數
promo_count_recent = COUNT(*) FILTER (WHERE comment_date >= cutoff - INTERVAL '90 days' 
                                       AND comment_text ~ '促銷|特價|打折|滿額|免運|團購')

-- Python 計算比例
df["promo_ratio_recent"] = df["promo_count_recent"] / (df["comment_count_90d"] + 1)
```

**意義**: 近 90 天中，提到「促銷」相關詞彙的評論比例

**假設**: 
- 高促銷比例 → 商品**靠價格戰**吸引討論，而非品質
- 促銷驅動的熱度不持久

**限制**:
- **Regex 匹配**: 可能誤判（例如「不是特價」也會匹配）
- **語境問題**: 「這個價格太貴不值得」不會被匹配，但其實是負面評價

---

### 7.7 歷史變動 (Historical Changes)

這些特徵捕捉商品在**銷售數據**上的歷史變化模式。

#### 7.7.1 曾經有銷量變化 (Had Any Change Pre)
```sql
MAX(CASE WHEN prev_sales IS NOT NULL 
          AND sales_count IS DISTINCT FROM prev_sales 
     THEN 1 ELSE 0 END) AS had_any_change_pre
```

**意義**: 在訓練截止日前，商品的銷量**是否曾經變化過**（0/1）

**用途**:
- **活躍度指標**: 1 = 商品有銷售記錄，0 = 可能是「殭屍商品」
- **過濾器**: 排除從未有銷量變化的商品

---

#### 7.7.2 歷史增長次數 (Num Increases Pre)
```sql
COUNT(*) FILTER (WHERE prev_sales IS NOT NULL 
                   AND sales_count > prev_sales) AS num_increases_pre
```

**意義**: 在訓練截止日前，銷量**增長的次數**

**語意解釋**:

| 增長次數 | 意義 |
|---------|------|
| **0** | 從未增長（新品或滯銷品） |
| **1-3** | 偶爾增長（正常商品） |
| **4-10** | 經常增長（穩定熱銷品） |
| **> 10** | 持續增長（超級爆品或刷單） |

**用途**: 區分「首次爆發」vs「持續成長」

---

### 7.8 特徵層級結構

```
基礎與統計層
├── 靜態屬性 (商品固有特性)
│   ├── price
│   └── has_image_urls, has_video_url, has_reply_content
│
├── 累積數據 (全生命週期)
│   ├── comment_count_pre (⭐⭐⭐⭐⭐ 最重要控制變數)
│   ├── score_mean
│   └── like_count_sum
│
├── 近期統計 (時間窗口)
│   ├── comment_count_7d
│   ├── comment_count_30d
│   └── comment_count_90d (⭐⭐⭐⭐ 常用分母)
│
├── 時間特徵 (活躍度)
│   └── days_since_last_comment (⭐⭐⭐⭐⭐ 最重要活躍指標)
│
├── 趨勢分段 (加速度基礎)
│   ├── comment_1st_30d, comment_2nd_30d, comment_3rd_30d
│   └── ratio_recent30_to_prev60
│
├── 關鍵字比例 (內容特性)
│   ├── sentiment_mean_recent
│   ├── neg_ratio_recent
│   └── promo_ratio_recent
│
└── 歷史變動 (銷售軌跡)
    ├── had_any_change_pre
    └── num_increases_pre
```

---

### 7.9 為什麼需要「控制變數」？

#### 問題: 不控制基礎變數會發生什麼？

**案例 1: 音量效應**
- 商品 A: 從 1 條 → 5 條評論（+400%）
- 商品 B: 從 100 條 → 120 條評論（+20%）
- **不控制**: 模型可能把 A 判為爆品（加速度高）
- **控制後**: `early_bird_momentum` 會獎勵 A，但 `validated_velocity` 會懲罰 A（音量太低）

**案例 2: 活躍度效應**
- 商品 C: 100 條評論，但最後一條是 90 天前
- 商品 D: 50 條評論，但最後一條是昨天
- **不控制**: 模型可能優先考慮 C（累積評論多）
- **控制後**: `days_since_last_comment` 會懲罰 C，獎勵 D

---

### 7.10 關鍵洞察

#### 洞察 1: 絕對值 vs 相對值
**基礎層提供「絕對值」，高階層計算「相對值」**

- `comment_count_90d` = 絕對值（100 條）
- `ratio_recent30_to_prev60` = 相對值（近期是前期的 2 倍）
- **兩者結合**: `validated_velocity = ratio × log1p(volume)` → 既看趨勢又看音量

#### 洞察 2: 時間衰減
**近期數據比歷史數據更重要**

- `score_mean` (全時期) → `sentiment_mean_recent` (近 90 天)
- `comment_count_pre` (累積) → `comment_count_90d` (近期)

#### 洞察 3: 多粒度時間窗口
**不同窗口捕捉不同頻率的訊號**

- `comment_count_7d`: 捕捉「本週爆發」
- `comment_count_30d`: 捕捉「本月趨勢」
- `comment_count_90d`: 捕捉「季度穩定性」

#### 洞察 4: 活躍度 > 累積量
**`days_since_last_comment` 比 `comment_count_pre` 更能預測未來潛力**

- 1000 條評論但 90 天無更新 → 過氣商品
- 10 條評論但每天都有新的 → 潛力新品

**結論**: 基礎統計層不是「主角」，而是「配角」—— 它們提供**基準線**和**控制變數**，讓高階特徵（物理層、BERT 層、融合層）能夠更準確地捕捉「爆品訊號」。

