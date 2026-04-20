# BSF（行為與資訊結構特徵）技術問答

> 產生日期：2026-04-20  
> 對應功能：`--feature-set +psych`  
> 主要原始碼：`Model/data_loader.py`、`fetch_spam_refs.py`

---

## 一、`feat_ncd_spam` 與 `feat_compression_ratio` 是否與 `feat_entropy_tfidf` 有關係？

### 結論：**三個特徵完全獨立計算，互不依賴，但都屬於「資訊理論」範疇**

這三個特徵雖然都使用「壓縮 / 資訊熵」的概念來偵測重複性，但**計算管道彼此分離**：

| 特徵 | 計算方式 | 核心理論 | 輸入資料範圍 |
|------|----------|----------|-------------|
| `feat_entropy_tfidf` | TF-IDF → KMeans(k≤5) → Shannon Entropy | 資訊熵（群組分布多元程度） | 近 90 天最多 100 則評論 |
| `feat_compression_ratio` | `len(UTF-8原始) / len(zlib.compress(原始))` | LZ77 壓縮理論（重複性偵測） | 截止日前**全部**評論合併 |
| `feat_ncd_spam` | NCD 公式比較評論與刷評範本的壓縮距離 | 標準化壓縮距離（NCD） | 截止日前**全部**評論合併 |

#### 計算順序（在 `data_loader.py` 中）

```
Step 1（約 L956–L1057）：計算 feat_entropy_tfidf、feat_entropy_emb、
                         feat_temporal_burstiness、feat_lexical_diversity
↓
Step 2（約 L1062–L1097）：計算 feat_compression_ratio、feat_ncd_spam
```

兩個步驟使用的 Python 變數、輸入文字、計算過程**完全沒有交叉**。
`feat_entropy_tfidf` 的結果不會傳入 `feat_compression_ratio` 或 `feat_ncd_spam` 的計算。

#### 唯一的間接關聯（融合特徵層面）

- `feat_entropy_tfidf` 被下游的 `momentum_tfidf` 使用（`kin_acc_abs × (category_fit_score+0.5) × (feat_entropy_tfidf+0.5)`）
- `feat_ncd_spam` 被 Surrogate Structure Features 的 `interaction_ncd_fit` 使用
- 但這些都是**後續融合特徵**，不是三者之間的直接依賴

#### 直覺上的差異對比

| 面向 | `feat_entropy_tfidf` | `feat_compression_ratio` | `feat_ncd_spam` |
|------|---------------------|--------------------------|----------------|
| 問的問題 | 評論主題多元嗎？ | 整體重複性高嗎？ | 像不像刷評範本？ |
| 高分代表 | 主題分散（可能真實） | 資訊密度高（不易壓縮，真實） | 不像刷評（個性化） |
| 低分代表 | 主題集中（可能刷評） | 高重複（易壓縮，刷評） | 極似範本（刷評） |

---

## 二、`feat_ncd_spam` 和 `feat_compression_ratio` 有做 K-Means 分群嗎？

### 結論：**沒有，這兩個特徵完全不涉及 K-Means**

K-Means 只出現在 `feat_entropy_tfidf` 和 `feat_entropy_emb` 的計算流程中：

```python
# data_loader.py L993–L1001（feat_entropy_tfidf 的計算）
n_clusters = min(len(texts), 5)
if n_clusters > 1:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_local)
    counts = np.bincount(labels)
    probs = counts / len(labels)
    ent = entropy(probs, base=2)
    df.at[idx, "feat_entropy_tfidf"] = ent
```

`feat_compression_ratio` 和 `feat_ncd_spam` 的計算（L1062–L1097）只用到 `zlib.compress()`，是純粹的**壓縮演算法**，不做任何分群。

### 三個特徵的演算法對比

```
feat_entropy_tfidf 流程：
  評論文字 → TF-IDF 向量化（max_features=100）
           → KMeans 分群（k = min(n_texts, 5)）
           → 各群樣本比例 p_i
           → Shannon Entropy = -Σ(p_i × log₂(p_i))

feat_compression_ratio 流程：
  評論文字 → UTF-8 encode → zlib.compress()
           → ratio = len(原始位元組) / len(壓縮後位元組)

feat_ncd_spam 流程：
  評論文字 → UTF-8 encode → zlib.compress()  → C(x)
  刷評範本 → 預先計算            → LEN_C_REF = C(y)
  兩者合併 → UTF-8 encode → zlib.compress()  → C(x + y)
           → NCD = (C(x+y) - min(C(x), C(y))) / max(C(x), C(y))
           → clip 至 [0, 1]
```

---

## 三、`SPAM_REFERENCES` 怎麼來的？

### 結論：**由 `fetch_spam_refs.py` 自動從資料庫查詢後手動複製貼入**

#### 3.1 生成工具：`fetch_spam_refs.py`

這支腳本（根目錄下）的邏輯是：

```python
# fetch_spam_refs.py 核心 SQL
SELECT comment_text, COUNT(*) as freq
FROM product_comments
WHERE LENGTH(comment_text) >= 6    -- 過濾太短（< 6 字元）
  AND LENGTH(comment_text) <= 50   -- 過濾長文章（> 50 字元，視為有機評論）
GROUP BY comment_text
ORDER BY freq DESC
LIMIT 20;
```

過濾條件的設計邏輯：
- **6 字元下限**：太短的文字（如「好」、「OK」）無法提供足夠的模式資訊給 NCD 運算
- **50 字元上限**：長篇評論通常是有機的真實評論，刷評範本傾向短而一致

腳本執行後會列印 Top 20 高頻重複評論，並自動排版為可直接複製的 Python 程式碼：

```
SPAM_REFERENCES = [
    "👍👍👍👍👍👍👍👍👍👍",
    "商品包裝完整，出貨速度快速。",
    ...
]
```

#### 3.2 手動篩選 Top 10 後複製進 `Model/data_loader.py`

查詢結果的前 10 筆被手動選取，貼入 `data_loader.py` 的模組層級變數（L32–L43）：

```python
# data_loader.py L31–L47
# Mined from DB Analysis (Top frequent templates on Momo)
SPAM_REFERENCES = [
    "👍👍👍👍👍👍👍👍👍👍",
    "商品包裝完整，出貨速度快速。",
    "出貨快速，包裝完整。",
    "出貨速度快，包裝完整。",
    "出貨速度快，包裝完整",
    "商品品質不錯 價格合理 送貨速度可以 推薦~",
    "出貨速度快，價格便宜！",
    "出貨速度快，商品包裝完整",
    "商品不錯喔，已回購多次，物美價廉",
    "出貨速度超快",
]
# Pre-compute the compressed reference for efficiency
REF_CONCAT = " ".join(SPAM_REFERENCES)
REF_BYTES = REF_CONCAT.encode('utf-8')
LEN_C_REF = len(zlib.compress(REF_BYTES))  # 預先壓縮，避免每個商品重複計算
```

> [!NOTE]
> `特徵說明.md` 中寫的「共 9 筆」實際上是描述錯誤，原始碼中實際為 **10 筆**。
> `SPAM_REFERENCES` 清單有 10 個字串（index 0–9），不是 9 筆。文件需要修正。

#### 3.3 完整流程圖

```
① 研究者執行：python fetch_spam_refs.py
   ↓
② SQL 查詢 product_comments 資料表
   條件：6 <= LENGTH(comment_text) <= 50
   排序：按 comment_text 出現頻次 DESC
   取前 20 名
   ↓
③ 腳本列印 Top 20 高頻「短評論模板」及對應 Python 程式碼
   ↓
④ 研究者手動挑選 Top 10，複製貼入 Model/data_loader.py
   ↓
⑤ 模組載入時預先計算 REF_CONCAT、REF_BYTES、LEN_C_REF
   ↓
⑥ 每個商品計算 feat_ncd_spam 時，以 LEN_C_REF 作為 C(y) 參考值
```

#### 3.4 為什麼合併後再壓縮（NCD 原理說明）

NCD 公式來自 Kolmogorov 複雜度理論的近似：

```
NCD(x, y) = (C(x+y) - min(C(x), C(y))) / max(C(x), C(y))
```

- `C(x)` = 壓縮評論文字的位元組數
- `C(y)` = 壓縮全部刷評範本的位元組數（預先計算）
- `C(x+y)` = 壓縮「評論 + 範本」的位元組數

直覺：LZ77（zlib 底層）會利用**重複子字串**來壓縮。若評論 x 與範本 y 用詞高度重疊，壓縮器能在 x 中找到 y 的模式，`C(x+y)` 就會比 `C(x)+C(y)` 小很多，NCD 值就低（像刷評）。

---

## 四、摘要對照表

| 問題 | 答案 |
|------|------|
| `feat_ncd_spam` 與 `feat_entropy_tfidf` 有關聯嗎？ | ❌ 無關，完全獨立計算管道 |
| `feat_compression_ratio` 與 `feat_entropy_tfidf` 有關聯嗎？ | ❌ 無關，完全獨立計算管道 |
| `feat_ncd_spam` 有做 K-Means 嗎？ | ❌ 沒有，只用 zlib 壓縮 |
| `feat_compression_ratio` 有做 K-Means 嗎？ | ❌ 沒有，只用 zlib 壓縮 |
| K-Means 出現在哪些特徵？ | ✅ 只有 `feat_entropy_tfidf` 和 `feat_entropy_emb` |
| `SPAM_REFERENCES` 是人工挑選的嗎？ | ✅ 是，由 `fetch_spam_refs.py` 查資料庫後，研究者手動選 Top 10 |
| `SPAM_REFERENCES` 幾筆？ | ✅ 實際上是 **10 筆**（特徵說明.md 寫的「9 筆」有誤） |

---

## 五、待修正事項

> [!CAUTION]
> `doc/特徵說明.md` 第 238 行寫道「刷評範本庫（共 9 筆，來自資料庫高頻模板分析）」，
> 實際上 `SPAM_REFERENCES` 清單有 **10 個元素**。
> 建議將「9 筆」更正為「10 筆」。

---

*本文件依據 `Model/data_loader.py`（L20, L32–L47, L956–L1097）及 `fetch_spam_refs.py`（L18–L57）產生。*
