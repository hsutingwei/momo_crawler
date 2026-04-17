# SF Verbalizer 擴充流程 — 工程規格文件

**Pipeline：** `sf_01_filter_highconf.py` → `sf_02_mine_candidates.py` → human review → updated `label_map`  
**範圍：** 以語料導向方式擴充 NLI seed verbalizer；不涉及模型重訓。  
**最後更新：** 2026-04-17

---

## 1. Pipeline 總覽

### 目的

這條 pipeline **不是**用來訓練新的分類器。  
它的用途是擴充 zero-shot NLI 打分步驟（`comment_semantic_scores`）所使用的 seed `label_map`，讓 verbalizer 能涵蓋更多實際出現在 momo 電商評論中的表面表達變體。

NLI 模型會把每個 label 的文字描述（verbalizer）當成 hypothesis。若 verbalizer 太窄，評論即使表達了相同語意，但字面形式不同，也可能分數偏低而被漏掉。這條 pipeline 會從高信心評論子集中挖掘各 label 的代表性 unigram，經人工審閱後再回填到 verbalizer 中。

### Pipeline 流程

```text
comment_semantic_scores          （seed NLI 分數，已預先算好）
         |
         v
sf_01_filter_highconf.py         （threshold + exclusion filtering）
         |
         v
sf_highconf_comments             （各 label 的高信心評論子集）
         |
         v
sf_02_mine_candidates.py         （discriminative score + 多層過濾）
         |
         v
sf_keyword_candidates            （排序後的 unigram 候選詞，附分數與 example）
         |
         v
human review                     （對每個 token 做 accept / keep / reject）
         |
         v
updated label_map / verbalizers  （擴充後的 NLI hypothesis 字串）
```

### 這條 Pipeline 不做什麼

- 不屬於 supervised classifier training loop
- 不會自動更新 verbalizer（人工審閱為必要步驟）
- 目前不做 bigram / trigram mining（現階段僅支援 unigram）
- 不會取代 SF 的理論定義

---

## 2. Step 1: High-Confidence Comment Filtering

### 2.1 功能

`sf_01_filter_highconf.py` 的責任是從 `comment_semantic_scores`（seed NLI 已打分的全量評論）中，為每個 label 篩出**語意信號純度夠高的評論子集**，寫入 `sf_highconf_comments`。

直接對全量評論做詞彙挖掘無效：全量語料的高頻詞是通用詞，不反映任何語意維度。篩選高信心子語料是為了讓後一步的 discriminative mining 有乾淨的語意一致語料可以計算。

### 2.2 Inputs

| Source                    | Type         | 說明                                                         |
| ------------------------- | ------------ | ------------------------------------------------------------ |
| `comment_semantic_scores` | DB table     | seed NLI 對全量評論的每個 label 打分結果，每行對應一則評論   |
| `LABEL_CONFIG`            | in-code dict | 每個 label 的 source column、預設 threshold、exclusion rules |
| `--threshold`             | CLI arg      | 可覆蓋各 label 預設門檻（僅在 single-label 模式有效）        |
| `--run-id`                | CLI arg      | 本輪篩選的版本 ID，寫入 output 表，供後續步驟 JOIN 使用      |

### 2.3 Filtering Logic

每個 label 套用兩層條件：

**Layer 1 — Primary threshold：** 目標 label 的 NLI 分數 ≥ threshold  
**Layer 2 — Exclusion constraints：** 其他指定 label 的分數 < exclusion upper bound（防止混合語意污染）

各 label 設定如下：

| Label                  | Source Column         | Threshold | Exclusion Rules                                       |
| ---------------------- | --------------------- | --------- | ----------------------------------------------------- |
| High_Arousal           | `score_arousal`       | 0.70      | `score_advertisement < 0.40`、`score_negative < 0.35` |
| High_Novelty           | `score_novelty`       | 0.70      | `score_advertisement < 0.40`、`score_negative < 0.35` |
| High_Repurchase_Intent | `score_repurchase`    | 0.70      | `score_advertisement < 0.40`                          |
| Negative_Complaint     | `score_negative`      | 0.65      | `score_advertisement < 0.45`                          |
| Advertisement          | `score_advertisement` | 0.65      | （無排除條件）                                        |

**Exclusion 設計意圖：**
- High_Arousal / High_Novelty 若同時有高 advertisement 分數 → 該評論可能是業配造勢，不是真實情感，排除
- High_Arousal / High_Novelty 若同時有高 negative 分數 → 混合語氣（例如：驚訝但失望），排除
- Repurchase_Intent 排除高 advertisement → 防止業配型「推薦回購」污染
- Advertisement 無排除條件 → 業配評論通常獨立語意，不混淆

**實際 SQL 結構（簡化版）：**

```sql
SELECT comment_id, score_arousal AS score
FROM comment_semantic_scores
WHERE score_arousal >= 0.70
  AND score_advertisement < 0.40
  AND score_negative < 0.35
ORDER BY score_arousal DESC;
```

**Sanity checks（程式內建）：**
- 結果 < 100 筆：記錄 WARNING，建議降 threshold
- 結果 > 20,000 筆：記錄 WARNING，建議升 threshold 確保純度

### 2.4 Output

**Target table：`sf_highconf_comments`**

| Column            | 說明                                  |
| ----------------- | ------------------------------------- |
| `run_id`          | 本輪篩選版本 ID（例如 `nli-seed-v1`） |
| `label_name`      | 語意 label（例如 `High_Novelty`）     |
| `comment_id`      | 評論 ID                               |
| `agg_score`       | 該 label 的 NLI 分數                  |
| `score_threshold` | 本輪使用的 threshold 值               |

`ON CONFLICT (run_id, label_name, comment_id) DO NOTHING`：冪等設計，重跑不重複寫入。

這張表是整個 pipeline 的「分析起點」，後續所有 mining 操作的 label 子語料邊界均由它界定。

### 2.5 CLI / Runtime Behavior

| 參數                    | 說明                                           |
| ----------------------- | ---------------------------------------------- |
| `--run-id <id>`         | 必填。本輪篩選的版本 ID，例如 `nli-seed-v1`    |
| `--label <name>`        | 指定單一 label 執行                            |
| `--all-labels`          | 批次跑全部 5 個 label，使用各自預設 threshold  |
| `--threshold <float>`   | 覆蓋預設 threshold，僅在 single-label 模式有效 |
| `--dry-run`（預設開啟） | 只印統計數字，不寫入 DB                        |
| `--no-dry-run`          | 正式寫入 `sf_highconf_comments`                |

`--label` 和 `--all-labels` 互斥，不可同時使用。

---

## 3. Step 2: Candidate Mining

### 3.1 功能

`sf_02_mine_candidates.py` 從 `sf_highconf_comments` 指定的評論子集中，計算每個 unigram 對該 label 的語意鑑別力分數（discriminative score），經多層過濾後寫入 `sf_keyword_candidates`，供人工審閱用。

**這支程式不會自動更新 verbalizer。** 它只產生候選詞列表。

### 3.2 Inputs

| Source                 | Type     | 說明                                                                   |
| ---------------------- | -------- | ---------------------------------------------------------------------- |
| `sf_highconf_comments` | DB table | Step 1 的輸出，界定本輪 label 子語料的 comment_id 集合                 |
| `tfidf_term_freq`      | DB table | 全語料的 term-document TF 矩陣，per `(corpus_id, comment_id, term_id)` |
| `tfidf_doc_freq`       | DB table | 全語料各 term 的 document frequency                                    |
| `tfidf_vocab`          | DB table | term_id → token 文字映射，需指定 `pipeline_version`                    |
| `tfidf_corpus`         | DB table | 全語料總文件數 `total_docs`，per `corpus_id`                           |
| `comment_tokens`       | DB table | CKIP 斷詞 + 詞性標記結果，用於 POS filtering                           |

### 3.3 Candidate Scoring Formula

#### 數學定義

令：
- $\mathcal{D}_l$ = label $l$ 的高信心評論子集（由 `sf_highconf_comments` 定義）
- $\mathcal{D}$ = 全語料（`tfidf_corpus` 的 `total_docs`）
- $\text{TF}_l(w)$ = term $w$ 在 $\mathcal{D}_l$ 中所有文件的詞頻加總（`label_total_tf`）
- $|\mathcal{D}_l|$ = $\mathcal{D}_l$ 的文件數（`label_size`）
- $\text{DF}(w)$ = term $w$ 在 $\mathcal{D}$ 中出現的文件數（`global_df`）
- $|\mathcal{D}|$ = 全語料總文件數（`total_docs`）
- $\varepsilon$ = smoothing constant = `0.001`

$$
\text{disc\_score}(w, l) = \ln\!\left(\frac{\,\text{TF}_l(w) \;/\; |\mathcal{D}_l|\,}{\;\text{DF}(w) \;/\; |\mathcal{D}| \;+\; \varepsilon\;}\right)
$$

#### 各項解釋

| 項目                                      | 意義                                                                             |
| ----------------------------------------- | -------------------------------------------------------------------------------- |
| 分子 $\text{TF}_l(w) / \|\mathcal{D}_l\|$ | term $w$ 在 label 子語料中的平均每文件詞頻：反映該詞「在語意相關評論裡有多密集」 |
| 分母 $\text{DF}(w) / \|\mathcal{D}\|$     | term $w$ 在全語料的文件覆蓋率：反映該詞「全語料有多普遍」                        |
| $\ln(\cdot)$                              | 拉開數量級差距，讓分數分布更線性                                                 |
| $\varepsilon$                             | Smoothing：防止低 DF 詞（僅出現於極少全語料文件）因分母趨近 0 而分數異常爆高     |

#### Score 解讀

| Score 範圍 | 意義                                                                             |
| ---------- | -------------------------------------------------------------------------------- |
| `> 0`      | 該 term 在 label 子語料中的密度高於全語料平均 → 具 label 特異性，候選 verbalizer |
| `≈ 0`      | 在 label 子語料中的分布和全語料沒有顯著差異 → 無鑑別力                           |
| `< 0`      | 在 label 子語料中的密度低於全語料平均 → 反指標                                   |

#### 實作對應（SQL CTE）

```sql
LN(
    (label_total_tf / label_n_docs)
    /
    ((global_df / total_docs) + 0.001)
) AS discriminative_score
```

計算全在 DB 端完成（PostgreSQL CTE），Python 只做後處理。

### 3.4 Filtering Layers

過濾依序執行，前層結果傳入下層。

---

#### Layer 1：SQL Base Filtering（DB 端）

| 條件                                 | 說明                                              | 必要性                                 |
| ------------------------------------ | ------------------------------------------------- | -------------------------------------- |
| `label_total_tf >= min_tf`（預設 5） | 排除在 label 子語料中詞頻過低的詞，確保有足夠樣本 | 防止偶發詞得高分                       |
| `label_doc_freq >= min_df`（預設 3） | 排除只出現在個別文件的詞                          | 防止單篇評論特有用語進候選池           |
| `LENGTH(token) >= 2`                 | 排除單字元 token                                  | 助詞、標點等                           |
| `token !~ '^[0-9]+$'`                | 排除純數字                                        | 日期、型號、數量等無語意詞             |
| `token NOT IN BASIC_STOPWORDS`       | 排除助詞、連接詞、電商無意義高頻詞                | 例：「的」、「了」、「商品」、「賣家」 |

`BASIC_STOPWORDS` 刻意不包含情感詞（「超」、「好」、「真」等），這類詞留給 Layer 3 或人工審閱判斷。

---

#### Layer 2：POS Filtering（Python 端，CKIP 詞性）

從 `comment_tokens` 查詢每個候選詞在 label 子語料中出現最多的詞性（`MODE() WITHIN GROUP` 取眾數），只保留以下 POS whitelist：

| POS Tag | 類型                 | 代表詞                                   |
| ------- | -------------------- | ---------------------------------------- |
| `Na`    | 普通名詞             | 體驗、驚喜、新奇                         |
| `Nb`    | 專有名詞             | 品牌名（由 `--exclude-nb` 控制是否排除） |
| `VH`    | 狀態動詞（形容詞用） | 好用、神奇、厲害                         |
| `VE`    | 存在動詞             | 有                                       |
| `VK`    | 心理動詞             | 喜歡、推薦                               |
| `VA`    | 不及物動詞           | 值得                                     |
| `A`     | 形容詞               | 新鮮、獨特                               |

**`--exclude-nb`：** 將 `Nb`（專有名詞）從有效白名單中移除，用於過濾品牌名。  
建議預設開啟：品牌名因評論集中而可能在某 label 子語料高頻，但品牌名本身不是 verbalizer。

副詞（`D`）不在白名單內：「超」、「很」、「真」在正向評論中近乎普遍出現，缺乏 label 特異性，一律排出，不送入 verbalizer 候選池。

`dominant_pos` 為 `NULL` 的 token（TF-IDF 有收但 `comment_tokens` 未索引的邊緣詞）預設**保留**，由人工審閱決定。

---

#### Layer 3：Generic Blacklist（Python 端）

對**所有 label** 都無鑑別力的泛化詞，一律排除，不區分 label：

| 類別            | 詞例                               |
| --------------- | ---------------------------------- |
| 評論元語言      | 評價、評論、評分、留言、回饋       |
| 泛化體驗詞      | 效果、感受、功效、功能、有感、有效 |
| 泛化認知動詞    | 知道、覺得、認為、感覺             |
| Meta-commentary | 聽說、觀察                         |
| 泛化期待詞      | 希望、期待、期望                   |
| 泛化正向評語    | 不錯                               |
| 人際關係詞      | 朋友、家人、老婆、老公、媽媽、爸爸 |
| 口感 / 味道描述 | 味道、口味、口感、甜度、風味       |
| 泛化商品詞      | 品牌、品質                         |

設計原則：只放「高確定性的跨 label 無鑑別力詞彙」，不貪多。情感評價詞（超/好/真）不在此列，留給人工審閱。

可用 `--no-generic-blacklist` 關閉，用於 debug 時對比。

---

#### Layer 4：Label-Specific Blacklist（Python 端）

各 label 的專屬排除詞，針對「在此 label 子語料中意外高頻，但語意上是語料偏誤（corpus artifact）而非語意信號」的詞：

| Label          | 排除詞                                                               | 排除原因                                                                            |
| -------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `High_Novelty` | 葉黃素、益生菌、膠原、蛋白、生醫、魚油、膠囊、維生素、維他命、乳酸菌 | 本語料含大量保健品評論，消費者初次嘗試保健品時表達新奇 → 成分名高頻，但不是新奇語意 |
| `High_Novelty` | 成份、成效、日期                                                     | 同上，保健品語料 artifact                                                           |
| `High_Novelty` | 鮭魚、鱈魚、甜甜、酸酸、繽紛                                         | 食品/零食語料的口味描述 artifact                                                    |
| `High_Novelty` | 天王、中江、歐可                                                     | 品牌名被 CKIP 標為 `Na` 而漏網（`--exclude-nb` 未能攔截）                           |
| 其他 4 labels  | （空集合，待補）                                                     | 待後續 mining run 中發現後逐步補充                                                  |

可用 `--no-label-blacklist` 關閉，用於 debug。

---

#### Layer 5：Top-K Truncation

依 `discriminative_score DESC` 排序後，取前 `--top-k` 筆（預設 150）寫入輸出表。  
實作上預先 fetch `top_k * 4` 筆（`QUERY_BUFFER_FACTOR = 4`），預留 Layer 2–4 過濾後仍能湊足 `top_k`。

### 3.5 Output

**Target table：`sf_keyword_candidates`**

| Column                 | 說明                                 |
| ---------------------- | ------------------------------------ |
| `mining_run`           | 本輪 mining 的版本 ID                |
| `label_name`           | 所屬 label                           |
| `token`                | 候選詞文字                           |
| `candidate_type`       | 固定為 `unigram`（目前只做 unigram） |
| `label_total_tf`       | 在 label 子語料中的總詞頻            |
| `label_doc_freq`       | 在 label 子語料中的文件數            |
| `global_df`            | 在全語料中的文件數                   |
| `discriminative_score` | 差異化分數（越高越具 label 特異性）  |
| `dominant_pos`         | 最常見 CKIP 詞性                     |
| `example_comment_ids`  | 得分最高的前 3 筆評論 ID（ARRAY）    |
| `is_approved`          | 人工審閱結果（不被 pipeline 覆蓋）   |
| `review_note`          | 審閱備註（不被 pipeline 覆蓋）       |

**Upsert 設計：** `ON CONFLICT (mining_run, label_name, token, candidate_type) DO UPDATE`，重跑時更新分數，但 `is_approved` / `review_note` / `reviewed_at` **刻意不覆蓋**，保留人工審閱結果。

**`example_comment_ids` 的用途：** 人工審閱時查原始評論脈絡，確認候選詞在實際語境中的語意是否符合該 label 定義。這是避免純靠統計分數決定納入的關鍵機制。

---

## 4. Data Model / Table Roles

| Table                     | Role                                             | Key Columns                                                                                                                      | Produced By                      | Consumed By                                                        |
| ------------------------- | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------ |
| `comment_semantic_scores` | Seed NLI 全量打分結果                            | `comment_id`, `score_arousal`, `score_novelty`, `score_repurchase`, `score_negative`, `score_advertisement`                      | NLI scoring pipeline（upstream） | `sf_01`                                                            |
| `sf_highconf_comments`    | 各 label 高信心評論 ID 子集，pipeline 的分析邊界 | `run_id`, `label_name`, `comment_id`, `agg_score`, `score_threshold`                                                             | `sf_01`                          | `sf_02`（JOIN 取 label 子語料）、`sf_02`（取 example_comment_ids） |
| `tfidf_term_freq`         | 全語料 term-document TF 矩陣                     | `corpus_id`, `comment_id`, `term_id`, `tf`                                                                                       | TF-IDF pipeline（upstream）      | `sf_02`（label 子語料 TF 聚合）                                    |
| `tfidf_doc_freq`          | 全語料各 term 的 document frequency              | `corpus_id`, `term_id`, `df`                                                                                                     | TF-IDF pipeline（upstream）      | `sf_02`（discriminative score 分母）                               |
| `tfidf_vocab`             | term_id → token 文字映射                         | `term_id`, `pipeline_version`, `token`                                                                                           | TF-IDF pipeline（upstream）      | `sf_02`（取 token 文字、過濾純數字／長度）                         |
| `tfidf_corpus`            | 全語料 total_docs 紀錄                           | `corpus_id`, `total_docs`                                                                                                        | TF-IDF pipeline（upstream）      | `sf_02`（分母歸一化）                                              |
| `comment_tokens`          | CKIP 斷詞 + 詞性標記結果                         | `comment_id`, `pipeline_version`, `token`, `pos_tag`                                                                             | CKIP NLP pipeline（upstream）    | `sf_02`（POS filtering）                                           |
| `sf_keyword_candidates`   | 候選詞池，附分數與審閱欄位                       | `mining_run`, `label_name`, `token`, `discriminative_score`, `dominant_pos`, `example_comment_ids`, `is_approved`, `review_note` | `sf_02`                          | human review → updated `label_map`                                 |

---

## 5. Human Review Layer

### 為什麼不能直接全部回填

`sf_keyword_candidates` 的候選詞是統計挖掘結果，存在以下已知問題，無法直接採用：

1. **語料偏誤（corpus artifact）：** 語料中特定品類（保健品、食品）評論密集，品類特有詞彙（成分名、口味詞）可能意外得高分，即使 label-specific blacklist 已處理，仍有漏網。
2. **品牌名殘留：** `--exclude-nb` 只能排除 CKIP 標為 `Nb` 的詞。部分品牌名被標為 `Na`，需人工識別。
3. **物流詞 / 促銷詞：** 「退貨」、「免運」等物流或促銷詞可能在 Negative_Complaint 子語料高頻，但這類詞不適合直接成為情感語意的 verbalizer descriptor。
4. **語意歧義：** 某些詞在不同語境下承載不同語意（例：「發現」可以是「發現新奇事物」也可以是「發現問題」），需要查閱 `example_comment_ids` 原始評論才能判斷。

### 人工審閱要做的事

1. 按 `discriminative_score DESC` 從頭逐筆看
2. 查閱 `example_comment_ids` 對應的原始評論，確認該詞在脈絡中的語意
3. 對每筆候選詞做以下三種判斷，記入 `is_approved` 與 `review_note`

### 三種審閱結果

| 判斷                | 條件                                                           | 動作                                      |
| ------------------- | -------------------------------------------------------------- | ----------------------------------------- |
| **accept**          | 語意明確符合該 label 定義，example comments 中穩定以此語意出現 | 納入擴充版 verbalizer                     |
| **keep-for-review** | 語意具歧義，或 example 數量不足以判斷，或跨 label 出現         | 暫不納入，記錄觀察，待語料擴充後重評      |
| **reject**          | 確認為品牌名、品類名、物流詞、促銷詞、語料偏誤 artifact        | 排除，視情況補入 label-specific blacklist |

### 避免污染 verbalizer 的操作原則

- **品牌名：** dominant_pos 為 `Nb` 應已被 `--exclude-nb` 排除；標為 `Na` 的品牌名需靠 example comments 辨識，reject 後補入 label-specific blacklist
- **品類名：** 同品類出現在多個 label 的候選詞中 → 跨 label 共現代表無鑑別力 → reject
- **物流詞：** 例：「退貨」在 Negative_Complaint 可能高頻，但「退貨」描述的是行為結果，非情感語意 → reject 或 keep-for-review
- **促銷詞：** 例：「免運」、「折扣」 → reject，加入 generic blacklist

---

## 6. Seed-to-Verbalizer Refinement Logic

### Input Verbalizer（Seed）

```python
label_map = {
    "High_Arousal":           "驚豔、激動、太神了",
    "High_Novelty":           "新奇、初次體驗、相見恨晚",
    "High_Repurchase_Intent": "回購意願高、忠實粉絲",
    "Negative_Complaint":     "憤怒、失望、反推",
    "Advertisement":          "業配、廣告、湊字數"
}
```

這個 seed 是 NLI 打分的初始 hypothesis。它的問題是：每個 label 只有 2-3 個短語，在中文口語評論中表達覆蓋面不足。

### Candidate Generation

1. **sf_01** 以 seed 分數篩出各 label 的高信心評論子集
2. **sf_02** 在子語料中計算 discriminative score，挖掘在子語料中集中度高於全語料的詞彙
3. 候選詞附帶 `discriminative_score`、`dominant_pos`、`example_comment_ids` 寫入 `sf_keyword_candidates`

### Review

- 人工依 Section 5 的流程逐筆判斷
- 只有 **accept** 的詞才進入下一步

### Output Verbalizer（Expanded）

```python
label_map = {
    "High_Arousal":           "驚豔、驚喜、不可思議、太神了、厲害、無敵、震撼、驚人",
    "High_Novelty":           "新奇、初次體驗、相見恨晚、特別、不同、難得、發現、神奇",
    "High_Repurchase_Intent": "回購、長期、固定、首選、忠實、囤貨、會再買、一直買",
    "Negative_Complaint":     "憤怒、失望、反推、客服、傻眼、過期、嚴重、可惜、問題、退貨",
    "Advertisement":          "代言、代言人、官網、網頁、網評、風評、老牌子",
}
```

### 關鍵設計原則

> **Theoretical definition does not change.**  
> Only verbalizer coverage is expanded.

- 五個 SF 語意維度的定義（High_Arousal 是情緒喚醒度、High_Novelty 是新奇感…）固定不動
- 變動的只有 NLI 模型收到的 hypothesis 字串（verbalizer）
- 這條 pipeline 是 verbalizer 覆蓋面的語料校準，不是語意定義的修改
- 可重複執行：語料更新後重跑 sf_01 + sf_02，產生新的候選池，再做一輪審閱

---

## 7. Current Implementation Scope and Limitations

### Current Scope

- 支援 5 個預定義 label：`High_Arousal`、`High_Novelty`、`High_Repurchase_Intent`、`Negative_Complaint`、`Advertisement`
- 以 `run_id` 版本管理篩選輪次，支援重複執行（冪等寫入）
- Discriminative score 計算在 DB 端完成（PostgreSQL CTE），Python 負責後處理
- 目前只挖掘 unigram
- 每個 label 獨立執行 mining（不跨 label 聯合計算）
- 每次 mining 結果保留人工審閱欄位（`is_approved`、`review_note`），重跑不覆蓋
- 支援 dry-run 模式，可在不寫入的情況下預覽統計與 top-20 候選詞

### Known Limitations

**1. 只有 unigram**  
目前不支援 bigram / trigram mining。「會再買」、「相見恨晚」這類多字詞組目前是以整體字串形式手動加入 verbalizer，不是被挖掘出來的。

**2. 無自動 prototype sentence composition**  
候選詞是離散的 token，沒有自動組合成 NLI hypothesis 用的描述句型，目前的 verbalizer 仍是以「頓號分隔的詞彙串」為主，而非語意更完整的句子。

**3. Label-specific blacklist 仍不完整**  
目前只有 `High_Novelty` 的 blacklist 有實質內容，其他 4 個 label 的 blacklist 為空集合，需在後續 mining run 中逐步發現並補入。

**4. 品類偏誤無法完全自動處理**  
語料中若某品類（例：保健品）評論密集，導致品類特有詞彙在某些 label 子語料中異常高頻，目前的過濾機制（POS + blacklist）只能處理已知的 artifact 詞，新出現的品類偏誤詞需在 review 階段手動識別並補入 blacklist。

**5. `--exclude-nb` 依賴 CKIP 詞性準確度**  
CKIP 不是 100% 準確，部分品牌名被標為 `Na`，`--exclude-nb` 無法攔截，需靠人工審閱補救。

**6. 不同 run_id 之間無 diff 機制**  
目前沒有工具可以自動比較兩輪 mining 的候選詞變化（新增 / 消失 / 分數大幅變動），需手動查詢。

---

## 8. Appendix: Example Commands

### sf_01 — High-Confidence Filtering

```bash
# Dry-run 單一 label（只看統計，不寫入）
python sf_01_filter_highconf.py \
    --run-id nli-seed-v1 \
    --label High_Novelty \
    --dry-run

# 正式寫入 High_Novelty
python sf_01_filter_highconf.py \
    --run-id nli-seed-v1 \
    --label High_Novelty \
    --no-dry-run

# 批次寫入全部 5 個 label
python sf_01_filter_highconf.py \
    --run-id nli-seed-v1 \
    --all-labels \
    --no-dry-run

# 調整 threshold（僅 single-label 有效）
python sf_01_filter_highconf.py \
    --run-id nli-seed-v1 \
    --label Negative_Complaint \
    --threshold 0.60 \
    --no-dry-run
```

### sf_02 — Candidate Mining

```bash
# Dry-run High_Novelty（預覽 top-20 候選詞）
python sf_02_mine_candidates.py \
    --run-id nli-seed-v1 \
    --label-name High_Novelty \
    --pipeline-version "20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac" \
    --corpus-id 5 \
    --exclude-nb \
    --dry-run

# 正式寫入 High_Novelty
python sf_02_mine_candidates.py \
    --run-id nli-seed-v1 \
    --label-name High_Novelty \
    --pipeline-version "20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac" \
    --corpus-id 5 \
    --exclude-nb \
    --mining-run mine-v2-high-novelty-tf5-df3 \
    --top-k 100 \
    --no-dry-run

# Dry-run High_Arousal
python sf_02_mine_candidates.py \
    --run-id nli-seed-v1 \
    --label-name High_Arousal \
    --pipeline-version "20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac" \
    --corpus-id 5 \
    --exclude-nb \
    --dry-run

# Debug 用：關閉所有 blacklist，看原始分數排名
python sf_02_mine_candidates.py \
    --run-id nli-seed-v1 \
    --label-name High_Novelty \
    --pipeline-version "20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac" \
    --corpus-id 5 \
    --no-generic-blacklist \
    --no-label-blacklist \
    --dry-run
```
