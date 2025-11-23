# Phase 2：Label Schema 設計與模擬分析（Product-Level, N=7197）

## 1. 前提說明

- 資料總筆數：**N = 7197（以 product 為單位）**
- 已有欄位：
  - `max_raw_delta`：cutoff 後每個 product 的最大銷售量絕對成長（本次 - 上次）。
  - `max_raw_ratio`：cutoff 後每個 product 的最大相對成長率（sales / prev_sales - 1）。
- 基礎族群：
  - `delta >= 10` 的樣本共有 **464 筆**（與 baseline positive 數量一致）。

以下三套 Label Schema 都建立在這個前提之上。

---

## 2. Schema A：Strict Ratio Levels（嚴格比例分級）

### 2.1 設計邏輯

- 目標：將「有成長（delta >= 10）」的樣本，依 ratio 切成中度與高度成長。
- 規則（在 `max_raw_delta >= 10 且 max_raw_ratio >= 0.1` 的前提下）：
  - Class 0：無成長（不符合上述條件）
  - Class 1：**中度成長**：`0.1 <= ratio < 0.3`
  - Class 2：**高度成長**：`ratio >= 0.3`

### 2.2 模擬分布（N=7197）

- Class 0（無成長）：**6733（93.55%）**
- Class 1（中度成長）：**27（0.38%）**
- Class 2（高度成長）：**437（6.07%）**

### 2.3 評估與註解

- Class 1 的 27 筆樣本，即先前在 Hybrid Strict 中被剔除的高銷量商品：
  - 高基數、delta 很大、ratio ≈ 0.25。
- **優點**：
  - 能清楚分離出「高基數但相對成長較低」的特例群。
- **缺點**：
  - Class 1 數量過少（0.38%），單獨建模難度高，容易導致 multi-class 模型不穩定。

> 建議用途：  
> - 作為 **診斷與分析 label 邊界** 的輔助 Schema。  
> - 不建議當成主力 multi-class 設計。

---

## 3. Schema B：Growth Quality（成長品質：Volume vs Efficiency）

### 3.1 設計邏輯

- 目標：區分「量大但成長率低」與「效率高」的成長。
- 規則：
  - Class 0：無成長：`max_raw_delta < 10`
  - Class 1：**Volume Growth**：「穩定大品項」  
    `max_raw_delta >= 10 AND max_raw_ratio < 0.2`
  - Class 2：**Efficiency Growth**：「成長效率高」  
    `max_raw_delta >= 10 AND max_raw_ratio >= 0.2`

### 3.2 模擬分布（N=7197）

- Class 0（無成長）：**6733（93.55%）**
- Class 1（Volume Growth）：**45（0.63%）**
- Class 2（Efficiency Growth）：**419（5.82%）**

### 3.3 評估與註解

- 相較 Schema A，Class 1 略為增加到 45 筆，但仍然非常少。
- **優點**：
  - 語意上好講：可以對比「穩定巨頭」 vs 「高效率成長」。
- **缺點**：
  - 類別比例仍高度不平衡，Class 1 作為獨立類別風險較高。

> 建議用途：  
> - 可作為 **第二候選 Schema**，或用於補充分析。  
> - 不一定適合成為主力 multi-class 任務。

---

## 4. Schema C：Percentile Based（分位數分級）

### 4.1 設計邏輯

- 基礎族群：`max_raw_delta >= 10` 的 464 筆樣本。
- 在該族群中計算 `max_raw_ratio` 的分位數：
  - P33 ≈ **0.67**
  - P66 ≈ **1.00**
- 規則：
  - Class 0：無成長：`max_raw_delta < 10`
  - Class 1（Low Tier）：`max_raw_ratio < 0.67`
  - Class 2（Mid Tier）：`0.67 <= max_raw_ratio < 1.0`
  - Class 3（Top Tier）：`max_raw_ratio >= 1.0`（銷量翻倍以上）

### 4.2 模擬分布（N=7197）

- Class 0（無成長）：**6733（93.55%）**
- Class 1（Low Tier）：**149（2.07%）**
- Class 2（Mid Tier）：**55（0.76%）**
- Class 3（Top Tier）：**260（3.61%）**

### 4.3 重要發現

- 在 delta >= 10 的 464 筆成長樣本中，有 **260 筆（約 56%）** 的 `max_raw_ratio >= 1.0`：
  - 可視為「銷量翻倍級」的 **爆品**。
- Class 3 的樣本數 260 足以支撐獨立的爆品預測任務。

### 4.4 評估與註解

- **優勢**：
  - 商業語意非常清楚：
    - Class 3：爆品（銷量翻倍）。
    - Class 1 / 2：一般成長與中度成長。
  - 類別比例相比 A/B 更合理：
    - Class 3 有足夠樣本量。
    - Class 1 也有一定規模，可觀察其與 Class 3 在特徵上的差異。
- **潛在挑戰**：
  - Class 2 只有 55 筆，仍偏少，訓練 multi-class 模型時需搭配 class weight 或重抽樣策略。

> 綜合評估：  
> **Schema C 是 Phase 2 最推薦實作的 multi-class 設計**，  
> 並可額外衍生出一個二元任務：  
> - y=1：Class 3（爆品）  
> - y=0：其他（非爆品）

---

## 5. 整合進現有 Pipeline 的設計草案

### 5.1 config/label_experiments.yaml

新增一個多分類實驗配置，例如：

```yaml
- name: "multiclass_percentile"
  strategy: "multiclass"
  params:
    delta_threshold: 10
    # 定義各類別的 ratio 上限（不含），最後一類為無限大
    # Class 1: < 0.67, Class 2: < 1.0, Class 3: >= 1.0
    ratio_buckets: [0.67, 1.0]
