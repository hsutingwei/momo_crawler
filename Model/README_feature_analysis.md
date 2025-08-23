# 特徵分析工具使用說明

## 概述

`feature_analysis.py` 是一個專門用於驗證特徵是否足以區分類別的工具。它提供兩種主要的分析方法：

1. **Dense 特徵分佈分析**：分析每個 dense 特徵在 y=0 vs y=1 之間的分佈差異
2. **特徵空間可視化**：使用 PCA/t-SNE/UMAP 將高維特徵投影到 2D 空間進行可視化

## 功能特點

### ✅ 動態數據載入
- 與 `train.py` 使用完全相同的數據載入方式
- 支援 `product_level` 和 `comment_level` 兩種模式
- 自動適應未來的數據載入方式變化

### ✅ 多維度分析
- **Cohen's d**：標準化平均差，衡量兩組數據的分離程度
- **互信息 (Mutual Information)**：衡量特徵與目標變數的相關性
- **Mann-Whitney U 檢定**：非參數統計檢定
- **重疊係數**：衡量兩個分佈的重疊程度

### ✅ 可視化分析
- **PCA**：主成分分析，線性降維
- **t-SNE**：t-分佈隨機鄰域嵌入，非線性降維
- **UMAP**：統一流形近似和投影，現代降維方法

### ✅ 智能結論生成
- 自動計算特徵充分性評分 (0-100)
- 根據分析結果生成具體建議
- 為未來的資料庫儲存準備標準化格式

## 使用方法

### 基本用法

```bash
python Model/feature_analysis.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --vocab-mode global \
  --top-n 100 \
  --pipeline-version v1 \
  --exclude-products 8918452 \
  --outdir Model/analysis_outputs
```

### 保存可視化圖表

```bash
python Model/feature_analysis.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --vocab-mode global \
  --top-n 100 \
  --save-plots \
  --plot-format png \
  --outdir Model/analysis_outputs
```

### 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--mode` | 分析模式：product_level 或 comment_level | product_level |
| `--date-cutoff` | 數據截止日期 | 2025-06-25 |
| `--pipeline-version` | 數據處理管道版本 | 環境變數 PIPELINE_VERSION |
| `--vocab-mode` | 詞彙模式 | global |
| `--top-n` | TF-IDF 特徵數量 | 100 |
| `--exclude-products` | 排除的商品 ID | 8918452 |
| `--keyword` | 單一關鍵字分析 | None |
| `--outdir` | 輸出目錄 | Model/analysis_outputs |
| `--save-plots` | 是否保存可視化圖表 | False |
| `--plot-format` | 圖表格式：png/pdf/svg | png |

## 輸出結果

### 1. 控制台輸出

```
=== 特徵分析工具 ===
分析 ID: 12345678-1234-1234-1234-123456789abc

數據摘要:
  總樣本數: 7196
  正樣本數: 423
  負樣本數: 6773
  不平衡比例: 16.01:1
  Dense 特徵數: 10
  TF-IDF 特徵數: 100

=== Dense 特徵分佈分析 ===
price:
  Cohen's d: 0.199
  Mutual Info: 0.007
  p-value: 8.628e-02
  重疊係數: 0.809

總結:
  總特徵數: 10
  顯著差異特徵: 6 (60.0%)
  高分離特徵: 6 (60.0%)

=== 特徵空間可視化分析 ===
分離度評分:
  PCA: 0.478
  t-SNE: 1.710
  UMAP: 2.110

=== 分析完成 ===
特徵充分性評分: 64.0/100
建議:
  1. 特徵充分性中等，可以嘗試調整模型參數或使用集成方法
```

### 2. 檔案輸出

#### 詳細結果檔案
- `feature_analysis_detailed_YYYYMMDD_HHMMSS.json`
- 包含完整的分析結果，包括每個特徵的詳細統計

#### 資料庫格式檔案
- `feature_analysis_db_ready_YYYYMMDD_HHMMSS.json`
- 標準化格式，適合未來儲存到資料庫

#### 可視化圖表（可選）
- `feature_visualization_YYYYMMDD_HHMMSS.png`
- 包含 PCA、t-SNE、UMAP 三種降維方法的可視化

## 評分標準

### 特徵充分性評分 (0-100)

| 評分範圍 | 等級 | 說明 |
|----------|------|------|
| 0-30 | 低 | 特徵不足以區分類別，需要重新設計 |
| 31-50 | 較低 | 特徵分離度有限，建議增加特徵 |
| 51-70 | 中等 | 特徵有一定區分能力，可嘗試模型優化 |
| 71-85 | 良好 | 特徵品質較好，適合進一步優化 |
| 86-100 | 優秀 | 特徵品質很好，可以專注於模型調優 |

### 評分組成

1. **特徵分離度 (40%)**：基於 Cohen's d 和互信息的高分離特徵比例
2. **可視化分離度 (30%)**：基於 PCA/t-SNE/UMAP 的分離度評分
3. **數據品質 (30%)**：基於樣本數、特徵數量的綜合評分

## 資料庫整合

### 未來資料庫 Schema 建議

```sql
-- 特徵分析結果表
CREATE TABLE feature_analysis_results (
    analysis_id UUID PRIMARY KEY,
    analysis_timestamp TIMESTAMP,
    run_id UUID REFERENCES ml_runs(run_id),
    analysis_config JSONB,
    data_summary JSONB,
    feature_analysis JSONB,
    visualization_analysis JSONB,
    conclusions JSONB,
    artifacts JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 特徵分離度指標表
CREATE TABLE feature_separation_metrics (
    analysis_id UUID REFERENCES feature_analysis_results(analysis_id),
    feature_name VARCHAR(200),
    cohens_d NUMERIC(10,6),
    mutual_info NUMERIC(10,6),
    p_value NUMERIC(10,6),
    overlap_coefficient NUMERIC(10,6),
    is_significant BOOLEAN,
    has_high_separation BOOLEAN,
    PRIMARY KEY (analysis_id, feature_name)
);
```

### 資料庫格式範例

```json
{
  "analysis_id": "12345678-1234-1234-1234-123456789abc",
  "analysis_timestamp": "2025-08-23T14:18:35.123456",
  "analysis_config": {
    "mode": "product_level",
    "date_cutoff": "2025-06-25",
    "vocab_mode": "global",
    "top_n": "100"
  },
  "data_summary": {
    "total_samples": 7196,
    "positive_samples": 423,
    "negative_samples": 6773,
    "imbalance_ratio": 16.01,
    "dense_features_count": 10,
    "tfidf_features_count": 100
  },
  "feature_analysis": {
    "summary_stats": {
      "total_features": 10,
      "features_with_significant_diff": 6,
      "features_with_high_separation": 6
    },
    "feature_analysis": [...]
  },
  "visualization_analysis": {
    "visualization_results": {
      "pca": {"separation_score": 0.478},
      "tsne": {"separation_score": 1.710},
      "umap": {"separation_score": 2.110}
    }
  },
  "conclusions": {
    "feature_sufficiency_score": 64.0,
    "recommendations": [
      "特徵充分性中等，可以嘗試調整模型參數或使用集成方法"
    ]
  }
}
```

## 常見問題

### Q: 為什麼不平衡處理沒有效果？
A: 使用此工具分析特徵分離度。如果特徵本身不足以區分類別，不平衡處理的效果會很有限。

### Q: 如何判斷特徵是否足夠？
A: 查看以下指標：
- Cohen's d > 0.5 或 Mutual Info > 0.1 的特徵比例
- 可視化分離度評分
- 整體特徵充分性評分

### Q: 如何改進特徵品質？
A: 根據建議：
- 增加更多相關特徵
- 進行特徵工程
- 使用更複雜的模型
- 調整模型參數

## 依賴套件

確保安裝以下套件：
```bash
pip install seaborn matplotlib umap-learn imbalanced-learn
```

或使用 requirements.txt：
```bash
pip install -r requirements.txt
```
