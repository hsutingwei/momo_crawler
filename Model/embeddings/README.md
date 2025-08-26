# Embedding Feature Generation Pipeline

本目錄包含用於產生 embedding 特徵的管線，支援兩種模式：

1. **句向量模式**：使用 `sentence-transformers` 直接對整句進行編碼
2. **詞向量模式**：使用 `fastText` 或 `word2vec` 對斷詞結果進行 SIF 加權平均

## 安裝依賴

### 基本依賴
```bash
pip install numpy pandas scipy scikit-learn psycopg2-binary python-dotenv
```

### 句向量模式依賴
```bash
pip install sentence-transformers torch
```

### 詞向量模式依賴
```bash
# 選擇其中一種
pip install fasttext  # 或
pip install gensim
```

## 快速開始

### 1. 句向量模式（推薦）

使用 `paraphrase-multilingual-MiniLM-L12-v2` 模型對正規化文本進行編碼：

```bash
python emb_build.py \
  --mode comment_level \
  --text-source norm \
  --embed-model paraphrase-multilingual-MiniLM-L12-v2 \
  --batch-size 64 --device cuda --fp16 \
  --date-cutoff 2025-06-25 \
  --pipeline-version 20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac \
  --outdir Model/embeddings/outputs \
  --run-name sent_mMiniLM_norm_20250825 \
  --limit 5000
```

### 2. 詞向量模式

使用 `fastText` 和 SIF 加權：

```bash
python emb_build.py \
  --mode comment_level \
  --word-emb fasttext \
  --word-emb-path ./assets/cc.zh.300.bin \
  --idf-source db --sif-alpha 1e-3 --sif-remove-pc 1 \
  --date-cutoff 2025-06-25 \
  --pipeline-version 20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac \
  --outdir Model/embeddings/outputs \
  --run-name ckip_fasttext_sif_20250825 \
  --limit 5000
```

## 參數說明

### 基本參數
- `--mode`: 聚合模式 (`product_level` | `comment_level`)
- `--date-cutoff`: 日期截止點 (YYYY-MM-DD)
- `--pipeline-version`: 正規化文本的 pipeline 版本
- `--limit`: 限制樣本數量（測試用）

### 文本來源參數
- `--text-source`: 文本來源 (`raw` | `norm`)

### 句向量參數（Pipeline A）
- `--embed-model`: sentence transformer 模型名稱
- `--pooling`: 池化方法 (`mean` | `cls` | `max`)

### 詞向量參數（Pipeline B）
- `--word-emb`: 詞向量類型 (`fasttext` | `word2vec`)
- `--word-emb-path`: 詞向量模型路徑
- `--idf-source`: IDF 來源 (`db` | `none`)
- `--sif-alpha`: SIF alpha 參數
- `--sif-remove-pc`: 移除主成分數量

### 處理參數
- `--batch-size`: 批次大小
- `--device`: 處理裝置 (`cuda` | `cpu` | `auto`)
- `--fp16`: 啟用半精度（句向量模式）

### 輸出參數
- `--outdir`: 輸出目錄
- `--run-name`: 自訂執行名稱
- `--with-y`: 同時產生 y 標籤

## 輸出檔案

每次執行會在 `outputs/<run_name>/` 目錄下產生：

1. `manifest.json`: 執行參數和統計資訊
2. `meta.csv`: 樣本元資料（row_ix, comment_id, product_id, capture_time, n_tokens）
3. `X_embed.npz`: 嵌入向量矩陣（N × dim）
4. `sample_preview.csv`: 前 20 筆樣本預覽

## 範例輸出

### manifest.json
```json
{
  "run_name": "sent_mMiniLM_norm_20250825",
  "started_at": "2025-08-25T10:21:00+08:00",
  "finished_at": "2025-08-25T10:25:32+08:00",
  "device": "cuda",
  "mode": "comment_level",
  "date_cutoff": "2025-06-25",
  "pipeline_version": "20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac",
  "text_source": "norm",
  "embed_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "pooling": "mean",
  "n_samples": 5000,
  "dim": 384,
  "batch_size": 64,
  "files": {
    "meta": "meta.csv",
    "X_embed": "X_embed.npz",
    "preview": "sample_preview.csv"
  }
}
```

### 終端輸出範例
```
==================================================
EMBEDDING GENERATION SUMMARY
==================================================
Samples processed: 5,000
Vector dimension: 384
Processing time: 45.32s
Speed: 110.3 samples/sec
Vector statistics:
  Mean: 0.0123
  Std:  0.9876
  Min:  -3.4567
  Max:  4.1234
==================================================
```

## 注意事項

1. **資料庫連線**：使用與 `csv_to_db.py` 相同的環境變數設定
2. **記憶體使用**：大資料集建議使用 `--limit` 先測試
3. **GPU 加速**：句向量模式支援 CUDA 加速，建議使用 `--fp16` 節省記憶體
4. **詞向量模型**：需要預先下載 fastText 或 word2vec 模型檔案

## 故障排除

### 常見錯誤

1. **ImportError: sentence-transformers not installed**
   ```bash
   pip install sentence-transformers
   ```

2. **ImportError: fasttext not installed**
   ```bash
   pip install fasttext
   ```

3. **ValueError: pipeline_version required**
   - 使用 `--text-source norm` 時必須提供 `--pipeline-version`

4. **CUDA out of memory**
   - 減少 `--batch-size`
   - 使用 `--fp16` 啟用半精度
   - 使用 `--device cpu` 切換到 CPU

### 效能優化

1. **GPU 記憶體不足**：
   - 減少 batch_size（預設 32）
   - 啟用 fp16
   - 使用較小的模型

2. **處理速度慢**：
   - 使用 GPU（`--device cuda`）
   - 增加 batch_size（在記憶體允許範圍內）
   - 使用 `--limit` 先測試小樣本

## 進階用法

### 自訂模型

可以替換不同的 sentence transformer 模型：

```bash
# 使用中文專用模型
--embed-model distiluse-base-multilingual-cased-v2

# 使用更快的模型
--embed-model all-MiniLM-L6-v2
```

### 批次處理

可以寫腳本批次處理多個設定：

```bash
#!/bin/bash
for version in "20250813_xxx" "20250814_yyy"; do
  python emb_build.py \
    --pipeline-version $version \
    --run-name "batch_$version" \
    --limit 10000
done
```
