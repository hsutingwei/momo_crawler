# CSV to Database Converter

將爬蟲產生的CSV檔案轉換並儲存到MySQL資料庫的程式。

## 功能特色

- **資料驗證**: 嚴格驗證CSV資料格式，確保資料庫資料品質
- **錯誤追蹤**: 詳細記錄所有驗證錯誤，便於後續修正
- **去重機制**: 避免重複插入相同資料
- **批次處理**: 支援大量資料的批次插入，提升效能
- **錯誤報告**: 自動生成詳細的錯誤統計報告
- **檔案管理**: 支援移動已處理的檔案到指定目錄

## 資料庫結構

### 主要資料表

1. **products** - 商品基本資料
2. **sales_snapshots** - 銷售快照資料
3. **product_comments** - 商品評論資料
4. **data_validation_errors** - 資料驗證錯誤記錄
5. **file_sync_logs** - 檔案同步記錄

### 資料型別優化

- 價格: `DECIMAL(10,2)` (數值型別)
- 銷售數量: `INT` (整數型別)
- 評論時間: `TIMESTAMP` (時間戳記)
- 評分: `DECIMAL(2,1)` (小數型別)

## 安裝與設定

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 設定資料庫連接

設定環境變數或在 `config/database.py` 中修改：

```bash
export DB_HOST=localhost
export DB_PORT=3306
export DB_USER=your_username
export DB_PASSWORD=your_password
export DB_NAME=momo_crawler
```

### 3. 初始化資料庫

```bash
python csv_to_db.py --init-db
```

## 使用方法

### 基本用法

```bash
# 處理所有CSV檔案
python csv_to_db.py

# 處理指定關鍵字的檔案
python csv_to_db.py --keyword 口罩

# 處理指定目錄的檔案
python csv_to_db.py --base-dir /path/to/csv/files

# 移動已處理的檔案
python csv_to_db.py --move-processed

# 不生成錯誤報告
python csv_to_db.py --no-report
```

### 進階用法

```bash
# 組合使用
python csv_to_db.py --keyword 口罩 --move-processed --base-dir ./data
```

## 程式架構

```
momo_crawler-main/
├── csv_to_db.py              # 主程式
├── config/
│   └── database.py           # 資料庫設定
├── utils/
│   ├── data_validator.py     # 資料驗證器
│   ├── error_logger.py       # 錯誤記錄器
│   ├── db_operations.py      # 資料庫操作
│   └── file_utils.py         # 檔案處理工具
├── logs/
│   └── csv_to_db.log         # 程式日誌
├── backup/                   # 檔案備份目錄
├── processed/                # 已處理檔案目錄
└── error_report_*.txt        # 錯誤報告檔案
```

## 資料驗證規則

### 商品資料 (products)
- 商品ID: 必須為正整數
- 商品名稱: 最大500字元
- 價格: 支援千分位符號，轉換為數值
- 商品連結: 必須為有效URL

### 銷售快照 (sales_snapshots)
- 商品ID: 必須為正整數
- 銷售數量: 支援"萬"單位，自動轉換
- 擷取時間: 支援多種日期格式
- 去重: 商品ID + 擷取時間組合唯一

### 商品評論 (product_comments)
- 評論ID: 唯一識別碼
- 評論內容: 自動清理換行和逗號
- 評論時間: 支援多種日期格式
- 評分: 0-5分範圍
- 布林值: 支援多種格式 (true/false, 1/0, 是/否)

## 錯誤處理

### 錯誤類型
- `format_error`: 資料格式錯誤
- `null_value`: 空值錯誤
- `invalid_range`: 數值範圍錯誤
- `duplicate`: 重複資料錯誤

### 錯誤報告
程式會自動生成錯誤報告，包含：
- 總體統計
- 錯誤類型統計
- 欄位錯誤統計
- 同步狀態統計

## 日誌記錄

程式會記錄詳細的執行日誌：
- 檔案處理進度
- 資料驗證結果
- 資料庫操作狀態
- 錯誤詳細資訊

日誌檔案: `csv_to_db.log`

## 效能優化

- **批次插入**: 使用 `executemany` 進行批次操作
- **索引優化**: 針對常用查詢建立索引
- **去重檢查**: 預先載入已存在資料，避免重複處理
- **記憶體管理**: 分批處理大量資料

## 注意事項

1. **資料庫權限**: 確保資料庫使用者有建立表和插入資料的權限
2. **檔案編碼**: 預設使用 UTF-8 編碼，支援自動偵測其他編碼
3. **磁碟空間**: 確保有足夠空間存放備份和已處理檔案
4. **網路連接**: 確保資料庫連接穩定

## 故障排除

### 常見問題

1. **資料庫連接失敗**
   - 檢查資料庫服務是否啟動
   - 確認連接參數是否正確
   - 檢查防火牆設定

2. **編碼錯誤**
   - 程式會自動嘗試多種編碼格式
   - 檢查CSV檔案是否損壞

3. **記憶體不足**
   - 減少批次處理大小
   - 分批處理大量檔案

4. **重複資料**
   - 檢查去重邏輯是否正確
   - 確認唯一索引設定

## 開發者資訊

- 支援的資料庫: MySQL 5.7+, MariaDB 10.2+
- Python版本: 3.7+
- 授權: MIT License 