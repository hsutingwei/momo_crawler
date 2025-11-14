# 資料庫檢查工具使用說明

## 概述

這個工具可以幫助你檢查 PostgreSQL 資料庫中的數據，包括商品資料、銷售快照、評論等。

## 使用方法

### 方法一：使用批次檔案 (推薦)

直接執行 `check_db.bat`，會出現選單讓你選擇要檢查的項目：

```bash
check_db.bat
```

### 方法二：直接使用 Python 指令

```bash
# 顯示所有表的基本資訊
python check_db.py --table-info

# 顯示關鍵字統計
python check_db.py --keywords

# 顯示商品資料 (前10筆)
python check_db.py --products

# 顯示商品資料 (前20筆，指定關鍵字)
python check_db.py --products --limit 20 --keyword "益生菌"

# 顯示銷售快照資料
python check_db.py --snapshots

# 顯示評論資料
python check_db.py --comments

# 顯示檔案同步狀態
python check_db.py --sync-status

# 顯示資料驗證錯誤
python check_db.py --errors

# 顯示所有資訊
python check_db.py --all
```

## 可用參數

- `--table-info`: 顯示所有表的基本資訊和記錄數
- `--keywords`: 顯示所有關鍵字及其統計資訊
- `--products`: 顯示商品資料
- `--snapshots`: 顯示銷售快照資料
- `--comments`: 顯示評論資料
- `--sync-status`: 顯示檔案同步狀態
- `--errors`: 顯示資料驗證錯誤
- `--all`: 顯示所有資訊
- `--keyword KEY`: 指定關鍵字篩選
- `--limit N`: 設定顯示記錄數限制 (預設: 10)

## 資料庫表結構

### products (商品資料)
- `id`: 商品ID (主鍵)
- `name`: 商品名稱
- `price`: 價格
- `product_link`: 商品連結
- `keyword`: 關鍵字
- `created_at`: 建立時間

### sales_snapshots (銷售快照)
- `id`: 快照ID (主鍵)
- `product_id`: 商品ID (外鍵)
- `sales_count`: 銷售數量
- `sales_unit`: 銷售單位
- `capture_time`: 擷取時間
- `created_at`: 建立時間

### product_comments (商品評論)
- `id`: 評論ID (主鍵)
- `comment_id`: 留言ID (唯一)
- `product_id`: 商品ID (外鍵)
- `comment_text`: 評論內容
- `customer_name`: 客戶名稱
- `comment_date`: 評論時間
- `score`: 評分
- `capture_time`: 擷取時間

### data_validation_errors (資料驗證錯誤)
- `file_path`: 檔案路徑
- `keyword`: 關鍵字
- `data_type`: 資料類型
- `field_name`: 欄位名稱
- `error_type`: 錯誤類型
- `error_message`: 錯誤訊息

### file_sync_logs (檔案同步記錄)
- `keyword`: 關鍵字
- `file_type`: 檔案類型
- `file_path`: 檔案路徑
- `sync_status`: 同步狀態
- `total_records`: 總記錄數
- `valid_records`: 有效記錄數
- `error_records`: 錯誤記錄數

## 常見問題

### Q: 如何檢查特定關鍵字的資料？
A: 使用 `--keyword` 參數，例如：
```bash
python check_db.py --products --keyword "益生菌"
```

### Q: 如何顯示更多記錄？
A: 使用 `--limit` 參數，例如：
```bash
python check_db.py --products --limit 50
```

### Q: 如何檢查資料匯入是否成功？
A: 使用 `--sync-status` 查看檔案同步狀態：
```bash
python check_db.py --sync-status
```

### Q: 如何檢查資料驗證錯誤？
A: 使用 `--errors` 查看驗證錯誤：
```bash
python check_db.py --errors
```

## 環境設定

確保你的 `.env` 檔案包含正確的資料庫連接資訊：

```
DB_HOST=your_host
DB_PORT=5432
DB_USER=your_user
DB_PASSWORD=your_password
DB_NAME=your_database
```

## 注意事項

1. 確保資料庫連接正常
2. 大量資料查詢時建議使用 `--limit` 參數限制顯示數量
3. 如果遇到連接問題，檢查 `.env` 檔案設定
4. 工具會自動處理資料庫連接的開啟和關閉 