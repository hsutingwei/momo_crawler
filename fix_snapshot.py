import pandas as pd
import os
import time
import random
import requests
import re

# === 參數設定 ===
keyword = '益生菌'
product_csv_path = f'{keyword}_商品資料.csv'
snapshot_path = f'crawler/{keyword}_商品銷售快照.csv'
ecode = 'utf-8-sig'
current_time_str = '20250625191529'  # 請填入你當時的時間字串

# === 取得銷售數量的函式（複製自你的 crawler.py）===
def extract_number(input_string):
    match = re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', str(input_string))
    if match:
        number_str = match.group(0).replace(',', '')
        return int(float(number_str))
    return None

def get_current_sales(goods_code, host="momoshop"):
    url = "https://eccapi.momoshop.com.tw/user/getGoodsComment"
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    payload = {
        "goodsCode": str(goods_code),
        "host": host,
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        if data is None or data.get("saleCount") is None:
            print(f"【警告】無法取得 {goods_code} 的銷售數")
            return 0
        sales_text = data.get("saleCount")
        count = extract_number(sales_text)
        if count is None:
            return 0
        return str(count) + ('萬' if sales_text.endswith('萬') else '')
    except Exception as e:
        print(f"HTTP 請求錯誤: {e}")
        return 0

# === 讀取商品資料 ===
df = pd.read_csv(product_csv_path, encoding=ecode)

# === 讀取現有快照，避免重複 ===
if os.path.exists(snapshot_path):
    df_snapshot = pd.read_csv(snapshot_path, encoding=ecode, dtype={'商品ID': str})
    existing = set(zip(df_snapshot['商品ID'], df_snapshot['擷取時間']))
else:
    existing = set()

# === 準備要補寫的資料 ===
all_rows = []
for _, row in df.iterrows():
    key = (str(row['商品ID']), current_time_str)
    if key in existing:
        continue  # 已經有這筆快照就跳過
    product_id = row['商品ID']
    product_name = row['商品名稱']
    price = row['價格']
    link = row['商品連結']
    latest_sales = get_current_sales(product_id)
    all_rows.append([
        product_id,
        product_name,
        price,
        latest_sales,
        link,
        current_time_str
    ])
    time.sleep(random.uniform(1, 2.2))  # 避免被封鎖

# === 補寫進快照檔 ===
if all_rows:
    df_to_append = pd.DataFrame(all_rows, columns=[
        '商品ID', '商品名稱', '價格', '銷售數量', '商品連結', '擷取時間'
    ])
    # 寫入前自動備份
    if os.path.exists(snapshot_path):
        import shutil
        shutil.copy(snapshot_path, snapshot_path + '.bak')
    df_to_append.to_csv(snapshot_path, mode='a', encoding=ecode, index=False, header=False)
    print(f"補寫完成，共補 {len(df_to_append)} 筆。")
else:
    print("沒有需要補寫的資料。")