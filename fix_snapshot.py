import pandas as pd
import os

# === 參數設定 ===
keyword = '益生菌'
current_time_str = '20250625191529'  # 請填入你當時的時間字串
comment_path = f'crawler/{keyword}_商品留言資料_{current_time_str}.csv'
snapshot_path = f'crawler/{keyword}_商品銷售快照.csv'
ecode = 'utf-8-sig'

# === 讀取留言資料 ===
df_comment = pd.read_csv(comment_path, encoding=ecode)
df_comment.columns = df_comment.columns.str.strip()  # <--- 自動去除欄位名稱空白

# 只保留每個商品一筆（同一擷取時間）
df_snapshot = df_comment.drop_duplicates(subset=['商品ID', '資料擷取時間'])[
    ['商品ID', '商品名稱', '價格', '銷售數量', '商品連結', '資料擷取時間']
]
df_snapshot = df_snapshot.rename(columns={'資料擷取時間': '擷取時間'})

# === 讀取現有快照，避免重複 ===
if os.path.exists(snapshot_path):
    df_existing = pd.read_csv(snapshot_path, encoding=ecode, dtype={'商品ID': str})
    df_existing.columns = df_existing.columns.str.strip()  # <--- 這裡也建議加
    existing = set(zip(df_existing['商品ID'], df_existing['擷取時間']))
else:
    existing = set()

# === 只補寫還沒寫進去的資料 ===
to_append = df_snapshot[
    ~df_snapshot.apply(lambda row: (str(row['商品ID']), str(row['擷取時間'])) in existing, axis=1)
]

if not to_append.empty:
    # 寫入前自動備份
    if os.path.exists(snapshot_path):
        import shutil
        shutil.copy(snapshot_path, snapshot_path + '.bak')
    to_append.to_csv(snapshot_path, mode='a', encoding=ecode, index=False, header=False)
    print(f"補寫完成，共補 {len(to_append)} 筆。")
else:
    print("沒有需要補寫的資料。")