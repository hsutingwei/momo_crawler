# -*- coding: utf-8 -*-
"""
修正商品留言檔案中的「銷售數量」欄位，
從快照檔抓出擷取時間 >= 留言檔案時間的對應值，覆寫原檔。
同時修復因換行錯誤造成的斷行問題。
"""
import os
import re
import pandas as pd
from datetime import datetime

# ---- 參數 ----
KEYWORD = '益生菌'
CRAWLER_DIR = r"C:\YvesProject\中央\線上評論\momo_crawler-main\crawler"
SNAPSHOT_PATH = os.path.join(CRAWLER_DIR, f"{KEYWORD}_商品銷售快照.csv")

ecode = 'utf-8-sig'

# ---- 修復斷行工具 ----
def is_valid_row(line):
    return re.match(r'^\d+,', line.strip()) is not None

def clean_broken_csv(input_path, output_path):
    buffer = ''
    with open(input_path, 'r', encoding=ecode) as infile, open(output_path, 'w', encoding=ecode) as outfile:
        for raw in infile:
            line = raw.rstrip('\n')
            if is_valid_row(line):
                if buffer:
                    outfile.write(buffer + '\n')
                buffer = line
            else:
                buffer += ' ' + line
        if buffer:
            outfile.write(buffer + '\n')

# ---- 主程式 ----

def update_comment_sales():
    # 讀取銷售快照，轉換時間和數量
    df_snap = pd.read_csv(SNAPSHOT_PATH, encoding=ecode)
    df_snap['擷取時間_dt'] = pd.to_datetime(df_snap['擷取時間'], format='%Y%m%d%H%M%S')
    df_snap['銷售數量_num'] = (
        df_snap['銷售數量'].astype(str)
               .str.replace(',','')
               .str.replace(r'\.0+$','', regex=True)
    )
    df_snap['銷售數量_num'] = pd.to_numeric(df_snap['銷售數量_num'], errors='coerce').fillna(0).astype(int)

    # 處理每個留言檔
    for fname in os.listdir(CRAWLER_DIR):
        if not fname.startswith(f"{KEYWORD}_商品留言資料_") or not fname.endswith('.csv'):
            continue
        m = re.search(r'_(\d{14})\.csv$', fname)
        if not m:
            continue
        ts_str = m.group(1)
        file_ts = datetime.strptime(ts_str, '%Y%m%d%H%M%S')

        comment_path = os.path.join(CRAWLER_DIR, fname)
        tmp_fixed = comment_path + '.fixed'

        # 修復換行錯誤
        clean_broken_csv(comment_path, tmp_fixed)

                # 讀取修復後檔案
        df_com = pd.read_csv(tmp_fixed, encoding=ecode)
        # 去除所有欄位名稱前後空白
        df_com.columns = df_com.columns.str.strip()
        # 若欄位名有誤，如 " 商品ID"，先重命名
        if ' 商品ID' in df_com.columns:
            df_com.rename(columns={' 商品ID':'商品ID'}, inplace=True)
        # 確保有商品ID欄位
        if '商品ID' not in df_com.columns:
            print(f"跳過 {fname}：找不到商品ID欄位")
            os.remove(tmp_fixed)
            continue

        # 建立映射：商品ID -> 新銷售數量
        mapping = {}
        unique_ids = df_com['商品ID'].astype(str).unique()
        for pid in unique_ids:
            # 篩選快照：ID / 時間 >= 檔案時間
            df_p = df_snap[(df_snap['商品ID'].astype(str) == pid) & (df_snap['擷取時間_dt'] >= file_ts)]
            if df_p.empty:
                # 若無，退而求其次取最新一筆
                df_p = df_snap[df_snap['商品ID'].astype(str) == pid]
            if df_p.empty:
                mapping[pid] = None
            else:
                # 取最早符合條件的
                val = df_p.sort_values('擷取時間_dt').iloc[0]['銷售數量_num']
                mapping[pid] = val

        # 套用映射覆寫銷售數量欄位
        df_com['銷售數量'] = df_com['商品ID'].astype(str).map(mapping).fillna(0).astype(int)

        # 儲存回原檔
        df_com.to_csv(comment_path, index=False, encoding=ecode)
        os.remove(tmp_fixed)
        print(f"已更新 {fname} 的銷售數量。")

if __name__ == '__main__':
    update_comment_sales()