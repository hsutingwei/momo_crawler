# -*- coding: utf-8 -*-
"""
修正商品留言檔案中的「銷售數量」欄位，
從快照檔抓出擷取時間 >= 留言檔案時間的對應值，並覆寫留言檔。
同時，以留言檔的 timestamp 為主，將快照檔新增或覆寫對應記錄。
也修復因換行錯誤造成的斷行問題。
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
    with open(input_path, 'r', encoding=ecode) as infile, \
         open(output_path, 'w', encoding=ecode) as outfile:
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
    # 1) 讀取並預處理快照檔 (原始)
    df_snap_raw = pd.read_csv(SNAPSHOT_PATH, encoding=ecode)
    df_snap = df_snap_raw.copy()
    # 轉 Timestamp 與整數
    df_snap['擷取時間_dt'] = pd.to_datetime(df_snap['擷取時間'], format='%Y%m%d%H%M%S')
    df_snap['銷售數量_num'] = (
        df_snap['銷售數量'].astype(str)
               .str.replace(',', '')
               .str.replace(r'\.0+$', '', regex=True)
    )
    df_snap['銷售數量_num'] = pd.to_numeric(
        df_snap['銷售數量_num'], errors='coerce'
    ).fillna(0).astype(int)

    # 2) 處理每個留言檔案
    for fname in os.listdir(CRAWLER_DIR):
        if not fname.startswith(f"{KEYWORD}_商品留言資料_") or not fname.endswith('.csv'):
            continue
        m = re.search(r'_(\d{14})\.csv$', fname)
        if not m:
            continue
        ts_str = m.group(1)
        file_ts = datetime.strptime(ts_str, '%Y%m%d%H%M%S')

        comment_path = os.path.join(CRAWLER_DIR, fname)
        fixed_path   = comment_path + '.fixed'

        # 2.1 修復斷行
        clean_broken_csv(comment_path, fixed_path)

        # 2.2 讀取並標準化
        df_com = pd.read_csv(fixed_path, encoding=ecode)
        df_com.columns = df_com.columns.str.strip()
        # 修正欄位名
        if '  商品ID' in df_com.columns:
            df_com.rename(columns={'  商品ID':'商品ID'}, inplace=True)
        if ' 商品ID' in df_com.columns:
            df_com.rename(columns={' 商品ID':'商品ID'}, inplace=True)
        if '商品ID' not in df_com.columns:
            print(f"跳過 {fname}：找不到商品ID 欄位")
            os.remove(fixed_path)
            continue

        # 2.3 建立映射：商品ID -> 選時快照銷售數量
        mapping = {}
        for pid in df_com['商品ID'].astype(str).unique():
            sel = df_snap[(df_snap['商品ID'].astype(str)==pid)
                         & (df_snap['擷取時間_dt']>=file_ts)]
            if sel.empty:
                sel = df_snap[df_snap['商品ID'].astype(str)==pid]
            if sel.empty:
                mapping[pid] = None
            else:
                mapping[pid] = int(
                    sel.sort_values('擷取時間_dt')
                       .iloc[0]['銷售數量_num']
                )

        # 2.4 覆寫留言檔中的 銷售數量
        df_com['銷售數量'] = (
            df_com['商品ID'].astype(str)
                  .map(mapping)
                  .fillna(0)
                  .astype(int)
        )
        df_com.to_csv(comment_path, index=False, encoding=ecode)
        os.remove(fixed_path)
        print(f"✔ 已更新留言檔: {fname}")

        # 2.5 同步更新快照檔 (df_snap_raw)
        for pid, sale_num in mapping.items():
            if sale_num is None:
                continue
            mask = (
                df_snap_raw['商品ID'].astype(str)==pid
            ) & (df_snap_raw['擷取時間']==ts_str)
            if mask.any():
                df_snap_raw.loc[mask, '銷售數量'] = str(sale_num)
            else:
                # 從 df_com 取一行作為範本
                row = df_com[df_com['商品ID'].astype(str)==pid].iloc[0]
                new = {
                    '商品ID': pid,
                    '商品名稱': row.get('商品名稱',''),
                    '價格':      row.get('價格',''),
                    '銷售數量': str(sale_num),
                    '商品連結': row.get('商品連結',''),
                    '擷取時間': ts_str
                }
                # 使用 pd.concat 取代已移除的 append
                # 直接在末尾新增一行，保留欄位順序
                df_snap_raw.loc[len(df_snap_raw)] = [
                    new['商品ID'],
                    new['商品名稱'],
                    new['價格'],
                    new['銷售數量'],
                    new['商品連結'],
                    new['擷取時間']
                ]

    # 3) 寫回快照檔
    df_snap_raw.to_csv(SNAPSHOT_PATH, index=False, encoding=ecode)
    print(f"✔ 已覆寫快照檔：{SNAPSHOT_PATH}")

if __name__ == '__main__':
    update_comment_sales()