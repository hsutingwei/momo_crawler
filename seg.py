# -*- coding: utf-8 -*-
"""
對 preprocess 資料夾中多個 pre_<keyword>_<timestamp>.csv 檔案，執行 CKIP 斷詞、POS 標註、過濾，
並統一輸出到 seg/<keyword>/ 資料夾：
    seg/<keyword>/seg_<keyword>_<timestamp>.csv
    seg/<keyword>/pos_<keyword>_<timestamp>.csv
    seg/<keyword>/fin_<keyword>_<timestamp>.csv
"""

import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib as plt
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import torch
import csv
from tqdm import tqdm
import argparse

# Check CUDA、cuDNN version
#print(torch.version.cuda)
#print(torch.backends.cudnn.version())

# ---------- 參數 & 路徑 ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keyword', type=str, help='關鍵字，對應 crawler 檔案名前綴')
    return ap.parse_args()

args = parse_args()

# 參數
keyword = args.keyword or '益生菌'    # 預設可改
ROOT = r"C:\YvesProject\中央\線上評論\momo_crawler-main"
DIR_PRE  = os.path.join(ROOT, 'preprocess')
DIR_SEG  = os.path.join(ROOT, 'seg', keyword)
os.makedirs(DIR_SEG, exist_ok=True)

ecode = 'utf-8-sig'

# Initialize drivers
print("Initializing drivers ... WS")
ws_driver = CkipWordSegmenter(model="bert-base", device=0)
print("Initializing drivers ... POS")
pos_driver = CkipPosTagger(model="bert-base", device=0)
#print("Initializing drivers ... NER")
#ner_driver = CkipNerChunker(model="bert-base", device=0)
#print("Initializing drivers ... all done")

def do_CKIP_WS(text):
    """
    對文章進行斷詞

    :param str text: 欲斷詞的字串
    :return string[][]: 斷詞後的結果(二維陣列)
    """
    # ws_driver 回傳 List[List[str]]，每個子 List 是一句
    results = ws_driver([str(text)])
    return results[0]  # 取第一句的 token 列表

def do_CKIP_POS(tokens):
    """對詞組進行詞性標示"""
    return pos_driver(tokens)

def pos_filter(pos_list):
    """
    保留名詞與動詞
    只保留名詞(N*)和動詞(V*)
    """
    return [p for p in pos_list if p.startswith('N') or p.startswith('V')]

# 處理每個預處理檔
for fname in os.listdir(DIR_PRE):
    if not fname.startswith(f'pre_{keyword}_') or not fname.endswith('.csv'):
        continue
    # 取得 timestamp
    m = re.search(rf'pre_{keyword}_(\d{{14}})\.csv', fname)
    if not m:
        continue
    ts = m.group(1)
    in_path = os.path.join(DIR_PRE, fname)

    seg_out = os.path.join(DIR_SEG, f'seg_{keyword}_{ts}.csv')
    pos_out = os.path.join(DIR_SEG, f'pos_{keyword}_{ts}.csv')
    fin_out = os.path.join(DIR_SEG, f'fin_{keyword}_{ts}.csv')

    with open(in_path, 'r', encoding=ecode, newline='') as fr, \
         open(seg_out, 'w', encoding=ecode, newline='') as fseg, \
         open(pos_out, 'w', encoding=ecode, newline='') as fpos, \
         open(fin_out, 'w', encoding=ecode, newline='') as ffin:
        reader = csv.reader(fr)
        writer_seg = csv.writer(fseg)
        writer_pos = csv.writer(fpos)
        writer_fin = csv.writer(ffin)

        for row in tqdm(reader, desc=f"Processing {fname}"):
            # 留言位於第5欄 (index=5)
            if len(row) <= 5 or not row[5]:
                writer_seg.writerow([])
                writer_pos.writerow([])
                writer_fin.writerow([])
                continue
            text = row[5]

            # CKIP 斷詞 (回傳 List[str])
            ws_tokens = do_CKIP_WS(text)
            writer_seg.writerow(ws_tokens)

            # CKIP POS 標註 -> List[List[str]]，取第一句作為序列
            pos_lists = do_CKIP_POS(ws_tokens)
            pos_tokens = pos_lists[0] if pos_lists else []
            writer_pos.writerow(pos_tokens)

            # 清理：保留名詞(N*)/動詞(V*) 且長度>1
            fin_tokens = [tok for tok, tag in zip(ws_tokens, pos_tokens)
                          if len(tok) > 1 and (tag.startswith("N") or tag.startswith("V"))]
            writer_fin.writerow(fin_tokens)

    print(f"Finished: {fname} -> seg/{keyword}/  (ts={ts})")