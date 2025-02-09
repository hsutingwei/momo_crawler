import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib as plt
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import torch

# Check CUDA、cuDNN version
#print(torch.version.cuda)
#print(torch.backends.cudnn.version())

# Initialize drivers
print("Initializing drivers ... WS")
ws_driver = CkipWordSegmenter(model="bert-base", device=0)
print("Initializing drivers ... POS")
pos_driver = CkipPosTagger(model="bert-base", device=0)
#print("Initializing drivers ... NER")
#ner_driver = CkipNerChunker(model="bert-base", device=0)
#print("Initializing drivers ... all done")

ecode = 'utf-8-sig'
key = 'chanel香水'
dirPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\preprocess\pre_" + key + ".csv"
outPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\seg\seg_" + key + ".csv"
posPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\seg\pos_" + key + ".csv"
finPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\seg\fin_" + key + ".csv"
lineArr = [] # 所有留言資料陣列
segArr = [] # 斷詞最終結果陣列
posArr = [] # 詞性結果陣列
clearArr = [] # 最終清理的陣列

def do_CKIP_WS(article):
    """
    對文章進行斷詞

    :param str article: 欲斷詞的字串
    :return string[][]: 斷詞後的結果(二維陣列)
    """
    ws_results = ws_driver([str(article)])
    return ws_results

def do_CKIP_POS(ws_result):
    """
    對詞組進行詞性標示

    :param string[][] ws_result: do_CKIP_WS 的斷詞結果(二維陣列)
    """
    if len(ws_result) == 0:
        return []
    pos = pos_driver(ws_result[0])
    all_list = []
    for sent in pos:
        all_list.append(sent)

    return all_list

def pos_filter(pos):
    """
    保留名詞與動詞
    (do_CKIP_POS 回傳的詞性標示)

    """
    for i in list(set(pos)):
        if i.startswith("N") or i.startswith("V"):
            return "Yes"
        else:
            continue

def cleaner(ws_results, pos_results):
    """
    執行資料清洗
    """
    word_lst = []
    if len(ws_results) == 0 or len(pos_results) == 0:
        return []
    for ws, pos in zip(ws_results[0], pos_results):
        # in_stopwords_or_not = ws not in stopwords  #詞組是否存為停用詞
        if_len_greater_than_1 = len(ws) > 1        #詞組長度必須大於1
        is_V_or_N = pos_filter(pos)                #詞組是否為名詞、動詞
        if if_len_greater_than_1 and is_V_or_N == "Yes":
            word_lst.append(str(ws))
        else:
            pass
    return word_lst

with open(dirPath, 'r', encoding=ecode) as f:
    tmp_data = f.read()
    lineArr = tmp_data.splitlines()

# 斷詞
# 只取留言斷詞
for line in lineArr:
    tmp_data = line.split(',')
    tmp_segArr = [] if tmp_data[4] == '' else do_CKIP_WS(tmp_data[4])
    tmp_posArr = [] if tmp_segArr == '' else do_CKIP_POS(tmp_segArr)
    # 將多維陣列轉成一維陣列
    flattened = [i for item in tmp_segArr for i in item]
    flattened2 = [','.join(sub_array) for sub_array in tmp_posArr]
    segArr.append(';'.join(flattened)) # 斷詞
    posArr.append(';'.join(flattened2)) # 詞性標示
    tmp_finArr = [] if tmp_posArr == '' else cleaner(tmp_segArr, tmp_posArr)
    clearArr.append(';'.join(tmp_finArr)) # 詞句清理

with open(outPath, 'w', encoding=ecode) as f:
    for line in segArr:
        f.write(line + '\n')

with open(posPath, 'w', encoding=ecode) as f:
    for line in posArr:
        f.write(line + '\n')

with open(finPath, 'w', encoding=ecode) as f:
    for line in clearArr:
        f.write(line + '\n')
