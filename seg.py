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
#print("Initializing drivers ... POS")
#pos_driver = CkipPosTagger(model="bert-base", device=0)
#print("Initializing drivers ... NER")
#ner_driver = CkipNerChunker(model="bert-base", device=0)
#print("Initializing drivers ... all done")

ecode = 'utf-8-sig'
dirPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\preprocess\pre1.csv"
outPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\seg\seg1.csv"
lineArr = [] # 所有留言資料陣列
finalArr = [] # 斷詞最終結果陣列

# 對文章進行斷詞
def do_CKIP_WS(article):
    ws_results = ws_driver([str(article)])
    return ws_results

print(do_CKIP_WS('品質很好值得購買商品也符合期待'))

with open(dirPath, 'r', encoding=ecode) as f:
    tmp_data = f.read()
    lineArr = tmp_data.splitlines()

# 斷詞
# 只取留言斷詞
for line in lineArr:
    tmp_data = line.split(',')
    # 將二維陣列轉成一維陣列
    flattened = [item[0] for item in do_CKIP_WS(tmp_data[4])]
    finalArr.append('，'.join(flattened))

with open(outPath, 'w', encoding=ecode) as f:
    for line in finalArr:
        f.write(line + '\n')
