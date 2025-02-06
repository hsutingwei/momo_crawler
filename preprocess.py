import os
from opencc import OpenCC

### 此處理主要將商品名稱以及留言繁簡轉換
### 並根據留言ID去重複

ecode = 'utf-8-sig'
cc = OpenCC('s2tw')
dirPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\crawler"
outPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\preprocess\pre1.csv"
lineArr = [] # 暫存所有留言爬蟲的陣列
textObj = {} # 將所有留言根據留言ID去重複
finalArr = [] # 預處理最終結果

for f in os.listdir(dirPath):
    if os.path.isfile(os.path.join(dirPath, f)) and "留言資料" in f:
        with open(dirPath + "/" + f, 'r', encoding=ecode) as f:
            tmp_data = f.read()
            lineArr = tmp_data.split('\n')

for line in lineArr:
    if line is not '':
        text = line.split(',')
        tmp_id = text[5]
        if tmp_id not in textObj:
            text[1] = cc.convert(text[1])
            text[4] = cc.convert(text[4])
            finalArr.append(','.join(text))


with open(outPath, 'w', encoding=ecode) as f:
    for line in finalArr:
        f.write(line + '\n')