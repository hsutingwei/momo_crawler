import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from pprint import pprint

ecode = 'utf-8-sig'
dirPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\seg\fin1.csv"
outPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\seg\test.csv"
lineArr = [] # 所有留言(已斷詞)陣列
finalArr = [] # 斷詞最終結果陣列

with open(dirPath, 'r', encoding=ecode) as f:
    tmp_data = f.read()
    tmp_lineArr = tmp_data.splitlines()
    for line in tmp_lineArr:
        lineArr.append(line.split(';'))

print(lineArr)
