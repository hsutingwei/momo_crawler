import os

ecode = 'utf-8-sig'
dirPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\crawler"

for f in os.listdir(dirPath):
    if os.path.isfile(os.path.join(dirPath, f)) and "留言資料" in f:
        with open(dirPath + "/" + f, 'r', encoding=ecode) as f:
            data = f.read()
