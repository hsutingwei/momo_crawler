# -*- coding: utf-8 -*-
"""
把原先一次寫一檔（pre_<key>.csv）的流程，改成：
  1. 針對 crawler 資料夾內所有「<keyword>_商品留言資料_YYYYMMDDHHMMSS.csv」
  2. 各自輸出一檔 → preprocess/pre_<keyword>_YYYYMMDDHHMMSS.csv

使用方法：
    python preprocess_keyword_timestamp.py --keyword "益生菌"
若不加參數，預設 key 取程式開頭的變數。
"""

import os
import re
import unicodedata
import argparse
import csv
from opencc import OpenCC

### 此處理主要將商品名稱以及留言繁簡轉換
### 並根據留言ID去重複

# ---------- 參數 & 路徑 ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--keyword', type=str, help='關鍵字，對應 crawler 檔案名前綴')
    return ap.parse_args()

args = parse_args()

keyword = args.keyword or '益生菌'    # 預設可改

ROOT = r"C:\YvesProject\中央\線上評論\momo_crawler-main"
DIR_CRAWLER = os.path.join(ROOT, 'crawler')
DIR_PRE     = os.path.join(ROOT, 'preprocess')
os.makedirs(DIR_PRE, exist_ok=True)

ecode = 'utf-8-sig'
cc = OpenCC('s2tw')

# ---- 修復功能 ----

def is_valid_row(line):
    return re.match(r'^\d+,', line.strip()) is not None

def clean_broken_csv(input_path, output_path):
    """
    修復因留言中斷行或嵌入逗號導致的 CSV 換行錯誤。
    以「行開頭非數字,」判斷是否為斷行，並合併到前一行。
    """
    with open(input_path, 'r', encoding=ecode) as infile, \
         open(output_path, 'w', encoding=ecode) as outfile:
        buffer = ''
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

def do_money_special(arr, start_index=3):
    """
    因爬蟲時沒有正確將價格的千分位符號處理好，導致千分位符號與 csv 衝突，故此 function 專門處理這個問題

    :param list arr: 單行的數據陣列
    :return list: 處理好後將處理完的單行陣列回傳
    """
    # 長度不足就不處理
    if len(arr) <= start_index:
        return arr

    price_token = arr[start_index].replace('"', '')
    end_index = start_index + 1

    for i in range(start_index + 1, len(arr)):
        tok = arr[i].replace('"', '')
        if tok.isdigit() and len(tok) == 3:          # 千分位塊
            price_token += tok
        else:
            end_index = i
            break

    arr[start_index] = price_token
    # 如果真有千分位塊才刪除；避免 list 切超過長度
    if end_index > start_index + 1:
        del arr[start_index + 1:end_index]

    return arr

def do_replace_to_Chinese(str, seg_char = ';'):
    """
    去除非中文符號，把非中文替還成 ';'。
    此動作可保留所有漢字符號，除了繁體字，簡體字、再簡字、三簡字等，亞洲國家的漢字也會保留，例如日文漢字等國家的漢字都會保留

    :param str str: 來源字串
    :param str str: 替換符號
    :return str: 替換後結果
    """
    str = re.sub(r'[^\u3400-\u9fcc\uf900-\ufa2d\u3041-\u3129\uff66-\uff9f]+', seg_char, str)
    str = re.sub(r'[[]-?() \f\n\r\t\v:ο\';×~+&!"_#`『』˙>\/%<β=÷*€ˊιμ$ˇ°χˋˍλ}¯@| {επδθ±§ ^αγ  ·,.:：﹕;；<>《》︽︾＜＞「」［］　『』【】〔〕︹︺︻︼﹁﹂﹃﹄［］﹝﹞＼／﹨∕？﹖、‘’′｜∣∥↖↘↗↙︱︳︴↑↓－¯〈〉?[{]}\|!`~@#$%^&*()-=_+┬┴├─┼┤┌┐╞═╪╡│▕└┘╭╮╰╯╔╦╗╠═╬╣╓╥╖╒╤╕║╚╩╝╟╫╢╙╨╜╞╪╡╘╧╛＿ˍ▁▂▃▄▅▆▇█▏▎▍▌▋▊▉◢', seg_char, str)
    str = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⓪❶❷❸❹❺❻❼❽❾❿⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉１２３４５６７８９０ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵￥￡€￠¥〔〕【】﹝﹞〈〉﹙﹚《》（）｛｝﹛﹜︵︶︷︸︹︺︻︼︽︾︿﹀＜＞∩∪≒≌∵∴×／﹣±≦≧﹤﹥≠≡¼½¾³²∞√㏒∷°÷ˇ㏑∫∮∠∟⊿＋﹢⊥╳⊾℅㎎㎏㎜㎝㎞㎡㏄㏎㏕℃℉‰˙ˊˇˋ\u3105-\u3129]+', seg_char, str)
    str = re.sub(r'[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]+', seg_char, str)
    str = re.sub(r'[' + seg_char + ']+', seg_char, str)

    # 移除字串開頭和結尾的逗號
    if str.startswith(seg_char):
        str = str[1:]
    if str.endswith(seg_char):
        str = str[:-1]

    return str

def full2half(str: str) -> str:
    """
    全形轉半形
    """
    return unicodedata.normalize("NFKC", str)

# ---------- 主流程 ----------

# 遍歷每個留言檔
for fname in os.listdir(DIR_CRAWLER):
    if keyword in fname and '商品留言資料_' in fname and fname.endswith('.csv'):
        m = re.search(r'(\d{14})', fname)
        if not m:
            continue
        ts = m.group(1)
        src_path = os.path.join(DIR_CRAWLER, fname)
        fixed_path = os.path.join(DIR_PRE, f'_fixed_{fname}')
        out_path = os.path.join(DIR_PRE, f'pre_{keyword}_{ts}.csv')

        # 修復原始檔換行錯誤到暫存
        clean_broken_csv(src_path, fixed_path)

        seen_ids = set()
        lines_out = []
        # 讀取修復後暫存檔
        with open(fixed_path, 'r', encoding=ecode, newline='') as fr:
            reader = csv.reader(fr)
            for row in reader:
                if len(row) < 6:
                    continue
                row = do_money_special(row)
                msg_id = row[5]
                if msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)

                row[1] = cc.convert(full2half(row[1]))
                row[4] = cc.convert(full2half(row[4]))
                row[4] = do_replace_to_Chinese(row[4])
                lines_out.append(row)

        # 移除暫存檔
        os.remove(fixed_path)

        # 寫出預處理檔
        with open(out_path, 'w', encoding=ecode, newline='') as fw:
            writer = csv.writer(fw)
            writer.writerows(lines_out)
        print(f'輸出預處理: {out_path} 共 {len(lines_out)} 筆')