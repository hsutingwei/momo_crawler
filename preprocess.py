# -*- coding: utf-8 -*-
"""
拆分單檔預處理：
1. 針對 crawler 資料夾所有「<keyword>_商品留言資料_YYYYMMDDHHMMSS.csv」
2. 修復換行斷行與千分位錯位
3. 簡繁轉換、全半形、去除非中字符
4. 根據留言 ID 去重
5. 個別輸出到 preprocess/pre_<keyword>_<timestamp>.csv
用法：python preprocess.py --keyword "益生菌"
"""
import os
import re
import csv
import unicodedata
import argparse
from opencc import OpenCC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--keyword', type=str, help='搜尋關鍵字', default='益生菌')
    return p.parse_args()

args = parse_args()
keyword = args.keyword

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_CRAWLER = os.path.join(ROOT, 'crawler')
DIR_PRE = os.path.join(ROOT, 'preprocess')
os.makedirs(DIR_PRE, exist_ok=True)

ecode = 'utf-8-sig'
cc = OpenCC('s2tw')

# 判斷行是否為新紀錄
row_start_re = re.compile(r'^\d+,')
def is_valid_row(line):
    return bool(row_start_re.match(line))

# 修復內嵌換行
def clean_broken_csv(src, dst):
    buf = ''
    with open(src, 'r', encoding=ecode) as f, open(dst, 'w', encoding=ecode) as out:
        for raw in f:
            line = raw.rstrip('\n')
            if is_valid_row(line):
                if buf:
                    out.write(buf + '\n')
                buf = line
            else:
                buf += ' ' + line
        if buf:
            out.write(buf + '\n')

# 處理千分位：只在價格欄含逗號時合併
def fix_price(tokens, idx=3):
    if idx >= len(tokens): return tokens
    price = tokens[idx].replace('"','')
    if ',' not in price:
        return tokens
    parts = [price]
    i = idx + 1
    while i < len(tokens) and tokens[i].replace('"','').isdigit() and len(tokens[i].replace('"',''))==3:
        parts.append(tokens[i].replace('"',''))
        i += 1
    tokens[idx] = ''.join(parts)
    del tokens[idx+1:i]
    return tokens

# 保留中文字，其他替換成分隔符
def sanitize_chinese(text, seg_char = ';'):
    """
    去除非中文符號，把非中文替還成 ';'。
    此動作可保留所有漢字符號，除了繁體字，簡體字、再簡字、三簡字等，亞洲國家的漢字也會保留，例如日文漢字等國家的漢字都會保留

    :param str text: 來源字串
    :param str str: 替換符號
    :return str: 替換後結果
    """
    text = re.sub(r'[^\u3400-\u9fcc\uf900-\ufa2d\u3041-\u3129\uff66-\uff9f]+', seg_char, text)
    text = re.sub(r'[[]-?() \f\n\r\t\v:ο\';×~+&!"_#`『』˙>\/%<β=÷*€ˊιμ$ˇ°χˋˍλ}¯@| {επδθ±§ ^αγ  ·,.:：﹕;；<>《》︽︾＜＞「」［］　『』【】〔〕︹︺︻︼﹁﹂﹃﹄［］﹝﹞＼／﹨∕？﹖、‘’′｜∣∥↖↘↗↙︱︳︴↑↓－¯〈〉?[{]}\|!`~@#$%^&*()-=_+┬┴├─┼┤┌┐╞═╪╡│▕└┘╭╮╰╯╔╦╗╠═╬╣╓╥╖╒╤╕║╚╩╝╟╫╢╙╨╜╞╪╡╘╧╛＿ˍ▁▂▃▄▅▆▇█▏▎▍▌▋▊▉◢', seg_char, text)
    text = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳⓪❶❷❸❹❺❻❼❽❾❿⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉１２３４５６７８９０ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ⒜⒝⒞⒟⒠⒡⒢⒣⒤⒥⒦⒧⒨⒩⒪⒫⒬⒭⒮⒯⒰⒱⒲⒳⒴⒵￥￡€￠¥〔〕【】﹝﹞〈〉﹙﹚《》（）｛｝﹛﹜︵︶︷︸︹︺︻︼︽︾︿﹀＜＞∩∪≒≌∵∴×／﹣±≦≧﹤﹥≠≡¼½¾³²∞√㏒∷°÷ˇ㏑∫∮∠∟⊿＋﹢⊥╳⊾℅㎎㎏㎜㎝㎞㎡㏄㏎㏕℃℉‰˙ˊˇˋ\u3105-\u3129]+', seg_char, text)
    text = re.sub(r'[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ]+', seg_char, text)
    text = re.sub(r'[' + seg_char + ']+', seg_char, text)

    # 移除字串開頭和結尾的逗號
    if text.startswith(seg_char):
        text = text[1:]
    if text.endswith(seg_char):
        text = text[:-1]

    return text

# 全形轉半形
def full2half(s):
    return unicodedata.normalize('NFKC', s)

for fname in os.listdir(DIR_CRAWLER):
    if f'{keyword}_商品留言資料_' not in fname or not fname.endswith('.csv'):
        continue
    m = re.search(r'_(\d{14})\.csv$', fname)
    if not m: continue
    ts = m.group(1)
    src = os.path.join(DIR_CRAWLER, fname)
    tmp = os.path.join(DIR_PRE, f'_tmp_{fname}')
    out = os.path.join(DIR_PRE, f'pre_{keyword}_{ts}.csv')

    # 1. 修復換行
    clean_broken_csv(src, tmp)

    seen = set()
    rows_out = []
    with open(tmp, 'r', encoding=ecode, newline='') as rf:
        reader = csv.reader(rf)
        for row in reader:
            if len(row) < 6:
                continue
            #row = fix_price(row)
            msg_id = row[6]
            if msg_id in seen:
                continue
            seen.add(msg_id)
            # 名稱全半形 + 繁簡轉換
            row[1] = cc.convert(full2half(row[1]))
            # 留言全半形 + 繁簡 + 保留中文
            row[5] = sanitize_chinese(cc.convert(full2half(row[5])))
            rows_out.append(row)

    os.remove(tmp)
    # 寫出
    with open(out, 'w', encoding=ecode, newline='') as wf:
        writer = csv.writer(wf)
        writer.writerows(rows_out)
    print(f'輸出 {out}，共 {len(rows_out)} 筆')