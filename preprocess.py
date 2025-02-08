import os
from opencc import OpenCC
import re

### 此處理主要將商品名稱以及留言繁簡轉換
### 並根據留言ID去重複

ecode = 'utf-8-sig'
cc = OpenCC('s2tw')
dirPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\crawler"
outPath = r"C:\YvesProject\中央\線上評論\momo_crawler-main\preprocess\pre1.csv"
lineArr = [] # 暫存所有留言爬蟲的陣列
textObj = {} # 將所有留言根據留言ID去重複
finalArr = [] # 預處理最終結果

def do_money_special(lineArr):
    """
    因爬蟲時沒有正確將價格的千分位符號處理好，導致千分位符號與 csv 衝突，故此 function 專門處理這個問題

    :param list lineArr: 單行的數據陣列
    :return list: 處理好後將處理完的單行陣列回傳
    """
    start_index = 3
    result = lineArr[start_index].replace('"', '')
    haveError = False

    # 從 start_index + 1 開始檢查後續的索引值
    end_index = start_index + 1
    for i in range(start_index + 1, len(lineArr)):
        # 判斷該值是否為數字，且值為3位數 (千分位)
        if lineArr[i].replace('"', '').isdigit() and len(lineArr[i].replace('"', '')) == 3:
            result += lineArr[i].replace('"', '')
            haveError = True
        else:
            end_index = i # 紀錄非數字部分的索引
            break  # 遇到非數字則停止

    # 將累加的結果轉回整數並替換到原陣列
    lineArr[start_index] = result

    # 刪除中間的數字部分
    del lineArr[start_index + 1:end_index]
        
    return lineArr

def do_replace_to_Chinese(str, seg_char = ';'):
    """
    去除非中文符號，把非中文替還成 ';'

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

for f in os.listdir(dirPath):
    if os.path.isfile(os.path.join(dirPath, f)) and "留言資料" in f:
        with open(dirPath + "/" + f, 'r', encoding=ecode) as f:
            tmp_data = f.read()
            lineArr = tmp_data.split('\n')

### 只將商品名稱以及留言做簡/繁轉換
### 並根據留言ID去重複
for line in lineArr:
    if line is not '':
        text = line.split(',')
        text = do_money_special(text)
        tmp_id = text[5]
        if tmp_id not in textObj:
            text[1] = cc.convert(text[1]) # 簡繁轉換
            text[4] = cc.convert(text[4]) # 簡繁轉換
            text[4] = do_replace_to_Chinese(text[4]) # 去除非中文
            finalArr.append(','.join(text))


with open(outPath, 'w', encoding=ecode) as f:
    for line in finalArr:
        f.write(line + '\n')