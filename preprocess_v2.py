# -*- coding: utf-8 -*-
"""
CKIP 版前處理：
1) 掃描 crawler 目錄：<keyword>_商品留言資料_YYYYMMDDHHMMSS.csv
2) 修復拆行 → 基本文字正規化（NFKC、繁簡、清掉控制字元/URL、規整空白）
3) CKIP 斷詞 + 詞性
4) 斷詞後「數字+單位/幣值」規一（5毛/0.5元、3萬、7折、15%...）
5) 輸出：
   - preprocess/pre_<keyword>_<ts>_norm.csv        （新增 norm_text 欄位）
   - preprocess/pre_<keyword>_<ts>_tokens.jsonl    （每列包含 tokens/pos/numeric_mentions）
用法：
    python preprocess_v2.py --keyword "益生菌"
"""

import os
import sys
import re
import csv
import json
import unicodedata
import argparse
import psycopg2
from typing import List, Dict, Any, Tuple, Optional, Set
from opencc import OpenCC
# 添加專案路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import DatabaseConfig

# 可選：若你想跳過 CKIP（沒有安裝/離線），把 USE_CKIP=False，會退化為簡單的「以空白分詞」
USE_CKIP = True
if USE_CKIP:
    try:
        from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger
    except Exception as e:
        print("[WARN] 無法載入 ckip-transformers，將退化為簡易分詞（英文/數字/中文段）")
        USE_CKIP = False

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--keyword', type=str, default='', help='搜尋關鍵字')
    return p.parse_args()

args = parse_args()
keyword = args.keyword or '口罩'
pipeline_version = '20250812_6c945d166053fd2bcbdd26c18bff77a1502add2d'

ROOT = os.path.dirname(os.path.abspath(__file__))
DIR_CRAWLER = os.path.join(ROOT, 'crawler')
DIR_PRE = os.path.join(ROOT, 'preprocess_v3', keyword)
os.makedirs(DIR_PRE, exist_ok=True)

ENCODING = 'utf-8-sig'
cc = OpenCC('s2tw')  # 簡→繁（你抓到簡體時可轉繁）

# -------- CSV 斷行修復（沿用你的邏輯） --------
row_start_re = re.compile(r'^\d+,')  # 假設每列以流水號開頭
def is_valid_row(line: str) -> bool:
    return bool(row_start_re.match(line))

def repair_broken_csv(src: str, dst: str) -> None:
    buf = ''
    with open(src, 'r', encoding=ENCODING) as f, open(dst, 'w', encoding=ENCODING) as out:
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

# -------- 文字正規化 --------
url_re = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
control_re = re.compile(r'[\u0000-\u001F\u007F]+')

def normalize_text(s: str) -> str:
    if s is None:
        return ''
    # 全形→半形，Unicode 正規化
    s = unicodedata.normalize('NFKC', s)
    # 簡→繁
    s = cc.convert(s)
    # 移除 URL、控制字元
    s = url_re.sub(' ', s)
    s = control_re.sub(' ', s)
    # 規整空白
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# -------- CKIP 斷詞/詞性 --------
WS = None
POS = None
if USE_CKIP:
    # 預設載 GPU=False；若你要 GPU：CkipWordSegmenter(device=0)
    print("Initializing drivers ... WS")
    WS = CkipWordSegmenter(model="bert-base", device=0)
    print("Initializing drivers ... POS")
    POS = CkipPosTagger(model="bert-base", device=0)

def simple_tokenize(s: str) -> Tuple[List[str], List[str]]:
    """
    退化版分詞：將英文/數字/中文段切開（不精準，只作備援）。
    """
    # 將非中英文數字轉空白，再 split
    ss = re.sub(r'[^0-9A-Za-z\u4E00-\u9FFF\.%％\-~～￥¥$＄．・·\uFF10-\uFF19\uFF21-\uFF3A\uFF41-\uFF5A]', ' ', s)
    ss = re.sub(r'\s+', ' ', ss).strip()
    toks = ss.split() if ss else []
    poses = ['x'] * len(toks)
    return toks, poses

def ckip_tokenize(s: str) -> Tuple[List[str], List[str]]:
    if not s:
        return [], []
    if not USE_CKIP:
        return simple_tokenize(s)
    ws = WS([s])[0]
    pos = POS([ws])[0]
    return ws, pos

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

# -------- 數值/單位/幣值 規一 --------
# 幣值單位（台灣）：元/塊/圓，角/毛=0.1 元，分=0.01 元
CURRENCY_UNITS = {
    '元': 1.0, '塊': 1.0, '圓': 1.0,
    '角': 0.1, '毛': 0.1,
    '分': 0.01
}
CURRENCY_PREFIX = {
    '萬': 10000.0,
    '千': 1000.0,
    '百': 100.0
}
CURRENCY_SYMBOLS = {'$', '＄', '￥', '¥', 'NT$', 'NTD', 'TWD', '新台幣'}

def parse_number(s: str) -> Optional[float]:
    # 支援千分位、全形小數點、百分比
    if s is None:
        return None
    ss = s.replace(',', '')
    ss = ss.replace('．', '.')
    try:
        return float(ss)
    except Exception:
        return None

def normalize_numeric_tokens(tokens: List[str], pos: List[str]) -> Dict[str, Any]:
    """
    輸入 CKIP tokens/pos，輸出：
      - normalized_tokens: 對於數值單位合併後的可讀 token（例如 "0.5元"）
      - numeric_mentions: [{raw, value, unit, start_idx, end_idx, kind}]
        kind: currency/percent/discount/quantity/other
    規則：
      1) 數字 + (萬|千|百)* + (元|塊|圓|角|毛|分)
      2) (數字)% → 百分比 (value=數字/100)
      3) X折 → 折扣 (value=X/10)
      4) 區間 5~10 → 兩筆 mention
    """
    n = len(tokens)
    i = 0
    normalized_tokens: List[str] = []
    mentions: List[Dict[str, Any]] = []

    def push_token(t: str):
        normalized_tokens.append(t)

    while i < n:
        t = tokens[i]
        p = pos[i] if i < len(pos) else 'x'
        # 規則 2: 百分比
        m_pct = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*[%％]', t)
        if m_pct:
            v = float(m_pct.group(1)) / 100.0
            normalized = f"{v}"
            mentions.append({
                'raw': t, 'value': v, 'unit': 'ratio',
                'start_idx': i, 'end_idx': i, 'kind': 'percent'
            })
            push_token(normalized)
            i += 1
            continue

        # 規則 3: 折扣（7折、8.5折）
        m_disc = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*折', t)
        if m_disc:
            v = float(m_disc.group(1)) / 10.0
            mentions.append({
                'raw': t, 'value': v, 'unit': 'ratio',
                'start_idx': i, 'end_idx': i, 'kind': 'discount'
            })
            push_token(f"{v}")
            i += 1
            continue

        # 規則 1: 幣值/數量（數字 [+萬/千/百] + 單位(元/塊/圓/角/毛/分)）
        # 先抓可能的前綴符號（NT$, $, ￦ 等），通常會分成獨立 token；若黏在一起也盡量處理
        currency_sym = None
        if t in CURRENCY_SYMBOLS:
            currency_sym = t
            i += 1
            if i >= n:
                push_token(t)
                break
            t = tokens[i]
            p = pos[i] if i < len(pos) else 'x'

        # 如果 t 是數字（含小數）
        num = parse_number(t)
        if num is not None:
            start_i = i - 1 if currency_sym else i
            end_i = i

            # 可能緊接著前綴（萬/千/百）
            mul = 1.0
            j = i + 1
            if j < n and tokens[j] in CURRENCY_PREFIX:
                mul *= CURRENCY_PREFIX[tokens[j]]
                end_i = j
                j += 1

            # 再看單位（元/塊/圓/角/毛/分）
            unit = None
            if j < n and tokens[j] in CURRENCY_UNITS:
                unit = tokens[j]
                mul *= CURRENCY_UNITS[unit]
                end_i = j
                j += 1

            # 若是幣值或數量
            if unit is not None or currency_sym:
                value = num * mul
                raw = ' '.join(tokens[(start_i if start_i >= 0 else 0):j])
                mentions.append({
                    'raw': raw,
                    'value': value,
                    'unit': unit if unit else 'currency',
                    'start_idx': start_i if start_i >= 0 else i,
                    'end_idx': j - 1,
                    'kind': 'currency'
                })
                push_token(f"{value}{unit or ''}")
                i = j
                continue
            else:
                # 非幣值數字（可能是容量、數量…）
                push_token(tokens[i])
                i += 1
                continue

        # 區間樣式：5~10 / 5-10
        m_range = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*[-~～]\s*([0-9]+(?:\.[0-9]+)?)', t)
        if m_range:
            v1 = float(m_range.group(1))
            v2 = float(m_range.group(2))
            mentions.append({'raw': t, 'value': v1, 'unit': None, 'start_idx': i, 'end_idx': i, 'kind': 'quantity'})
            mentions.append({'raw': t, 'value': v2, 'unit': None, 'start_idx': i, 'end_idx': i, 'kind': 'quantity'})
            push_token(t)
            i += 1
            continue

        # 其他：原樣保留
        push_token(t)
        i += 1

    return {
        'normalized_tokens': normalized_tokens,
        'numeric_mentions': mentions
    }

def process_file(src_csv: str, ts: str, keyword: str, token_dict: Set[str]):
    tmp = os.path.join(DIR_PRE, f'_tmp_{os.path.basename(src_csv)}')
    out_norm_csv = os.path.join(DIR_PRE, f'pre_{keyword}_{ts}_norm.csv')
    out_tokens_jsonl = os.path.join(DIR_PRE, f'pre_{keyword}_{ts}_tokens.jsonl')

    # 1) 修復拆行
    repair_broken_csv(src_csv, tmp)

    seen_ids = set()
    out_rows = []
    jsonl_fp = open(out_tokens_jsonl, 'w', encoding=ENCODING)

    with open(tmp, 'r', encoding=ENCODING, newline='') as rf:
        reader = csv.reader(rf)
        for row in reader:
            # 你原檔是以 row[6] 當留言 ID、row[1] 商品名稱、row[5] 留言
            if len(row) < 7:
                continue

            try:
                msg_id = row[6]
            except Exception:
                continue

            if msg_id in seen_ids:
                continue
            seen_ids.add(msg_id)

            # 商品名稱正規化（維持你原本流程）
            if len(row) > 1:
                row[1] = normalize_text(row[1])

            # 留言文本：做正規化，但不刪英文/數字
            raw_comment = row[5] if len(row) > 5 else ''
            norm_text = normalize_text(raw_comment)

            # 保留中文字，其他替換成分隔符
            norm_text = sanitize_chinese(norm_text)

            # 斷詞
            tokens, poses = ckip_tokenize(norm_text)
            # 斷詞後處理（長詞 > 4，用字典(2,3字)切分）
            tokens, poses = split_long_tokens_with_pos_rmm(tokens, poses, token_dict)

            # 數值/幣值規一
            # num_norm = normalize_numeric_tokens(tokens, poses)
            # normalized_tokens = num_norm['normalized_tokens']
            # numeric_mentions = num_norm['numeric_mentions']

            # CSV：在最後加一欄 norm_text
            out_row = list(row)
            out_row.append(norm_text)
            out_rows.append(out_row)

            # JSONL：輸出 token/pos/numeric
            json_line = {
                'msg_id': msg_id,
                'norm_text': norm_text,
                'tokens': tokens,
                'pos': poses,
                # 'normalized_tokens': normalized_tokens,
                # 'numeric_mentions': numeric_mentions
            }
            jsonl_fp.write(json.dumps(json_line, ensure_ascii=False) + '\n')

    jsonl_fp.close()
    os.remove(tmp)

    # 2) 輸出正規化 CSV
    #   加 header（若你的原始檔沒有 header，這裡只附加 norm_text）
    with open(out_norm_csv, 'w', encoding=ENCODING, newline='') as wf:
        writer = csv.writer(wf)
        for r in out_rows:
            writer.writerow(r)

    print(f"輸出：{out_norm_csv}（{len(out_rows)} 筆）")
    print(f"輸出：{out_tokens_jsonl}（JSONL，每行一筆 tokens/pos/numeric）")

def load_token_dict(pipeline_version: str, conn) -> Set[str]:
    """
    從 comment_tokens 載入長度 2~3 的字詞，作為補充斷詞字典
    Args:
        pipeline_version (str): pipeline_version
        conn: psycopg2 connection
    Returns:
        Set[str]: token 字典
    """
    sql = """
        SELECT DISTINCT token
        FROM comment_tokens
        WHERE length(token) IN (2,3)
          AND pipeline_version = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (pipeline_version,))
        rows = cur.fetchall()
    return {r[0] for r in rows}

def split_long_tokens_with_pos_rmm(tokens: List[str], poses: List[str], dict_set: Set[str]) -> Tuple[List[str], List[str]]:
    """
    後序長詞優先（最大反向匹配, Right-to-left MaxMatch）：
    - 對長詞 (>4) 從右往左切，優先 3 字，再 2 字，否則落單字
    - 子詞 POS 全部繼承原詞 POS
    - 句子最終順序保持不變
    """
    out_tokens, out_poses = [], []

    for tok, pos in zip(tokens, poses):
        if len(tok) <= 4:
            out_tokens.append(tok); out_poses.append(pos); continue

        parts: List[str] = []
        j = len(tok)
        while j > 0:
            matched = None
            for L in (3, 2):  # 長詞優先
                if j - L >= 0 and tok[j-L:j] in dict_set:
                    matched = tok[j-L:j]; break
            if matched:
                parts.append(matched); j -= len(matched)
            else:
                parts.append(tok[j-1]); j -= 1

        parts.reverse()  # 還原左→右的原始閱讀順序
        out_tokens.extend(parts)
        out_poses.extend([pos] * len(parts))

    return out_tokens, out_poses

def main():
    db = DatabaseConfig()
    conn = db.get_connection()
    conn.autocommit = False
    cur = conn.cursor()
    try:
        token_dict = load_token_dict(pipeline_version, cur)
    except Exception as e:
        print(f"Error loading token dictionary: {e}")
        token_dict = set()
    finally:
        cur.close()
        conn.close()

    for fname in os.listdir(DIR_CRAWLER):
        if f'{keyword}_商品留言資料_' not in fname or not fname.endswith('.csv'):
            continue
        m = re.search(r'_(\d{14})\.csv$', fname)
        if not m:
            continue
        ts = m.group(1)
        src = os.path.join(DIR_CRAWLER, fname)
        process_file(src, ts, keyword, token_dict)

if __name__ == '__main__':
    main()
