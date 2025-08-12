# upload_preprocessed_to_db.py
# -*- coding: utf-8 -*-
"""
把 preprocess_v2/ 產出的 _norm.csv 與 _tokens.jsonl 上傳到 DB。
需求：
- 跳過兩種「第一行標題」
- comment_text_norm / comment_tokens_vec：ON CONFLICT UPSERT
- comment_tokens（明細）：先刪舊的同 pipeline_version，再批量插入
- 連線方式沿用 csv_to_db.py 的 DatabaseConfig

# 安裝需要的套件（若尚未安裝）
pip install psycopg2-binary python-dotenv

# 執行（必要參數：pipeline version）
python upload_preprocessed_to_db.py --pipeline-version ckip-pre-v1

# 只看掃描結果，不寫入
python upload_preprocessed_to_db.py --pipeline-version ckip-pre-v1 --dry-run

# 指定不同根目錄（若把檔案分不同關鍵字子資料夾也 OK，程式會遞迴找）
python upload_preprocessed_to_db.py --base-dir preprocess_v2 --pipeline-version ckip-pre-v1
"""

import os
import sys
import csv
import json
import argparse
from typing import Iterator, List, Tuple

from dotenv import load_dotenv
load_dotenv()

# 走原本的連線方式
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.database import DatabaseConfig  # noqa: E402

import psycopg2  # type: ignore
import psycopg2.extras as pgx  # type: ignore


def iter_files(root: str, suffix: str) -> Iterator[str]:
    for base, _, files in os.walk(root):
        for f in files:
            if f.endswith(suffix):
                yield os.path.join(base, f)


def is_norm_header(row: List[str]) -> bool:
    # _norm.csv 第一行標題開頭是「商品ID,商品名稱, ...」
    if not row:
        return False
    head = "".join(row[:2])
    return "商品ID" in head or "商品名稱" in head


def is_tokens_header(obj: dict) -> bool:
    # tokens.jsonl 第一行：{"msg_id":"留言ID", "norm_text":"留言", "tokens":["留言"], "pos":["Na"]}
    return (
        obj.get("msg_id") == "留言ID"
        or (obj.get("tokens") == ["留言"] and obj.get("norm_text") == "留言")
    )


def upsert_comment_text_norm(cur, rows: List[Tuple[str, str, str]]):
    """
    rows: [(comment_id, pipeline_version, norm_text)]
    """
    sql = """
    INSERT INTO comment_text_norm (comment_id, pipeline_version, norm_text)
    VALUES %s
    ON CONFLICT (comment_id) DO UPDATE
      SET pipeline_version = EXCLUDED.pipeline_version,
          norm_text = EXCLUDED.norm_text,
          created_at = CURRENT_TIMESTAMP;
    """
    pgx.execute_values(cur, sql, rows, page_size=1000)


def upsert_comment_tokens_vec(cur, rows: List[Tuple[str, str, list, list]]):
    """
    rows: [(comment_id, pipeline_version, tokens[], pos_tags[])]
    """
    sql = """
    INSERT INTO comment_tokens_vec (comment_id, pipeline_version, tokens, pos_tags)
    VALUES %s
    ON CONFLICT (comment_id) DO UPDATE
      SET pipeline_version = EXCLUDED.pipeline_version,
          tokens = EXCLUDED.tokens,
          pos_tags = EXCLUDED.pos_tags,
          created_at = CURRENT_TIMESTAMP;
    """
    pgx.execute_values(cur, sql, rows, page_size=500)


def replace_comment_tokens(cur, comment_id: str, pipeline_version: str, tokens: List[str], pos: List[str]):
    """
    先刪除該 comment_id + pipeline_version，再批量插入明細（帶 token_order）
    """
    cur.execute(
        "DELETE FROM comment_tokens WHERE comment_id=%s AND pipeline_version=%s",
        (comment_id, pipeline_version),
    )
    if not tokens:
        return
    data = [(comment_id, i, tok, (pos[i] if i < len(pos) else None), pipeline_version)
            for i, tok in enumerate(tokens)]
    sql = """
    INSERT INTO comment_tokens (comment_id, token_order, token, pos_tag, pipeline_version)
    VALUES %s
    """
    pgx.execute_values(cur, sql, data, page_size=1000)


def process_norm_file(cur, fpath: str, pipeline_version: str, batch: int = 2000) -> int:
    """
    讀取 *_norm.csv，上傳 comment_text_norm
    - 跳過第一行標題
    - 取留言ID=第7欄(row[6])，norm_text=最後一欄(row[-1])
    """
    inserted = 0
    buf: List[Tuple[str, str, str]] = []

    with open(fpath, "r", encoding="utf-8-sig", newline="") as rf:
        reader = csv.reader(rf)
        first = True
        for row in reader:
            if first:
                first = False
                if is_norm_header(row):
                    continue  # 跳過標題行
            if len(row) < 7:
                continue
            comment_id = str(row[6]).strip()
            if not comment_id or comment_id == "留言ID":
                continue
            norm_text = str(row[-1]) if row else ""
            buf.append((comment_id, pipeline_version, norm_text))
            if len(buf) >= batch:
                upsert_comment_text_norm(cur, buf)
                inserted += len(buf)
                buf = []
        if buf:
            upsert_comment_text_norm(cur, buf)
            inserted += len(buf)
    return inserted


def process_tokens_file(cur, fpath: str, pipeline_version: str, batch_vec: int = 500) -> Tuple[int, int]:
    """
    讀取 *_tokens.jsonl：
      - 跳過第一行標題 JSON
      - 對 comment_tokens_vec 走 UPSERT（批量）
      - 對 comment_tokens 逐筆 replace（刪+插）
    """
    vec_rows: List[Tuple[str, str, list, list]] = []
    first_line = True
    n_vec = 0
    n_detail = 0

    with open(fpath, "r", encoding="utf-8-sig") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            if first_line and is_tokens_header(obj):
                first_line = False
                continue
            first_line = False

            comment_id = str(obj.get("msg_id", "")).strip()
            if not comment_id or comment_id == "留言ID":
                continue

            tokens = obj.get("tokens") or []
            pos = obj.get("pos") or []

            # vec upsert buffer
            vec_rows.append((comment_id, pipeline_version, tokens, pos))
            if len(vec_rows) >= batch_vec:
                upsert_comment_tokens_vec(cur, vec_rows)
                n_vec += len(vec_rows)
                vec_rows = []

            # detail replace per comment
            replace_comment_tokens(cur, comment_id, pipeline_version, tokens, pos)
            n_detail += len(tokens)

    if vec_rows:
        upsert_comment_tokens_vec(cur, vec_rows)
        n_vec += len(vec_rows)

    return n_vec, n_detail


def main():
    ap = argparse.ArgumentParser(description="Upload preprocess_v2 outputs to DB")
    ap.add_argument("--base-dir", default="preprocess_v2", help="根目錄（含子資料夾）")
    ap.add_argument("--pipeline-version", required=True, help="例如 ckip-pre-v1")
    ap.add_argument("--dry-run", action="store_true", help="只掃描不寫入")
    args = ap.parse_args()

    db = DatabaseConfig()
    conn = db.get_connection()
    conn.autocommit = False

    try:
        cur = conn.cursor()

        # 先處理 _norm.csv
        total_norm = 0
        for f in iter_files(args.base_dir, "_norm.csv"):
            print(f"[norm] {f}")
            if args.dry_run:
                continue
            total_norm += process_norm_file(cur, f, args.pipeline_version)
            conn.commit()

        # 再處理 _tokens.jsonl
        total_vec = 0
        total_detail_tokens = 0
        for f in iter_files(args.base_dir, "_tokens.jsonl"):
            print(f"[tokens] {f}")
            if args.dry_run:
                continue
            n_vec, n_detail = process_tokens_file(cur, f, args.pipeline_version)
            total_vec += n_vec
            total_detail_tokens += n_detail
            conn.commit()

        if not args.dry_run:
            print(f"完成：comment_text_norm UPSERT {total_norm} 筆，"
                  f"comment_tokens_vec UPSERT {total_vec} 筆，"
                  f"comment_tokens 插入 token 明細 {total_detail_tokens} 筆")
        else:
            print("Dry-run 完成（未寫入 DB）")

    except Exception as e:
        conn.rollback()
        print("發生錯誤，已回滾：", e)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()