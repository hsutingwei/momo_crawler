# build_tfidf_index.py
# -*- coding: utf-8 -*-
"""
從 comment_tokens 增量建立/更新 TF-IDF 索引（tfidf_* 表）
- 支援 corpus: global / keyword=<值>
- 以 pipeline_version 分版本維護
- 增量安全：重建同一文件會先撤銷舊 DF、刪舊 TF，再寫入新 TF/DF


# 先安裝依賴
pip install psycopg2-binary python-dotenv

# 全域語料（global）— 針對某個 pipeline_version
python build_tfidf_index.py --pipeline-version ckip-pre-v1 --corpus-type global

# 只把「尚未索引過」的評論建進 index（增量）
python build_tfidf_index.py --pipeline-version ckip-pre-v1 --corpus-type global --only-missing

# 指定關鍵字語料（keyword=益生菌）
python build_tfidf_index.py --pipeline-version ckip-pre-v1 --corpus-type keyword --corpus-key 益生菌

# 先測 100 篇、每 200 篇提交一次
python build_tfidf_index.py --pipeline-version ckip-pre-v1 --corpus-type keyword --corpus-key 益生菌 --limit 100 --batch 200

# 預設只忽略 ';'
python build_tfidf_index.py --pipeline-version ckip-pre-v1 --corpus-type global

# 忽略多個符號：分號；全形分號；逗號；頓號
python build_tfidf_index.py --pipeline-version ckip-pre-v1 --corpus-type global \
  --ignore-delims ';' '；' ',' '、'

"""

import os
import sys
import argparse
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional, Set

from dotenv import load_dotenv
load_dotenv()

# 沿用 csv_to_db.py 的連線方式
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.database import DatabaseConfig  # type: ignore

import psycopg2  # type: ignore
import psycopg2.extras as pgx  # type: ignore


def get_or_create_corpus(cur, corpus_type: str, corpus_key: Optional[str], pipeline_version: str) -> int:
    cur.execute("""
        INSERT INTO tfidf_corpus (corpus_type, corpus_key, pipeline_version, total_docs)
        VALUES (%s, %s, %s, 0)
        ON CONFLICT (corpus_type, corpus_key, pipeline_version) DO NOTHING
        RETURNING corpus_id
    """, (corpus_type, corpus_key, pipeline_version))
    row = cur.fetchone()
    if row:
        return row[0]
    # already exists
    cur.execute("""
        SELECT corpus_id FROM tfidf_corpus
        WHERE corpus_type=%s AND corpus_key IS NOT DISTINCT FROM %s AND pipeline_version=%s
    """, (corpus_type, corpus_key, pipeline_version))
    return cur.fetchone()[0]


def load_stopwords(cur, pipeline_version: str) -> Set[str]:
    # 可選：若你沒有建立 nlp_stopwords，這段會失敗；加 try 保護
    try:
        cur.execute("""
            SELECT token FROM nlp_stopwords WHERE pipeline_version=%s
        """, (pipeline_version,))
        return {r[0] for r in cur.fetchall()}
    except Exception:
        return set()


def fetch_candidate_comment_ids(cur, corpus_type: str, corpus_key: Optional[str], pipeline_version: str,
                                limit: Optional[int] = None, only_missing: bool = False) -> List[str]:
    """
    從 comment_tokens 抓該 pipeline 的評論列表。
    - global: 全部該版本評論
    - keyword: 透過 products.keyword 過濾
    - only_missing=True: 只抓尚未被該 corpus 索引過的評論
    """
    params = [pipeline_version]
    where = ["ct.pipeline_version=%s"]
    join = ""
    post = ""

    if corpus_type == "keyword":
        join = "JOIN product_comments pc ON pc.comment_id=ct.comment_id JOIN products p ON p.id=pc.product_id"
        where.append("p.keyword=%s")
        params.append(corpus_key)

    if only_missing:
        post = "AND NOT EXISTS (SELECT 1 FROM tfidf_doc_index di WHERE di.comment_id=ct.comment_id AND di.corpus_id=%s)"
        # corpus_id 要在外面 format（呼叫時填）
    sql = f"""
        SELECT DISTINCT ct.comment_id
        FROM comment_tokens ct
        {join}
        WHERE {" AND ".join(where)}
        {post}
        ORDER BY ct.comment_id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return sql, params


def ensure_vocab(cur, pipeline_version: str, tokens: Iterable[str]) -> Dict[str, int]:
    toks = list({t for t in tokens if t})
    if not toks:
        return {}

    # 1) 批量 upsert 新詞進 vocab
    pgx.execute_values(
        cur,
        """
        INSERT INTO tfidf_vocab (token, pipeline_version)
        VALUES %s
        ON CONFLICT (token, pipeline_version) DO NOTHING
        """,
        [(t, pipeline_version) for t in toks],
        page_size=1000
    )

    # 2) 把 token -> term_id 查回來
    # 建議用 ANY(array)；若你偏好 IN，也可以用 tuple：IN %s, (tuple(toks),)
    cur.execute(
        """
        SELECT token, term_id
        FROM tfidf_vocab
        WHERE pipeline_version = %s
          AND token = ANY(%s)
        """,
        (pipeline_version, toks)  # toks 會被轉成 array
    )
    mapping = {token: term_id for token, term_id in cur.fetchall()}
    return mapping



def get_existing_terms_for_doc(cur, comment_id: str, corpus_id: int) -> List[int]:
    cur.execute("""
        SELECT DISTINCT term_id
        FROM tfidf_term_freq
        WHERE comment_id=%s AND corpus_id=%s
    """, (comment_id, corpus_id))
    return [r[0] for r in cur.fetchall()]


def decrement_df(cur, corpus_id: int, term_ids: List[int]) -> None:
    if not term_ids:
        return
    # df -= 1（不得小於 0）
    pgx.execute_values(cur, """
        UPDATE tfidf_doc_freq
        SET df = GREATEST(df - 1, 0), updated_at=NOW()
        WHERE corpus_id=%s AND term_id=%s
    """, [(corpus_id, tid) for tid in term_ids], template=None, page_size=1000)


def delete_old_tf(cur, comment_id: str, corpus_id: int) -> None:
    cur.execute("""
        DELETE FROM tfidf_term_freq
        WHERE comment_id=%s AND corpus_id=%s
    """, (comment_id, corpus_id))


def upsert_doc_index(cur, comment_id: str, corpus_id: int, pipeline_version: str, doc_len: int) -> bool:
    """
    回傳 True 表示是第一次索引（需把 total_docs += 1），False 表示更新舊文件
    """
    cur.execute("""
        SELECT 1 FROM tfidf_doc_index WHERE comment_id=%s AND corpus_id=%s
    """, (comment_id, corpus_id))
    is_new = cur.fetchone() is None

    cur.execute("""
        INSERT INTO tfidf_doc_index (comment_id, corpus_id, pipeline_version, doc_len, updated_at)
        VALUES (%s, %s, %s, %s, NOW())
        ON CONFLICT (comment_id, corpus_id)
        DO UPDATE SET pipeline_version=EXCLUDED.pipeline_version, doc_len=EXCLUDED.doc_len, updated_at=NOW()
    """, (comment_id, corpus_id, pipeline_version, doc_len))
    return is_new


def insert_tf(cur, comment_id: str, corpus_id: int, term_counts: Dict[int, int]) -> int:
    if not term_counts:
        return 0
    rows = [(comment_id, corpus_id, tid, tf) for tid, tf in term_counts.items()]
    pgx.execute_values(cur, """
        INSERT INTO tfidf_term_freq (comment_id, corpus_id, term_id, tf)
        VALUES %s
        ON CONFLICT (comment_id, corpus_id, term_id) DO UPDATE SET tf=EXCLUDED.tf
    """, rows, page_size=1000)
    return len(rows)


def increment_df(cur, corpus_id: int, term_ids: Iterable[int]) -> None:
    term_ids = list(set(term_ids))
    if not term_ids:
        return
    pgx.execute_values(cur, """
        INSERT INTO tfidf_doc_freq (corpus_id, term_id, df)
        VALUES %s
        ON CONFLICT (corpus_id, term_id)
        DO UPDATE SET df = tfidf_doc_freq.df + 1, updated_at = NOW()
    """, [(corpus_id, tid, 1) for tid in term_ids], page_size=1000)


def bump_total_docs(cur, corpus_id: int, n: int) -> None:
    if n <= 0:
        return
    cur.execute("""
        UPDATE tfidf_corpus SET total_docs = total_docs + %s, updated_at=NOW()
        WHERE corpus_id=%s
    """, (n, corpus_id))


def fetch_tokens_for_comment(
    cur,
    comment_id: str,
    pipeline_version: str,
    stopwords: Set[str],
    ignore_delims: Set[str],
    drop_punct: bool = True
) -> List[str]:
    cur.execute("""
        SELECT token FROM comment_tokens
        WHERE comment_id=%s AND pipeline_version=%s
        ORDER BY token_order
    """, (comment_id, pipeline_version))
    toks = [r[0] for r in cur.fetchall()]

    # 1) 丟掉分隔符號（由參數控制）
    if ignore_delims:
        toks = [t for t in toks if t not in ignore_delims]

    # 2) 可選：丟掉純符號 token（例如 "!"、"…"）
    if drop_punct:
        toks = [t for t in toks if t and not PUNCT_RE.match(t)]

    # 3) 停用詞（你現有流程也會載入）
    if stopwords:
        toks = [t for t in toks if t not in stopwords]

    return toks


def run(corpus_type: str, pipeline_version: str, corpus_key: Optional[str],
        only_missing: bool, limit: Optional[int], batch: int,
        ignore_delims: Set[str]) -> None:
    db = DatabaseConfig()
    conn = db.get_connection()
    conn.autocommit = False

    try:
        cur = conn.cursor()

        corpus_id = get_or_create_corpus(cur, corpus_type, corpus_key, pipeline_version)
        stopwords = load_stopwords(cur, pipeline_version)

        # 取得候選評論清單
        sql, params = fetch_candidate_comment_ids(cur, corpus_type, corpus_key, pipeline_version,
                                                  limit=limit, only_missing=only_missing)
        # 若只抓未索引，需填 corpus_id
        if "di.corpus_id=%s" in sql:
            params = params + [corpus_id]
        cur.execute(sql, params)
        comment_ids = [r[0] for r in cur.fetchall()]
        print(f"[INFO] 待索引文件數：{len(comment_ids)}（corpus_id={corpus_id} / {corpus_type}={corpus_key} / pv={pipeline_version}）")

        new_docs = 0
        processed = 0

        for idx, cid in enumerate(comment_ids, 1):
            # 取 tokens
            tokens = fetch_tokens_for_comment(cur, cid, pipeline_version, stopwords, ignore_delims)
            if not tokens:
                # 這篇沒有 tokens，跳過（若之前有索引，應該也撤銷；此處簡化略過）
                continue

            # term -> tf
            tf_counts = Counter(tokens)
            # 準備 vocab
            vocab_map = ensure_vocab(cur, pipeline_version, tf_counts.keys())

            # 撤銷舊 DF / 刪舊 TF
            old_term_ids = get_existing_terms_for_doc(cur, cid, corpus_id)
            if old_term_ids:
                decrement_df(cur, corpus_id, old_term_ids)
                delete_old_tf(cur, cid, corpus_id)

            # upsert doc_index（判斷是否新文件）
            is_new_doc = upsert_doc_index(cur, cid, corpus_id, pipeline_version, doc_len=sum(tf_counts.values()))
            if is_new_doc:
                new_docs += 1

            # 插入 TF
            term_counts_by_id = {vocab_map[t]: c for t, c in tf_counts.items() if t in vocab_map}
            insert_tf(cur, cid, corpus_id, term_counts_by_id)

            # 更新 DF（用新的 distinct term_id）
            increment_df(cur, corpus_id, term_counts_by_id.keys())

            processed += 1
            if processed % batch == 0:
                conn.commit()
                print(f"[INFO] 已處理 {processed}/{len(comment_ids)}，新文件 {new_docs}")

        # 補 total_docs
        bump_total_docs(cur, corpus_id, new_docs)
        conn.commit()
        print(f"[DONE] 完成索引：處理 {processed} 篇；新文件 {new_docs}；corpus_id={corpus_id}")

    except Exception as e:
        conn.rollback()
        print("[ERROR] 發生錯誤，交易已回滾：", e)
        raise
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Build/Update TF-IDF index from comment_tokens")
    ap.add_argument("--pipeline-version", required=True, help="對齊 comment_tokens.pipeline_version，例如 ckip-pre-v1")
    ap.add_argument("--corpus-type", choices=["global", "keyword"], default="global")
    ap.add_argument("--corpus-key", help="當 corpus-type=keyword 時，指定 keyword 值")
    ap.add_argument("--only-missing", action="store_true", help="只建立尚未索引過的文件")
    ap.add_argument("--limit", type=int, help="只處理前 N 篇（測試用）")
    ap.add_argument("--batch", type=int, default=500, help="每處理多少篇提交一次交易")
    ap.add_argument("--ignore-delims", nargs="*", default=[";"], help="在計算 TF-IDF 時要忽略的分隔符號，空白分隔多個，預設為 ';'")
    args = ap.parse_args()

    if args.corpus_type == "keyword" and not args.corpus_key:
        ap.error("--corpus-key 在 corpus-type=keyword 時必填")

    run(args.corpus_type, args.pipeline_version, args.corpus_key, args.only_missing, args.limit, args.batch, ignore_delims=set(args.ignore_delims or []))


if __name__ == "__main__":
    main()
