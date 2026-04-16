# sf_02_mine_candidates.py
# -*- coding: utf-8 -*-
"""
SF Pipeline Step 2：從高信心評論挖掘 unigram 候選詞

功能：
  - 讀取 sf_highconf_comments（某輪篩選結果）
  - 利用 tfidf_term_freq / tfidf_doc_freq / tfidf_vocab / tfidf_corpus
    計算每個 unigram 的 discriminative score
  - 附加 dominant_pos（從 comment_tokens 聚合）
  - 附加 3 筆 example_comment_ids（同 label 中得分最高的）
  - 寫入 sf_keyword_candidates

過濾層次（依序執行）：
  1. SQL 基礎過濾：min_tf / min_df / token 長度 / 純數字 / 基礎停用詞
  2. POS 白名單過濾：保留語意有效詞性（可加 --exclude-nb 排除專有名詞）
  3. 通用 blacklist：對所有 label 都無辨識力的泛詞（可用 --no-generic-blacklist 關閉）
  4. label-specific blacklist：各 label 的專屬排除詞（可用 --no-label-blacklist 關閉）

使用方式：
  # 先 dry-run，確認過濾統計（預設啟用所有 blacklist + --exclude-nb）
  python sf_02_mine_candidates.py \\
      --run-id nli-seed-v1 \\
      --label-name High_Novelty \\
      --pipeline-version "20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac" \\
      --corpus-id 5 \\
      --exclude-nb \\
      --dry-run

  # 正式寫入（指定 mining_run 版本號方便追蹤）
  python sf_02_mine_candidates.py \\
      --run-id nli-seed-v1 \\
      --label-name High_Novelty \\
      --pipeline-version "20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac" \\
      --corpus-id 5 \\
      --exclude-nb \\
      --mining-run mine-v2-high-novelty-tf5-df3 \\
      --top-k 100 \\
      --no-dry-run

  # debug 用：關閉所有 blacklist，看原始結果
  python sf_02_mine_candidates.py \\
      --run-id nli-seed-v1 \\
      --label-name High_Novelty \\
      --pipeline-version "20250813_2608d61dc5b6d77a4a4582546ccb8a595673b5ac" \\
      --corpus-id 5 \\
      --no-generic-blacklist \\
      --no-label-blacklist \\
      --dry-run
"""

import os
import sys
import argparse
import logging
from collections import defaultdict
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.database import DatabaseConfig  # type: ignore

import psycopg2  # type: ignore
import psycopg2.extras as pgx  # type: ignore


# ─────────────────────────────────────────────────────────────
# 常數：POS 白名單與基礎停用詞
# ─────────────────────────────────────────────────────────────

# CKIP 詞性白名單：只保留名詞、狀態/心理/不及物動詞、形容詞
# D（副詞）先不納入：超/很/真 這類在 High_Novelty 中較通用，先手動審閱再決定
POS_WHITELIST = frozenset({
    "Na",   # 普通名詞：體驗、驚喜、新奇
    "Nb",   # 專有名詞：品牌名（讓人工決定）
    "VH",   # 狀態動詞（實為形容詞用法）：好用、神奇、厲害
    "VE",   # 存在動詞：有
    "VK",   # 心理動詞：喜歡、推薦
    "VA",   # 不及物動詞：值得
    "A",    # 形容詞：新鮮、獨特
})

# 基礎停用詞：只過濾功能性詞彙
# 刻意不過濾「超/好/真/不/沒」等情感詞，由人工審閱決定
BASIC_STOPWORDS = frozenset({
    # 助詞、連接詞、語氣詞
    "的", "了", "是", "在", "也", "都", "被", "把", "讓", "從", "到",
    "和", "與", "或", "如", "這", "那", "嗎", "呢", "啊", "哦", "喔",
    "哈", "呀", "吧", "但", "而", "且", "已", "再", "還", "就",
    "因為", "所以", "但是", "然而", "雖然", "如果", "因此", "其實",
    # 電商常見無意義高頻詞（購買行為描述，不具語意鑑別力）
    "商品", "東西", "東東", "物品",
    "買到", "買了", "買的", "賣家", "賣場",
})

# discriminative score 公式中的分母平滑值（避免除以 0）
DF_SMOOTHING = 0.001

# dominant_pos / example_comment_ids 查詢時，先抓 top_k * BUFFER 筆
# 再套全部過濾後取 top_k
QUERY_BUFFER_FACTOR = 4  # 從 2 調高到 4，為三層 Python 過濾預留足夠空間


# ─────────────────────────────────────────────────────────────
# 通用 Blacklist：對所有 label 都無辨識力的泛詞
#
# 設計原則：
#   - 只放高確定性的，不貪多
#   - 這些詞不論在哪個 label 的高信心集合裡都會高頻出現
#   - 不放情感詞（超/好/真）—— 這些留給人工審閱
# ─────────────────────────────────────────────────────────────
GENERIC_BLACKLIST = frozenset({
    # 評論行為詞（meta-commentary，本身不是語意信號）
    "評價", "評論", "評分", "留言", "回饋",
    # 過泛的體驗詞（在幾乎所有 label 的高信心集合都高頻，無鑑別力）
    "效果", "感受", "功效", "功能", "作用",
    # 過泛的認知詞（單獨出現時無法對應特定語意）
    "知道", "覺得", "認為", "感覺",
    # 過泛的期待詞
    "希望", "期待", "期望",
    # 人際關係詞（使用情境描述，不是語意信號）
    "朋友", "家人", "老婆", "老公", "媽媽", "爸爸",
    # 口感/味道（對大部分語意 label 無鑑別力，保健品、食品語料常見）
    "味道", "口味", "口感", "甜度", "風味",
    # 商品描述泛詞
    "品牌", "品質",
})


# ─────────────────────────────────────────────────────────────
# Label-specific Blacklist：各 label 的專屬排除詞
#
# 設計原則：
#   - 只放「在此 label 高信心評論中高頻，但語意上是巧合」的詞
#   - 例如 High_Novelty 語料集中大量保健品評論，成分名出現率高
#     但 '益生菌' 本身不代表新奇語意
#   - 其他 label 預留空集合，後續擴增時補入
# ─────────────────────────────────────────────────────────────
LABEL_BLACKLIST: dict[str, frozenset] = {
    "High_Novelty": frozenset({
        # 健康食品成分名（高頻但非語意信號）
        "葉黃素", "益生菌", "膠原", "蛋白", "生醫",
        "魚油", "膠囊", "維生素", "維他命", "乳酸菌",
        # 食材/食品名
        "鮭魚", "鱈魚",
        # 品牌名（首次購買某品牌 → 語意是「首次」，但詞本身是品牌名）
        # 品牌名優先用 --exclude-nb 擋，此處補漏網的 Na 標記品牌
        "天王", "中江", "歐可",
    }),
    # 其他 label 的 blacklist 待後續補入
    "High_Arousal":           frozenset(),
    "High_Repurchase_Intent": frozenset(),
    "Negative_Complaint":     frozenset(),
    "Advertisement":          frozenset(),
}


# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DB 連線
# ─────────────────────────────────────────────────────────────
def get_conn():
    """關閉 autocommit，讓整批寫入在單一 transaction 內完成。"""
    db_cfg = DatabaseConfig()
    conn = db_cfg.get_connection()
    conn.autocommit = False
    return conn


# ─────────────────────────────────────────────────────────────
# Step A：確認 label 高信心評論數量
# ─────────────────────────────────────────────────────────────
def fetch_label_doc_count(cur, run_id: str, label_name: str) -> int:
    cur.execute(
        """
        SELECT COUNT(*)
        FROM sf_highconf_comments
        WHERE run_id = %s AND label_name = %s
        """,
        (run_id, label_name),
    )
    return cur.fetchone()[0]


# ─────────────────────────────────────────────────────────────
# Step B：計算 unigram discriminative scores（核心 SQL，在 DB 內聚合）
# ─────────────────────────────────────────────────────────────
def compute_discriminative_scores(
    cur,
    run_id: str,
    label_name: str,
    corpus_id: int,
    pipeline_version: str,
    min_tf: int,
    min_df: int,
    min_token_len: int,
    fetch_limit: int,
) -> list[dict]:
    """
    在 DB 端完成所有聚合計算，Python 只做後處理。

    回傳：[{term_id, token, label_total_tf, label_doc_freq, global_df,
            discriminative_score}, ...]
    按 discriminative_score DESC 排列，已套基礎 SQL 過濾規則。
    """
    sql = """
        WITH
        -- ① 取出本輪目標 label 的高信心評論 ID 集合
        label_docs AS (
            SELECT comment_id
            FROM sf_highconf_comments
            WHERE run_id    = %(run_id)s
              AND label_name = %(label_name)s
        ),
        -- ② 計算 label 子語料的文件數（用於 discriminative_score 分母）
        label_size AS (
            SELECT COUNT(*)::FLOAT AS n
            FROM label_docs
        ),
        -- ③ 取全語料 total_docs（從 tfidf_corpus 讀取，不估算）
        corpus_total AS (
            SELECT total_docs::FLOAT AS n
            FROM tfidf_corpus
            WHERE corpus_id = %(corpus_id)s
        ),
        -- ④ 在 label 文件中聚合每個 term 的 TF 與 DF
        --    只 JOIN label_docs 的文件，不看全語料的 TF
        label_term_agg AS (
            SELECT
                ttf.term_id,
                SUM(ttf.tf)::INTEGER            AS label_total_tf,
                COUNT(DISTINCT ttf.comment_id)::INTEGER AS label_doc_freq
            FROM tfidf_term_freq ttf
            JOIN label_docs ld ON ld.comment_id = ttf.comment_id
            WHERE ttf.corpus_id = %(corpus_id)s
            GROUP BY ttf.term_id
        )
        -- ⑤ 計算差異化分數：label 內頻率 / global 文件頻率
        --    score = ln( (label_total_tf / label_n_docs)
        --               / (global_df / total_docs + smoothing) )
        SELECT
            v.term_id,
            v.token,
            lta.label_total_tf,
            lta.label_doc_freq,
            df.df                           AS global_df,
            LN(
                (lta.label_total_tf / ls.n)
                /
                ((df.df / ct.n) + %(smoothing)s)
            )                               AS discriminative_score
        FROM label_term_agg lta
        -- 取 token 文字（需指定 pipeline_version 避免跨版本 term_id 混用）
        JOIN tfidf_vocab v
            ON  v.term_id          = lta.term_id
            AND v.pipeline_version = %(pipeline_version)s
        -- 取 global document frequency
        JOIN tfidf_doc_freq df
            ON  df.term_id   = lta.term_id
            AND df.corpus_id = %(corpus_id)s
        CROSS JOIN label_size  ls
        CROSS JOIN corpus_total ct
        WHERE lta.label_total_tf >= %(min_tf)s
          AND lta.label_doc_freq >= %(min_df)s
          AND LENGTH(v.token)    >= %(min_token_len)s
          -- 排除純數字 token（123、456 等）
          AND v.token            !~ '^[0-9]+$'
          -- 排除基礎停用詞（使用 != ALL(array) 等同 NOT IN）
          AND NOT (v.token        = ANY(%(stopwords)s))
        ORDER BY discriminative_score DESC
        LIMIT %(fetch_limit)s
    """
    cur.execute(
        sql,
        {
            "run_id":         run_id,
            "label_name":     label_name,
            "corpus_id":      corpus_id,
            "pipeline_version": pipeline_version,
            "min_tf":         min_tf,
            "min_df":         min_df,
            "min_token_len":  min_token_len,
            "stopwords":      list(BASIC_STOPWORDS),
            "smoothing":      DF_SMOOTHING,
            "fetch_limit":    fetch_limit,
        },
    )
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ─────────────────────────────────────────────────────────────
# Step C：取得 dominant_pos（批次查詢，一次處理全部 token）
# ─────────────────────────────────────────────────────────────
def fetch_dominant_pos(
    cur,
    run_id: str,
    label_name: str,
    pipeline_version: str,
    tokens: list[str],
) -> dict[str, str]:
    """
    對 label 高信心評論中出現的 tokens，計算每個 token 最常見的詞性。
    回傳：{token: dominant_pos_tag}
    """
    if not tokens:
        return {}

    sql = """
        SELECT
            ct.token,
            MODE() WITHIN GROUP (ORDER BY ct.pos_tag) AS dominant_pos
        FROM comment_tokens ct
        JOIN sf_highconf_comments hc
            ON  hc.comment_id  = ct.comment_id
            AND hc.run_id      = %(run_id)s
            AND hc.label_name  = %(label_name)s
        WHERE ct.pipeline_version = %(pipeline_version)s
          AND ct.token = ANY(%(tokens)s)
        GROUP BY ct.token
    """
    cur.execute(
        sql,
        {
            "run_id":           run_id,
            "label_name":       label_name,
            "pipeline_version": pipeline_version,
            "tokens":           tokens,
        },
    )
    return {row[0]: row[1] for row in cur.fetchall()}


# ─────────────────────────────────────────────────────────────
# Step D：取得 example_comment_ids（批次，window function，一次搞定）
# ─────────────────────────────────────────────────────────────
def fetch_example_comment_ids(
    cur,
    run_id: str,
    label_name: str,
    corpus_id: int,
    term_ids: list[int],
    n_examples: int = 3,
) -> dict[int, list[str]]:
    """
    對每個 term_id，在 label 高信心評論中找得分最高的 n 筆 comment_id。
    使用 window function 一次查完所有 term_id，不做 N 次 query。
    回傳：{term_id: [comment_id_1, comment_id_2, ...]}
    """
    if not term_ids:
        return {}

    sql = """
        WITH ranked AS (
            SELECT
                ttf.term_id,
                hc.comment_id,
                ROW_NUMBER() OVER (
                    PARTITION BY ttf.term_id
                    ORDER BY hc.agg_score DESC
                ) AS rn
            FROM sf_highconf_comments hc
            JOIN tfidf_term_freq ttf
                ON  ttf.comment_id = hc.comment_id
                AND ttf.corpus_id  = %(corpus_id)s
                AND ttf.term_id    = ANY(%(term_ids)s)
            WHERE hc.run_id     = %(run_id)s
              AND hc.label_name = %(label_name)s
        )
        SELECT
            term_id,
            ARRAY_AGG(comment_id ORDER BY rn) AS example_ids
        FROM ranked
        WHERE rn <= %(n_examples)s
        GROUP BY term_id
    """
    cur.execute(
        sql,
        {
            "run_id":      run_id,
            "label_name":  label_name,
            "corpus_id":   corpus_id,
            "term_ids":    term_ids,
            "n_examples":  n_examples,
        },
    )
    return {row[0]: list(row[1]) for row in cur.fetchall()}


# ─────────────────────────────────────────────────────────────
# Step E：POS 過濾（Python 端）
# ─────────────────────────────────────────────────────────────
def apply_pos_filter(
    candidates: list[dict],
    pos_map: dict[str, str],
    use_pos_filter: bool,
    exclude_nb: bool,
) -> tuple[list[dict], int]:
    """
    將 dominant_pos 合併進 candidates，並套用 POS 白名單過濾。

    exclude_nb=True 時，從有效白名單中移除 'Nb'（專有名詞），
    用於過濾品牌名。

    dominant_pos 為 None 的 token 保留（可能是 tfidf 收錄但 comment_tokens
    未索引的邊緣 token，讓人工審閱時決定）。

    回傳：(filtered_candidates, removed_count)
    """
    # 依參數決定本次有效的 POS 白名單
    effective_whitelist = POS_WHITELIST - {"Nb"} if exclude_nb else POS_WHITELIST

    result = []
    removed = 0
    for cand in candidates:
        token = cand["token"]
        pos = pos_map.get(token)
        cand["dominant_pos"] = pos

        if not use_pos_filter:
            result.append(cand)
            continue

        # pos 為 None：保留，標記為未知詞性，人工審閱
        if pos is None:
            result.append(cand)
        elif pos in effective_whitelist:
            result.append(cand)
        else:
            removed += 1
    return result, removed


# ─────────────────────────────────────────────────────────────
# Step E2：Blacklist 過濾（Python 端，在 POS 過濾之後執行）
# ─────────────────────────────────────────────────────────────
def apply_blacklist_filter(
    candidates: list[dict],
    label_name: str,
    use_generic: bool,
    use_label: bool,
) -> tuple[list[dict], int, int]:
    """
    依序套用通用 blacklist 和 label-specific blacklist。

    回傳：(filtered_candidates, generic_removed_count, label_removed_count)
    """
    generic_removed = 0
    label_removed = 0
    result = []

    label_bl = LABEL_BLACKLIST.get(label_name, frozenset())

    for cand in candidates:
        token = cand["token"]

        if use_generic and token in GENERIC_BLACKLIST:
            generic_removed += 1
            continue

        if use_label and token in label_bl:
            label_removed += 1
            continue

        result.append(cand)

    return result, generic_removed, label_removed


# ─────────────────────────────────────────────────────────────
# Step F：寫入 sf_keyword_candidates
# ─────────────────────────────────────────────────────────────
def upsert_candidates(
    cur,
    mining_run: str,
    label_name: str,
    candidates: list[dict],
    example_map: dict[int, list[str]],
    batch_size: int = 200,
) -> int:
    """
    批次 upsert sf_keyword_candidates。
    ON CONFLICT (mining_run, label_name, token, candidate_type) DO UPDATE：
      每次重跑會更新分數，保留人工審閱結果（is_approved / review_note 不覆蓋）。
    """
    if not candidates:
        return 0

    sql = """
        INSERT INTO sf_keyword_candidates (
            mining_run, label_name, token, candidate_type,
            label_total_tf, label_doc_freq, global_df,
            discriminative_score, dominant_pos,
            example_comment_ids
        )
        VALUES %s
        ON CONFLICT (mining_run, label_name, token, candidate_type)
        DO UPDATE SET
            label_total_tf       = EXCLUDED.label_total_tf,
            label_doc_freq       = EXCLUDED.label_doc_freq,
            global_df            = EXCLUDED.global_df,
            discriminative_score = EXCLUDED.discriminative_score,
            dominant_pos         = EXCLUDED.dominant_pos,
            example_comment_ids  = EXCLUDED.example_comment_ids
            -- is_approved / review_note / reviewed_at 刻意不覆蓋
    """
    inserted = 0
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        values = []
        for cand in batch:
            term_id = cand["term_id"]
            example_ids = example_map.get(term_id, [])
            values.append((
                mining_run,
                label_name,
                cand["token"],
                "unigram",
                cand["label_total_tf"],
                cand["label_doc_freq"],
                cand["global_df"],
                float(cand["discriminative_score"]),
                cand.get("dominant_pos"),
                example_ids if example_ids else None,
            ))
        pgx.execute_values(cur, sql, values, page_size=batch_size)
        inserted += len(batch)
        logger.info(f"  已寫入 {min(start + batch_size, len(candidates))} / {len(candidates)} 筆")
    return inserted


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def run_mining(
    run_id: str,
    label_name: str,
    mining_run: str,
    corpus_id: int,
    pipeline_version: str,
    min_tf: int,
    min_df: int,
    top_k: int,
    use_pos_filter: bool,
    exclude_nb: bool,
    use_generic_blacklist: bool,
    use_label_blacklist: bool,
    dry_run: bool,
) -> None:

    logger.info("=" * 60)
    logger.info(f"run_id           : {run_id}")
    logger.info(f"label_name       : {label_name}")
    logger.info(f"mining_run       : {mining_run}")
    logger.info(f"corpus_id        : {corpus_id}")
    logger.info(f"pipeline_version : {pipeline_version}")
    logger.info(f"min_tf / min_df  : {min_tf} / {min_df}")
    logger.info(f"top_k            : {top_k}")
    logger.info(f"pos_filter       : {use_pos_filter}  exclude_nb={exclude_nb}")
    logger.info(f"generic_blacklist: {use_generic_blacklist}")
    logger.info(f"label_blacklist  : {use_label_blacklist}")
    logger.info(f"mode             : {'DRY-RUN' if dry_run else 'WRITE'}")
    logger.info("=" * 60)

    conn = get_conn()
    try:
        with conn.cursor() as cur:

            # ── 確認高信心評論數量 ───────────────────────────────
            n_docs = fetch_label_doc_count(cur, run_id, label_name)
            if n_docs == 0:
                logger.error(
                    f"sf_highconf_comments 中找不到 run_id='{run_id}', "
                    f"label_name='{label_name}' 的資料。"
                    f"請先執行 sf_01_filter_highconf.py。"
                )
                return
            logger.info(f"高信心評論數量 : {n_docs:,} 筆")

            # ── 計算 discriminative scores（DB 端聚合）───────────
            # 先抓 top_k * BUFFER 筆，預留 POS 過濾的空間
            fetch_limit = top_k * QUERY_BUFFER_FACTOR
            logger.info(f"計算 discriminative scores（fetch_limit={fetch_limit}）...")
            raw_candidates = compute_discriminative_scores(
                cur,
                run_id=run_id,
                label_name=label_name,
                corpus_id=corpus_id,
                pipeline_version=pipeline_version,
                min_tf=min_tf,
                min_df=min_df,
                min_token_len=2,
                fetch_limit=fetch_limit,
            )
            logger.info(f"通過 SQL 基礎過濾的 unigram 數量 : {len(raw_candidates)}")

            if not raw_candidates:
                logger.warning(
                    "未找到符合條件的候選詞。"
                    "建議降低 --min-tf 或 --min-df，或確認 corpus_id 是否正確。"
                )
                return

            # ── 統計（dry-run 主要看這裡）────────────────────────
            scores = [c["discriminative_score"] for c in raw_candidates]
            logger.info(f"discriminative_score 統計（前 {len(scores)} 筆）：")
            logger.info(f"  最大: {max(scores):.4f}")
            logger.info(f"  最小: {min(scores):.4f}")
            logger.info(f"  Top 10 tokens: {[c['token'] for c in raw_candidates[:10]]}")

            # ── 批次取 dominant_pos ──────────────────────────────
            logger.info("取得 dominant_pos...")
            all_tokens = [c["token"] for c in raw_candidates]
            pos_map = fetch_dominant_pos(
                cur,
                run_id=run_id,
                label_name=label_name,
                pipeline_version=pipeline_version,
                tokens=all_tokens,
            )
            logger.info(f"  找到詞性資訊的 token 數量: {len(pos_map)}")

            # ── 過濾統計追蹤 ────────────────────────────────────
            after_sql = len(raw_candidates)

            # ── 層次 1：POS 過濾 ────────────────────────────────
            after_pos_list, pos_removed = apply_pos_filter(
                raw_candidates, pos_map, use_pos_filter, exclude_nb
            )
            after_pos = len(after_pos_list)

            # ── 層次 2 & 3：Blacklist 過濾 ──────────────────────
            final_candidates_full, generic_removed, label_removed = apply_blacklist_filter(
                after_pos_list,
                label_name=label_name,
                use_generic=use_generic_blacklist,
                use_label=use_label_blacklist,
            )
            after_blacklist = len(final_candidates_full)

            # ── 取最終 top_k ─────────────────────────────────────
            final_candidates = final_candidates_full[:top_k]

            # ── 過濾統計報告 ─────────────────────────────────────
            logger.info("─" * 50)
            logger.info("過濾統計：")
            logger.info(f"  SQL 基礎過濾後       : {after_sql:4d} 筆")
            if use_pos_filter:
                nb_note = "（含 Nb 排除）" if exclude_nb else ""
                logger.info(f"  POS 過濾後           : {after_pos:4d} 筆  (-{pos_removed}{nb_note})")
            else:
                logger.info(f"  POS 過濾             : 已關閉")
            if use_generic_blacklist:
                logger.info(f"  通用 blacklist 後    : {after_pos - generic_removed:4d} 筆  (-{generic_removed})")
            else:
                logger.info(f"  通用 blacklist       : 已關閉")
            if use_label_blacklist:
                logger.info(f"  Label blacklist 後   : {after_blacklist:4d} 筆  (-{label_removed})")
            else:
                logger.info(f"  Label blacklist      : 已關閉")
            logger.info(f"  最終 top_{top_k:<4d}       : {len(final_candidates):4d} 筆")
            logger.info("─" * 50)

            if dry_run:
                logger.info("[DRY-RUN] 不寫入資料庫。")
                logger.info("[DRY-RUN] Top 20 候選詞預覽：")
                for i, c in enumerate(final_candidates[:20], 1):
                    pos_tag = pos_map.get(c["token"], "?")
                    logger.info(
                        f"  {i:3d}. {c['token']:<10s} "
                        f"disc={c['discriminative_score']:+.4f}  "
                        f"tf={c['label_total_tf']:4d}  "
                        f"df={c['label_doc_freq']:4d}  "
                        f"pos={pos_tag}"
                    )
                return

            # ── 批次取 example_comment_ids ───────────────────────
            logger.info("取得 example_comment_ids...")
            final_term_ids = [c["term_id"] for c in final_candidates]
            example_map = fetch_example_comment_ids(
                cur,
                run_id=run_id,
                label_name=label_name,
                corpus_id=corpus_id,
                term_ids=final_term_ids,
                n_examples=3,
            )
            logger.info(f"  取得 example 的 term 數量: {len(example_map)}")

            # ── 寫入 sf_keyword_candidates ───────────────────────
            logger.info("寫入 sf_keyword_candidates...")
            inserted = upsert_candidates(
                cur,
                mining_run=mining_run,
                label_name=label_name,
                candidates=final_candidates,
                example_map=example_map,
            )

            conn.commit()
            logger.info(f"完成：{inserted} 筆已寫入 sf_keyword_candidates（mining_run='{mining_run}'）")

    except Exception as e:
        conn.rollback()
        logger.error(f"發生錯誤，已 rollback：{e}")
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="SF Pipeline Step 2：挖掘 unigram 候選詞"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="對應 sf_highconf_comments.run_id，例如 'nli-seed-v1'",
    )
    parser.add_argument(
        "--label-name",
        required=True,
        choices=[
            "High_Arousal", "High_Novelty", "High_Repurchase_Intent",
            "Negative_Complaint", "Advertisement"
        ],
        help="要挖掘的 label",
    )
    parser.add_argument(
        "--mining-run",
        default=None,
        help=(
            "寫入 sf_keyword_candidates.mining_run 的 ID。"
            "預設自動生成：mine-v1-{label_slug}-tf{min_tf}-df{min_df}"
        ),
    )
    parser.add_argument(
        "--pipeline-version",
        required=True,
        help="對應 comment_tokens.pipeline_version，例如 'ckip-pre-v1'",
    )
    parser.add_argument(
        "--corpus-id",
        type=int,
        required=True,
        help="tfidf_corpus.corpus_id（global corpus 的 ID）",
    )
    parser.add_argument(
        "--min-tf",
        type=int,
        default=5,
        help="label 子語料中最低總詞頻（預設 5）",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=3,
        help="label 子語料中最低文件頻率（預設 3）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=150,
        help="最終寫入的候選詞數量（預設 150）",
    )
    parser.add_argument(
        "--no-pos-filter",
        action="store_true",
        default=False,
        help="停用 POS 白名單過濾（納入所有詞性，方便 debug）",
    )
    parser.add_argument(
        "--exclude-nb",
        action="store_true",
        default=False,
        help="從 POS 白名單中移除 Nb（專有名詞），用於過濾品牌名",
    )
    parser.add_argument(
        "--no-generic-blacklist",
        dest="use_generic_blacklist",
        action="store_false",
        default=True,
        help="關閉通用 blacklist 過濾（預設開啟）",
    )
    parser.add_argument(
        "--no-label-blacklist",
        dest="use_label_blacklist",
        action="store_false",
        default=True,
        help="關閉 label-specific blacklist 過濾（預設開啟）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="只印預覽，不寫入 DB（預設 True）",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="實際寫入 sf_keyword_candidates",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 自動生成 mining_run ID（若未指定）
    if args.mining_run:
        mining_run = args.mining_run
    else:
        label_slug = args.label_name.lower().replace("_", "-")
        mining_run = (
            f"mine-v1-{label_slug}"
            f"-tf{args.min_tf}"
            f"-df{args.min_df}"
        )
    logger.info(f"mining_run ID: {mining_run}")

    run_mining(
        run_id=args.run_id,
        label_name=args.label_name,
        mining_run=mining_run,
        corpus_id=args.corpus_id,
        pipeline_version=args.pipeline_version,
        min_tf=args.min_tf,
        min_df=args.min_df,
        top_k=args.top_k,
        use_pos_filter=not args.no_pos_filter,
        exclude_nb=args.exclude_nb,
        use_generic_blacklist=args.use_generic_blacklist,
        use_label_blacklist=args.use_label_blacklist,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
