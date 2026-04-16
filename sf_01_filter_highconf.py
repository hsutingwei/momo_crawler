# sf_01_filter_highconf.py
# -*- coding: utf-8 -*-
"""
SF Pipeline Step 1：從 comment_semantic_scores 篩選高信心評論

功能：
  - 讀取既有 comment_semantic_scores（seed NLI 打分結果）
  - 套用 label threshold + 排除規則
  - 將符合條件的 comment_id 寫入 sf_highconf_comments

使用方式：
  # Dry-run（只看數量與分數分布，不寫 DB）
  python sf_01_filter_highconf.py --run-id nli-seed-v1 --label High_Novelty --dry-run

  # 實際寫入
  python sf_01_filter_highconf.py --run-id nli-seed-v1 --label High_Novelty

  # 調整門檻
  python sf_01_filter_highconf.py --run-id nli-seed-v1 --label High_Novelty --threshold 0.65

  # 批次跑所有 5 個 label（每個使用各自的預設門檻）
  python sf_01_filter_highconf.py --run-id nli-seed-v1 --all-labels

依賴：
  pip install psycopg2-binary python-dotenv
"""

import os
import sys
import argparse
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.database import DatabaseConfig  # type: ignore

import psycopg2  # type: ignore
import psycopg2.extras as pgx  # type: ignore


# ─────────────────────────────────────────────────────────────
# 每個 label 的預設設定
# source_col   : comment_semantic_scores 中對應的欄位名稱
# threshold    : 預設高信心門檻（可被 --threshold 覆蓋）
# exclude      : 其他 label 必須低於此值才納入（避免混合信號）
# ─────────────────────────────────────────────────────────────
LABEL_CONFIG: dict = {
    "High_Arousal": {
        "source_col": "score_arousal",
        "threshold":  0.70,
        "exclude": {
            "score_advertisement": 0.40,
            "score_negative":      0.35,
        },
    },
    "High_Novelty": {
        "source_col": "score_novelty",
        "threshold":  0.70,
        "exclude": {
            "score_advertisement": 0.40,
            "score_negative":      0.35,
        },
    },
    "High_Repurchase_Intent": {
        "source_col": "score_repurchase",
        "threshold":  0.70,
        "exclude": {
            "score_advertisement": 0.40,
        },
    },
    "Negative_Complaint": {
        "source_col": "score_negative",
        "threshold":  0.65,
        "exclude": {
            "score_advertisement": 0.45,
        },
    },
    "Advertisement": {
        "source_col": "score_advertisement",
        "threshold":  0.65,
        "exclude": {},
    },
}


# ─────────────────────────────────────────────────────────────
# Logging 設定
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
    """
    取得 psycopg2 連線。
    關閉 autocommit，讓批次寫入可以在單一 transaction 內完成，
    發生錯誤時整批 rollback，不留下半寫狀態。
    """
    db_cfg = DatabaseConfig()
    conn = db_cfg.get_connection()
    conn.autocommit = False  # 覆蓋 DatabaseConfig 的 AUTOCOMMIT 設定
    return conn


# ─────────────────────────────────────────────────────────────
# 查詢與統計
# ─────────────────────────────────────────────────────────────
def fetch_stats(cur, source_col: str, threshold: float, exclude: dict) -> dict:
    """
    只查詢統計數字（dry-run 與正式執行前的確認都用這個）。
    回傳：{ count, avg_score, min_score, max_score }
    """
    # 動態組出排除條件
    exclude_clauses = " ".join(
        f"AND {col} < {val}" for col, val in exclude.items()
    )

    sql = f"""
        SELECT
            COUNT(*)            AS cnt,
            AVG({source_col})   AS avg_score,
            MIN({source_col})   AS min_score,
            MAX({source_col})   AS max_score
        FROM comment_semantic_scores
        WHERE {source_col} >= %(threshold)s
          {exclude_clauses}
    """
    cur.execute(sql, {"threshold": threshold})
    row = cur.fetchone()
    return {
        "count":     row[0],
        "avg_score": float(row[1]) if row[1] is not None else 0.0,
        "min_score": float(row[2]) if row[2] is not None else 0.0,
        "max_score": float(row[3]) if row[3] is not None else 0.0,
    }


def fetch_highconf_rows(cur, source_col: str, threshold: float, exclude: dict) -> list:
    """
    取出符合條件的 (comment_id, score) 列表，供後續寫入。
    """
    exclude_clauses = " ".join(
        f"AND {col} < {val}" for col, val in exclude.items()
    )

    sql = f"""
        SELECT comment_id, {source_col} AS score
        FROM comment_semantic_scores
        WHERE {source_col} >= %(threshold)s
          {exclude_clauses}
        ORDER BY {source_col} DESC
    """
    cur.execute(sql, {"threshold": threshold})
    return cur.fetchall()


# ─────────────────────────────────────────────────────────────
# 寫入
# ─────────────────────────────────────────────────────────────
def upsert_highconf(
    cur,
    run_id: str,
    label_name: str,
    rows: list,
    threshold: float,
    batch_size: int = 500,
) -> int:
    """
    批次 upsert 至 sf_highconf_comments。
    使用 ON CONFLICT DO NOTHING：
      - 同一 (run_id, label_name, comment_id) 已存在時跳過
      - 適合重複執行（idempotent）

    回傳實際寫入的筆數。
    """
    if not rows:
        return 0

    inserted = 0
    sql = """
        INSERT INTO sf_highconf_comments
            (run_id, label_name, comment_id, agg_score, score_threshold)
        VALUES %s
        ON CONFLICT (run_id, label_name, comment_id) DO NOTHING
    """

    # 分批次寫入，避免單次 execute 過大
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        values = [
            (run_id, label_name, str(comment_id), float(score), threshold)
            for comment_id, score in batch
        ]
        pgx.execute_values(cur, sql, values, page_size=batch_size)
        inserted += len(batch)
        logger.info(f"  已處理 {min(start + batch_size, len(rows))} / {len(rows)} 筆")

    return inserted


# ─────────────────────────────────────────────────────────────
# 單一 label 的完整流程
# ─────────────────────────────────────────────────────────────
def run_label(
    run_id: str,
    label_name: str,
    threshold: Optional[float],
    dry_run: bool,
) -> None:
    """
    篩選並寫入單一 label 的高信心評論。
    """
    if label_name not in LABEL_CONFIG:
        raise ValueError(
            f"未知的 label_name: '{label_name}'。"
            f"合法值：{list(LABEL_CONFIG.keys())}"
        )

    cfg = LABEL_CONFIG[label_name]
    source_col   = cfg["source_col"]
    eff_threshold = threshold if threshold is not None else cfg["threshold"]
    exclude      = cfg["exclude"]

    logger.info("=" * 60)
    logger.info(f"label      : {label_name}")
    logger.info(f"source_col : {source_col}")
    logger.info(f"threshold  : {eff_threshold}")
    logger.info(f"exclude    : {exclude}")
    logger.info(f"run_id     : {run_id}")
    logger.info(f"mode       : {'DRY-RUN' if dry_run else 'WRITE'}")
    logger.info("=" * 60)

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # ── 統計（dry-run 或寫入前確認都執行）──────────────
            stats = fetch_stats(cur, source_col, eff_threshold, exclude)
            logger.info(f"符合條件筆數 : {stats['count']:,}")
            logger.info(f"平均分數     : {stats['avg_score']:.4f}")
            logger.info(f"最小分數     : {stats['min_score']:.4f}")
            logger.info(f"最大分數     : {stats['max_score']:.4f}")

            if stats["count"] == 0:
                logger.warning("符合條件的評論為 0 筆，請檢查 threshold 是否過高或資料來源是否正確。")
                return

            # ── 合理性警告 ──────────────────────────────────────
            if stats["count"] < 100:
                logger.warning(
                    f"筆數 {stats['count']} 低於 100，建議降低 --threshold 或檢查資料。"
                )
            if stats["count"] > 20000:
                logger.warning(
                    f"筆數 {stats['count']} 超過 20,000，建議提高 --threshold 以確保純度。"
                )

            # ── dry-run 到此結束 ────────────────────────────────
            if dry_run:
                logger.info("[DRY-RUN] 不寫入資料庫。使用 --no-dry-run 或移除 --dry-run 執行寫入。")
                return

            # ── 正式寫入 ────────────────────────────────────────
            logger.info("開始讀取符合條件的評論...")
            rows = fetch_highconf_rows(cur, source_col, eff_threshold, exclude)
            logger.info(f"讀取完成，共 {len(rows):,} 筆，開始寫入 sf_highconf_comments...")

            inserted = upsert_highconf(
                cur, run_id, label_name, rows, eff_threshold
            )

            conn.commit()
            logger.info(f"寫入完成：{inserted:,} 筆已寫入（ON CONFLICT DO NOTHING，重複筆數自動跳過）")

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
        description="SF Pipeline Step 1：從 comment_semantic_scores 篩選高信心評論"
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="篩選輪次 ID，例如 'nli-seed-v1'（會存入 sf_highconf_comments.run_id）",
    )
    parser.add_argument(
        "--label",
        choices=list(LABEL_CONFIG.keys()),
        help="要篩選的 label 名稱。與 --all-labels 二選一",
    )
    parser.add_argument(
        "--all-labels",
        action="store_true",
        help="批次跑全部 5 個 label（使用各自預設門檻）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="覆蓋預設門檻值（僅在指定單一 --label 時有效）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="只印統計數字，不寫入 DB（預設 True）",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="實際寫入 sf_highconf_comments",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 互斥檢查
    if args.all_labels and args.label:
        logger.error("--label 和 --all-labels 不能同時使用")
        sys.exit(1)
    if not args.all_labels and not args.label:
        logger.error("請指定 --label <名稱> 或使用 --all-labels")
        sys.exit(1)
    if args.all_labels and args.threshold is not None:
        logger.warning("--threshold 在 --all-labels 模式下無效，各 label 使用各自預設門檻")

    # 決定要跑哪些 label
    labels_to_run = list(LABEL_CONFIG.keys()) if args.all_labels else [args.label]

    for label in labels_to_run:
        try:
            run_label(
                run_id=args.run_id,
                label_name=label,
                threshold=args.threshold if not args.all_labels else None,
                dry_run=args.dry_run,
            )
        except Exception as e:
            logger.error(f"處理 {label} 時失敗：{e}")
            if not args.all_labels:
                sys.exit(1)
            # all-labels 模式：繼續處理下一個 label

    logger.info("全部完成。")


if __name__ == "__main__":
    main()
