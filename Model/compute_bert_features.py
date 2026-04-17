import os
import sys
import psycopg2
import psycopg2.extras
import torch
from transformers import pipeline
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.database import DatabaseConfig

def get_db_connection():
    db = DatabaseConfig()
    return db.get_connection()

def init_db(conn):
    """Initialize the semantic scores table."""
    with open(os.path.join(os.path.dirname(__file__), 'create_bert_table.sql'), 'r', encoding='utf-8') as f:
        ddl = f.read()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    print("Initialized database table.")

def fetch_pending_comments(conn, limit=1000, recompute=False, offset=0):
    """Fetch comments that haven't been processed yet (or all, if recompute=True)."""
    if recompute:
        sql = """
        SELECT comment_id, comment_text
        FROM product_comments
        WHERE comment_text IS NOT NULL
          AND length(comment_text) > 5
        ORDER BY comment_id
        LIMIT %s OFFSET %s
        """
        params = (limit, offset)
    else:
        sql = """
        SELECT pc.comment_id, pc.comment_text
        FROM product_comments pc
        LEFT JOIN comment_semantic_scores css ON pc.comment_id = css.comment_id
        WHERE css.comment_id IS NULL
          AND pc.comment_text IS NOT NULL
          AND length(pc.comment_text) > 5
        LIMIT %s
        """
        params = (limit,)
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        cur.execute(sql, params)
        return cur.fetchall()

def save_scores(conn, results):
    """Batch insert/update scores."""
    sql = """
    INSERT INTO comment_semantic_scores 
    (comment_id, score_arousal, score_novelty, score_repurchase, score_negative, score_advertisement)
    VALUES %s
    ON CONFLICT (comment_id) DO UPDATE SET
        score_arousal = EXCLUDED.score_arousal,
        score_novelty = EXCLUDED.score_novelty,
        score_repurchase = EXCLUDED.score_repurchase,
        score_negative = EXCLUDED.score_negative,
        score_advertisement = EXCLUDED.score_advertisement,
        updated_at = CURRENT_TIMESTAMP
    """
    values = []
    for r in results:
        values.append((
            r['comment_id'],
            r['scores']['High_Arousal'],
            r['scores']['High_Novelty'],
            r['scores']['High_Repurchase_Intent'],
            r['scores']['Negative_Complaint'],
            r['scores']['Advertisement']
        ))
    
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, values)
    conn.commit()

def main():
    parser = argparse.ArgumentParser(description="Compute BERT Zero-Shot features for comments.")
    parser.add_argument("--limit", type=int, default=100, help="Number of comments to process per run (for testing)")
    parser.add_argument("--batch_size", type=int, default=16, help="Inference batch size")
    parser.add_argument("--model", type=str, default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", help="Hugging Face model name")
    parser.add_argument("--device", type=int, default=-1, help="Device ID (-1 for CPU, 0 for GPU)")
    parser.add_argument("--recompute", action="store_true", default=False,
                        help="重新計算所有評論（包含已有分數的），用於更新 hypothesis 後的全量重跑")
    args = parser.parse_args()

    # Check for GPU
    if torch.cuda.is_available() and args.device == -1:
        print("CUDA is available. You might want to use --device 0")
    
    device = args.device if torch.cuda.is_available() or args.device == -1 else -1
    print(f"Using device: {device}")

    # Initialize Pipeline
    print(f"Loading model: {args.model}...")
    classifier = pipeline("zero-shot-classification", model=args.model, device=device)
    
    # Candidate Labels (Semantic Classes)
    candidate_labels = [
        "High_Arousal", 
        "High_Novelty", 
        "High_Repurchase_Intent", 
        "Negative_Complaint", 
        "Advertisement"
    ]
    
    # Chinese Translations for Hypothesis Template (Optional, but model is multilingual)
    # Ideally we feed Chinese labels if the text is Chinese, or English if the model is aligned.
    # mDeBERTa-v3-base-mnli-xnli works well with English labels on Chinese text.
    # But let's use Chinese labels for better alignment if needed. 
    # For now, we stick to English keys for the database, but we can map them.
    
    # Mapping for inference (if we want to use Chinese labels)
    label_map = {
        "High_Arousal": "驚豔、驚喜、不可思議、太神了、厲害、無敵、震撼、驚人",
        "High_Novelty": "新奇、初次體驗、相見恨晚、特別、不同、難得、發現、神奇",
        "High_Repurchase_Intent": "回購、長期、固定、首選、忠實、囤貨、會再買、一直買",
        "Negative_Complaint": "憤怒、失望、反推、客服、傻眼、過期、嚴重、可惜、問題、退貨",
        "Advertisement": "代言、代言人、官網、網頁、網評、風評、老牌子",
    }

    labels_zh = list(label_map.values())
    
    conn = get_db_connection()
    try:
        init_db(conn)

        if args.recompute:
            print("--recompute 模式：將重新計算所有評論的分數（ON CONFLICT DO UPDATE）")

        processed_total = 0
        offset = 0
        while True:
            comments = fetch_pending_comments(conn, limit=args.limit, recompute=args.recompute, offset=offset)
            if not comments:
                print("No pending comments found.")
                break

            print(f"Processing {len(comments)} comments...")

            # Prepare batch
            texts = [c['comment_text'] for c in comments]
            ids = [c['comment_id'] for c in comments]

            # Inference
            # We use the Chinese labels for inference
            outputs = classifier(texts, labels_zh, batch_size=args.batch_size, multi_label=True)

            results = []
            for i, output in enumerate(outputs):
                # Map back to English keys
                scores = {k: 0.0 for k in candidate_labels}
                for label, score in zip(output['labels'], output['scores']):
                    # Find which English key this Chinese label corresponds to
                    for eng_key, zh_val in label_map.items():
                        if zh_val == label:
                            scores[eng_key] = score
                            break

                results.append({
                    'comment_id': ids[i],
                    'scores': scores
                })

            save_scores(conn, results)
            processed_total += len(results)
            offset += len(comments)
            print(f"Saved {len(results)} scores. (累計 {processed_total} 筆)")

            # If we fetched fewer than limit, we are done
            if len(comments) < args.limit:
                break
                
    finally:
        conn.close()

if __name__ == "__main__":
    main()
