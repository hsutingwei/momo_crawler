CREATE TABLE IF NOT EXISTS comment_semantic_scores (
    comment_id TEXT PRIMARY KEY,
    score_arousal FLOAT,
    score_novelty FLOAT,
    score_repurchase FLOAT,
    score_negative FLOAT,
    score_advertisement FLOAT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (comment_id) REFERENCES product_comments(comment_id)
);

CREATE INDEX IF NOT EXISTS idx_semantic_updated_at ON comment_semantic_scores(updated_at);
