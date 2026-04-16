-- ============================================================
-- SF Label Prototype Expansion Pipeline — Phase 1 Migration
-- ============================================================
-- 目的：建立 SF 半自動擴增流程所需的 3 張新表
-- 不異動：comment_semantic_scores、ml_runs 主架構
--
-- 執行方式：
--   psql -U <user> -d momo_crawler -f Model/sf_pipeline_v1_migration.sql
--
-- 安全性：
--   - 全部使用 CREATE TABLE IF NOT EXISTS，可重複執行
--   - 若需要重建（清空重來），手動執行文末的 RESET SECTION
-- ============================================================


-- ─────────────────────────────────────────────────────────────
-- TABLE 1: sf_prototype_texts
-- 存放每個 label 的 hypothesis prototype 文字
-- 先建這張：sf_comment_scores（第二階段）會用 proto_id FK 引用它
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sf_prototype_texts (
    proto_id        SERIAL          PRIMARY KEY,

    -- label 名稱，對應 comment_semantic_scores 的欄位命名慣例
    -- 合法值：High_Arousal / High_Novelty / High_Repurchase_Intent
    --         Negative_Complaint / Advertisement
    label_name      TEXT            NOT NULL,

    -- 版本識別：'seed' = 初始假設標籤；'v1'/'v2' = 擴增後版本
    proto_version   TEXT            NOT NULL,

    -- 實際傳入 NLI 模型的 hypothesis 文字（中文短語）
    proto_text      TEXT            NOT NULL,

    -- 來源：seed=人工初設, semi_auto=本流程擴增候選, manual=人工撰寫
    source          TEXT            NOT NULL DEFAULT 'seed'
                    CHECK (source IN ('seed', 'semi_auto', 'manual')),

    -- 是否為目前使用中的 prototype（可停用舊版本而不刪除）
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,

    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- 同一個 label + 版本 + 文字只能存在一次（防止重複插入）
    UNIQUE (label_name, proto_version, proto_text)
);

CREATE INDEX IF NOT EXISTS idx_sf_proto_label_active
    ON sf_prototype_texts (label_name, is_active);

COMMENT ON TABLE sf_prototype_texts IS
    'SF 擴增流程：每個 label 的 NLI hypothesis prototype 文字，支援版本控管';
COMMENT ON COLUMN sf_prototype_texts.proto_version IS
    '''seed'' = 初始版，''v1''/''v2'' = 擴增後版本';
COMMENT ON COLUMN sf_prototype_texts.is_active IS
    'FALSE = 停用（不參與計分）但保留歷史記錄';


-- ─────────────────────────────────────────────────────────────
-- 插入 5 個 Seed Prototypes
-- 對應 comment_semantic_scores 的 5 個原始 hypothesis 標籤
-- ─────────────────────────────────────────────────────────────
INSERT INTO sf_prototype_texts (label_name, proto_version, proto_text, source)
VALUES
    ('High_Arousal',           'seed', '驚豔、激動、太神了',         'seed'),
    ('High_Novelty',           'seed', '新奇、初次體驗、相見恨晚',   'seed'),
    ('High_Repurchase_Intent', 'seed', '回購意願高、忠實粉絲',       'seed'),
    ('Negative_Complaint',     'seed', '憤怒、失望、反推',           'seed'),
    ('Advertisement',          'seed', '業配、廣告、湊字數',         'seed')
ON CONFLICT (label_name, proto_version, proto_text) DO NOTHING;


-- ─────────────────────────────────────────────────────────────
-- TABLE 2: sf_highconf_comments
-- 每一輪篩選結果：哪些 comment_id 被認定為某 label 的高信心評論
--
-- run_id 命名慣例：'nli-seed-v1'（seed 版打分，第一輪篩選）
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sf_highconf_comments (
    -- 篩選輪次 ID，例如 'nli-seed-v1' 或 'nli-proto-v2'
    -- 與使用的分數版本對應（seed 版 = 讀 comment_semantic_scores）
    run_id          TEXT            NOT NULL,

    -- 對應的 label 名稱
    label_name      TEXT            NOT NULL,

    -- 評論 ID，對齊 product_comments.comment_id VARCHAR(100)
    -- TEXT 在 PostgreSQL 中可與 VARCHAR(100) 的 FK 相容
    comment_id      VARCHAR(100)    NOT NULL,

    -- 篩選用的聚合分數
    -- seed 階段 = comment_semantic_scores 的對應 score_* 欄位值
    -- 第二階段（多 proto）= MAX(raw_score) across prototypes
    agg_score       REAL            NOT NULL,

    -- 記錄本輪使用的篩選門檻（方便後續復現與比較）
    score_threshold REAL            NOT NULL,

    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (run_id, label_name, comment_id),
    FOREIGN KEY (comment_id) REFERENCES product_comments(comment_id)
);

-- 主要查詢：某輪 + 某 label
CREATE INDEX IF NOT EXISTS idx_sf_highconf_run_label
    ON sf_highconf_comments (run_id, label_name);

-- 分數排序（挖掘時常用 ORDER BY agg_score DESC）
CREATE INDEX IF NOT EXISTS idx_sf_highconf_score_desc
    ON sf_highconf_comments (run_id, label_name, agg_score DESC);

COMMENT ON TABLE sf_highconf_comments IS
    'SF 擴增流程：每輪篩選出的高信心評論集合，是 keyword mining 的輸入';
COMMENT ON COLUMN sf_highconf_comments.run_id IS
    '對應使用哪個版本的分數篩選，例如 ''nli-seed-v1''';
COMMENT ON COLUMN sf_highconf_comments.score_threshold IS
    '本輪篩選使用的門檻值，紀錄用（實際過濾邏輯在 Python script 中執行）';


-- ─────────────────────────────────────────────────────────────
-- TABLE 3: sf_keyword_candidates
-- 存放 unigram / bigram 候選詞、差異化分數、人工審閱結果
--
-- mining_run 命名慣例：'mine-v1-novelty-0.70'
--   = 第一次挖掘，對 High_Novelty，使用 threshold=0.70 的高信心集合
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sf_keyword_candidates (
    candidate_id            SERIAL          PRIMARY KEY,

    -- 挖掘版本 ID（對應使用哪輪 sf_highconf_comments）
    mining_run              TEXT            NOT NULL,

    -- 對應 label
    label_name              TEXT            NOT NULL,

    -- 候選詞文字
    -- unigram：單一 token，例如 '相見恨晚'
    -- bigram：兩個相鄰 token 串接，例如 '初次體驗'
    token                   TEXT            NOT NULL,

    -- 候選類型
    candidate_type          TEXT            NOT NULL DEFAULT 'unigram'
                            CHECK (candidate_type IN ('unigram', 'bigram')),

    -- ── 頻率統計 ───────────────────────────────────────────────
    -- label 高信心子語料中的總詞頻（sum of tf）
    label_total_tf          INTEGER,

    -- label 子語料中出現該詞的文件數
    label_doc_freq          INTEGER,

    -- 全語料中出現該詞的文件數（來自 tfidf_doc_freq）
    global_df               INTEGER,

    -- ── 差異化分數 ──────────────────────────────────────────────
    -- log( label內頻率 / global文件頻率 )，越高越具辨識性
    discriminative_score    REAL,

    -- 主要詞性（unigram: 'Na', 'VH' 等；bigram: 'Na+VH' 格式）
    dominant_pos            TEXT,

    -- ── 代表性例句 ──────────────────────────────────────────────
    -- 3 筆包含此 token 的高信心評論 comment_id（VARCHAR(100) array）
    -- 查原文時 JOIN product_comments ON comment_id = ANY(example_comment_ids)
    example_comment_ids     VARCHAR(100)[],

    -- ── 人工審閱欄位 ────────────────────────────────────────────
    -- NULL = 尚未審閱, TRUE = 通過納入 prototype, FALSE = 拒絕
    is_approved             BOOLEAN,

    -- 人工備註（例如：'太廣泛'、'好，加入 v1'、'品牌名排除'）
    review_note             TEXT,

    -- 審閱時間（用於追蹤哪些已被審閱）
    reviewed_at             TIMESTAMPTZ,

    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- 同一輪挖掘中，同一 label + token + 類型只存一筆
    UNIQUE (mining_run, label_name, token, candidate_type)
);

-- 主要查詢：某次挖掘 + 某 label
CREATE INDEX IF NOT EXISTS idx_sf_kw_mining_label
    ON sf_keyword_candidates (mining_run, label_name);

-- 審閱工作流：找出尚未審閱 / 已通過的候選詞
CREATE INDEX IF NOT EXISTS idx_sf_kw_approved
    ON sf_keyword_candidates (mining_run, label_name, is_approved);

-- 按差異化分數排序（挖掘結果驗證、匯出 Excel 時常用）
CREATE INDEX IF NOT EXISTS idx_sf_kw_disc_score
    ON sf_keyword_candidates (mining_run, label_name, discriminative_score DESC NULLS LAST);

COMMENT ON TABLE sf_keyword_candidates IS
    'SF 擴增流程：候選 keyword / phrase 及人工審閱結果，是組裝新 prototype 的原材料';
COMMENT ON COLUMN sf_keyword_candidates.discriminative_score IS
    'log( label子語料詞頻/label文件數 / global_df/total_docs )，越高越具辨識性';
COMMENT ON COLUMN sf_keyword_candidates.example_comment_ids IS
    '3筆代表性評論的 comment_id（VARCHAR(100) array），JOIN product_comments 取原文';


-- ============================================================
-- 確認建立結果
-- ============================================================
SELECT
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size
FROM pg_tables
WHERE tablename IN ('sf_prototype_texts', 'sf_highconf_comments', 'sf_keyword_candidates')
  AND schemaname = 'public'
ORDER BY tablename;


-- ============================================================
-- [RESET SECTION] 若需要完全重建，手動解除下方註解後執行
-- 警告：會清除所有已寫入的資料
-- ============================================================
-- DROP TABLE IF EXISTS sf_keyword_candidates CASCADE;
-- DROP TABLE IF EXISTS sf_highconf_comments CASCADE;
-- DROP TABLE IF EXISTS sf_prototype_texts CASCADE;
