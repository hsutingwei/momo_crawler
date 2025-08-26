# -*- coding: utf-8 -*-
"""
Embedding utilities for feature generation pipeline.
- Database connection and data fetching
- Text preprocessing and tokenization
- SIF weighting and PCA removal
- Output utilities
"""

import os
import re
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import psycopg2
import psycopg2.extras
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import PCA

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.database import DatabaseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_conn():
    """Get database connection using existing DatabaseConfig."""
    db_config = DatabaseConfig()
    return db_config.get_connection()


def light_clean(text: str) -> str:
    """Light text cleaning for raw comments."""
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)           # URL
    text = re.sub(r'<[^>]+>', ' ', text)                         # HTML
    text = re.sub(r'\s+', ' ', text).strip()                     # Multiple spaces
    text = re.sub(r'([!?。，！、…])\1{2,}', r'\1\1', text)        # Long repeated punctuation
    return text


def fetch_texts(conn, 
                text_source: str = 'norm',
                pipeline_version: Optional[str] = None,
                date_cutoff: Optional[str] = None,
                limit: Optional[int] = None) -> List[Tuple[str, int, datetime, str]]:
    """
    Fetch texts from database based on parameters.
    
    Returns:
        List of (comment_id, product_id, capture_time, text) tuples
    """
    if text_source == 'norm':
        if not pipeline_version:
            raise ValueError("pipeline_version required for text_source='norm'")
        
        sql = """
        SELECT 
            pc.comment_id,
            pc.product_id,
            pc.capture_time,
            COALESCE(ctn.norm_text, pc.comment_text) as text
        FROM product_comments pc
        LEFT JOIN comment_text_norm ctn 
            ON pc.comment_id = ctn.comment_id 
            AND ctn.pipeline_version = %(pipeline_version)s
        WHERE ctn.norm_text IS NOT NULL
        """
    else:  # raw
        sql = """
        SELECT 
            pc.comment_id,
            pc.product_id,
            pc.capture_time,
            pc.comment_text as text
        FROM product_comments pc
        WHERE pc.comment_text IS NOT NULL
        """
    
    # Add date cutoff if specified
    if date_cutoff:
        sql += " AND pc.capture_time <= %(date_cutoff)s"
    
    sql += " ORDER BY pc.capture_time"
    
    if limit:
        sql += " LIMIT %(limit)s"
    
    params = {
        'pipeline_version': pipeline_version,
        'date_cutoff': date_cutoff,
        'limit': limit
    }
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    # Clean raw texts if needed
    if text_source == 'raw':
        rows = [(comment_id, product_id, capture_time, light_clean(text)) 
                for comment_id, product_id, capture_time, text in rows]
    
    logger.info(f"Fetched {len(rows)} texts from database")
    return rows


def fetch_tokens(conn,
                 pipeline_version: str,
                 date_cutoff: Optional[str] = None,
                 limit: Optional[int] = None) -> List[Tuple[str, int, datetime, List[str]]]:
    """
    Fetch CKIP tokens from comment_tokens table.
    
    Returns:
        List of (comment_id, product_id, capture_time, tokens) tuples
    """
    sql = """
    SELECT 
        pc.comment_id,
        pc.product_id,
        pc.capture_time,
        ARRAY_AGG(ct.token ORDER BY ct.token_order) as tokens
    FROM product_comments pc
    JOIN comment_tokens ct 
        ON pc.comment_id = ct.comment_id 
        AND ct.pipeline_version = %(pipeline_version)s
    """
    
    if date_cutoff:
        sql += " WHERE pc.capture_time <= %(date_cutoff)s"
    
    sql += """
    GROUP BY pc.comment_id, pc.product_id, pc.capture_time
    ORDER BY pc.capture_time
    """
    
    if limit:
        sql += " LIMIT %(limit)s"
    
    params = {
        'pipeline_version': pipeline_version,
        'date_cutoff': date_cutoff,
        'limit': limit
    }
    
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    
    logger.info(f"Fetched {len(rows)} tokenized comments from database")
    return rows


def load_idf_from_db(conn, pipeline_version: str) -> Dict[str, float]:
    """
    Load IDF values from tfidf_scores view.
    
    Returns:
        Dict mapping token to IDF value
    """
    sql = """
    SELECT DISTINCT
        ts.token,
        ts.idf
    FROM tfidf_scores ts
    JOIN comment_tokens ct 
        ON ts.comment_id = ct.comment_id 
        AND ct.pipeline_version = %(pipeline_version)s
    """
    
    with conn.cursor() as cur:
        cur.execute(sql, {'pipeline_version': pipeline_version})
        rows = cur.fetchall()
    
    idf_dict = {token: idf for token, idf in rows}
    logger.info(f"Loaded {len(idf_dict)} IDF values from database")
    return idf_dict


def sif_weight(tokens: List[str], idf_dict: Dict[str, float], alpha: float = 1e-3) -> np.ndarray:
    """
    Calculate SIF weights for tokens.
    
    Args:
        tokens: List of tokens
        idf_dict: Dict mapping token to IDF value
        alpha: SIF alpha parameter
    
    Returns:
        Array of weights for each token
    """
    if not tokens:
        return np.array([])
    
    # Calculate IDF for each token
    idfs = []
    for token in tokens:
        idf = idf_dict.get(token, 0.0)  # Default to 0 if not found
        idfs.append(idf)
    
    idfs = np.array(idfs)
    
    # SIF weight formula: a / (a + p(w))
    # where p(w) = exp(idf) / sum(exp(idf))
    if np.sum(np.exp(idfs)) > 0:
        p_w = np.exp(idfs) / np.sum(np.exp(idfs))
        weights = alpha / (alpha + p_w)
    else:
        weights = np.ones(len(tokens))
    
    return weights


def remove_pc(X: np.ndarray, npc: int = 1) -> np.ndarray:
    """
    Remove first n principal components from embedding matrix.
    
    Args:
        X: Embedding matrix (N, dim)
        npc: Number of principal components to remove
    
    Returns:
        Embedding matrix with PCs removed
    """
    if npc <= 0:
        return X
    
    pca = PCA(n_components=npc)
    pca.fit(X)
    
    # Remove the components
    X_centered = X - np.mean(X, axis=0)
    X_removed = X_centered - pca.transform(X_centered) @ pca.components_
    
    logger.info(f"Removed {npc} principal components from embedding matrix")
    return X_removed


def save_npz_compressed(path: str, arr: np.ndarray, name: str = 'arr_0'):
    """Save numpy array as compressed npz file."""
    np.savez_compressed(path, **{name: arr})
    logger.info(f"Saved {arr.shape} array to {path}")


def save_csv(path: str, rows: List[Tuple], columns: List[str]):
    """Save rows as CSV file."""
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False, encoding='utf-8')
    logger.info(f"Saved {len(rows)} rows to {path}")


def preview_rows(comment_ids: List[str], texts: List[str], vectors: np.ndarray, 
                n_preview: int = 20) -> List[Tuple]:
    """
    Create preview rows for sample_preview.csv.
    
    Returns:
        List of (comment_id, text_head, vec_head) tuples
    """
    preview_data = []
    
    for i in range(min(n_preview, len(comment_ids))):
        comment_id = comment_ids[i]
        text = texts[i] if i < len(texts) else ""
        vector = vectors[i] if i < len(vectors) else np.array([])
        
        # Truncate text to first 100 chars
        text_head = text[:100] + "..." if len(text) > 100 else text
        
        # Get first 5 dimensions of vector
        vec_head = vector[:5].tolist() if len(vector) >= 5 else vector.tolist()
        
        preview_data.append((comment_id, text_head, str(vec_head)))
    
    return preview_data


def create_manifest(run_name: str, start_time: datetime, end_time: datetime,
                   device: str, mode: str, date_cutoff: Optional[str],
                   pipeline_version: Optional[str], text_source: Optional[str],
                   embed_model: Optional[str], pooling: Optional[str],
                   word_emb: Optional[str], idf_source: Optional[str],
                   sif_alpha: Optional[float], sif_remove_pc: int,
                   n_samples: int, dim: int, batch_size: int,
                   files: Dict[str, str], notes: str = "") -> Dict:
    """Create manifest dictionary for saving as JSON."""
    return {
        "run_name": run_name,
        "started_at": start_time.isoformat(),
        "finished_at": end_time.isoformat(),
        "device": device,
        "mode": mode,
        "date_cutoff": date_cutoff,
        "pipeline_version": pipeline_version,
        "text_source": text_source,
        "embed_model": embed_model,
        "pooling": pooling,
        "word_emb": word_emb,
        "idf_source": idf_source,
        "sif_alpha": sif_alpha,
        "sif_remove_pc": sif_remove_pc,
        "n_samples": n_samples,
        "dim": dim,
        "batch_size": batch_size,
        "files": files,
        "notes": notes
    }


def save_manifest(path: str, manifest: Dict):
    """Save manifest as JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved manifest to {path}")


def print_summary(n_samples: int, dim: int, vectors: np.ndarray, 
                 oov_ratio: Optional[float] = None, idf_coverage: Optional[float] = None,
                 processing_time: float = 0.0):
    """Print processing summary to console."""
    print(f"\n{'='*50}")
    print(f"EMBEDDING GENERATION SUMMARY")
    print(f"{'='*50}")
    print(f"Samples processed: {n_samples:,}")
    print(f"Vector dimension: {dim}")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Speed: {n_samples/processing_time:.1f} samples/sec" if processing_time > 0 else "")
    
    if vectors is not None and len(vectors) > 0:
        print(f"Vector statistics:")
        print(f"  Mean: {np.mean(vectors):.4f}")
        print(f"  Std:  {np.std(vectors):.4f}")
        print(f"  Min:  {np.min(vectors):.4f}")
        print(f"  Max:  {np.max(vectors):.4f}")
    
    if oov_ratio is not None:
        print(f"OOV ratio: {oov_ratio:.2%}")
    
    if idf_coverage is not None:
        print(f"IDF coverage: {idf_coverage:.2%}")
    
    print(f"{'='*50}\n")
