# -*- coding: utf-8 -*-
"""
Embedding feature generation pipeline.
Supports both sentence-level and word-level embeddings.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from emb_utils import (
    get_db_conn, fetch_texts, fetch_tokens, load_idf_from_db,
    sif_weight, remove_pc, save_npz_compressed, save_csv,
    preview_rows, create_manifest, save_manifest, print_summary
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sentence_transformer(model_name: str, device: str = 'auto'):
    """Load sentence transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Loading sentence transformer: {model_name}")
        model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded on device: {device}")
        return model, device
    except ImportError:
        raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")


def load_word_embeddings(emb_type: str, emb_path: str):
    """Load word embeddings (fastText or word2vec)."""
    if emb_type == 'fasttext':
        try:
            import fasttext
            logger.info(f"Loading fastText model: {emb_path}")
            model = fasttext.load_model(emb_path)
            return model
        except ImportError:
            raise ImportError("fasttext not installed. Run: pip install fasttext")
    
    elif emb_type == 'word2vec':
        try:
            from gensim.models import KeyedVectors
            logger.info(f"Loading Word2Vec model: {emb_path}")
            model = KeyedVectors.load_word2vec_format(emb_path, binary=True)
            return model
        except ImportError:
            raise ImportError("gensim not installed. Run: pip install gensim")
    
    else:
        raise ValueError(f"Unsupported word embedding type: {emb_type}")


def encode_sentences(model, texts: List[str], batch_size: int = 32, 
                    pooling: str = 'mean', fp16: bool = False) -> np.ndarray:
    """Encode sentences using sentence transformer."""
    logger.info(f"Encoding {len(texts)} sentences with batch_size={batch_size}")
    
    # Filter empty texts
    valid_texts = []
    valid_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts.append(text.strip())
            valid_indices.append(i)
    
    if not valid_texts:
        logger.warning("No valid texts found")
        return np.zeros((len(texts), model.get_sentence_embedding_dimension()))
    
    # Encode in batches with progress tracking
    embeddings = []
    total_batches = (len(valid_texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(valid_texts), batch_size):
        batch_num = i // batch_size + 1
        batch_texts = valid_texts[i:i + batch_size]
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
        
        try:
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                fp16=fp16
            )
            embeddings.append(batch_embeddings)
            
            # Log memory usage every 10 batches
            if batch_num % 10 == 0:
                import psutil
                memory_usage = psutil.virtual_memory().percent
                logger.info(f"Memory usage: {memory_usage:.1f}%")
                
        except Exception as e:
            logger.error(f"Error in batch {batch_num}: {e}")
            # Create zero vectors for failed batch
            zero_batch = np.zeros((len(batch_texts), model.get_sentence_embedding_dimension()))
            embeddings.append(zero_batch)
    
    # Combine all batches
    try:
        all_embeddings = np.vstack(embeddings)
    except Exception as e:
        logger.error(f"Error combining batches: {e}")
        # Fallback: create zero matrix
        all_embeddings = np.zeros((len(valid_texts), model.get_sentence_embedding_dimension()))
    
    # Create full matrix with zeros for empty texts
    full_embeddings = np.zeros((len(texts), all_embeddings.shape[1]))
    for idx, emb_idx in enumerate(valid_indices):
        full_embeddings[emb_idx] = all_embeddings[idx]
    
    logger.info(f"Encoded {len(valid_texts)} valid texts out of {len(texts)} total")
    return full_embeddings


def encode_tokens(word_model, tokens_list: List[List[str]], 
                 idf_dict: Optional[Dict[str, float]] = None,
                 sif_alpha: float = 1e-3) -> Tuple[np.ndarray, float, float]:
    """Encode tokens using word embeddings with SIF weighting."""
    logger.info(f"Encoding {len(tokens_list)} tokenized documents")
    
    embeddings = []
    oov_count = 0
    total_tokens = 0
    idf_covered = 0
    
    for tokens in tokens_list:
        if not tokens:
            # Empty document - use zero vector
            if embeddings:
                zero_vec = np.zeros(embeddings[-1].shape)
                embeddings.append(zero_vec)
            else:
                # First document - need to get dimension from model
                if hasattr(word_model, 'get_dimension'):
                    dim = word_model.get_dimension()
                else:
                    # For fastText, get dimension from a sample word
                    sample_vec = word_model.get_word_vector('test')
                    dim = len(sample_vec)
                embeddings.append(np.zeros(dim))
            continue
        
        # Get word vectors for each token
        word_vectors = []
        valid_tokens = []
        
        for token in tokens:
            try:
                if hasattr(word_model, 'get_word_vector'):  # fastText
                    vec = word_model.get_word_vector(token)
                else:  # word2vec
                    vec = word_model[token]
                word_vectors.append(vec)
                valid_tokens.append(token)
            except (KeyError, ValueError):
                oov_count += 1
                continue
        
        total_tokens += len(tokens)
        
        if not word_vectors:
            # No valid tokens found - use zero vector
            if embeddings:
                zero_vec = np.zeros(embeddings[-1].shape)
                embeddings.append(zero_vec)
            else:
                # First document - need to get dimension
                if hasattr(word_model, 'get_dimension'):
                    dim = word_model.get_dimension()
                else:
                    sample_vec = word_model.get_word_vector('test')
                    dim = len(sample_vec)
                embeddings.append(np.zeros(dim))
            continue
        
        # Convert to numpy array
        word_vectors = np.array(word_vectors)
        
        # Apply SIF weighting if IDF is available
        if idf_dict is not None:
            weights = sif_weight(valid_tokens, idf_dict, sif_alpha)
            # Count IDF coverage
            for token in valid_tokens:
                if token in idf_dict:
                    idf_covered += 1
            
            # Apply weights
            weighted_vectors = word_vectors * weights.reshape(-1, 1)
            doc_vector = np.mean(weighted_vectors, axis=0)
        else:
            # Simple average
            doc_vector = np.mean(word_vectors, axis=0)
        
        embeddings.append(doc_vector)
    
    embeddings = np.array(embeddings)
    
    # Calculate statistics
    oov_ratio = oov_count / total_tokens if total_tokens > 0 else 0.0
    idf_coverage = idf_covered / total_tokens if total_tokens > 0 and idf_dict else None
    
    logger.info(f"Encoded {len(embeddings)} documents")
    logger.info(f"OOV ratio: {oov_ratio:.2%}")
    if idf_coverage is not None:
        logger.info(f"IDF coverage: {idf_coverage:.2%}")
    
    return embeddings, oov_ratio, idf_coverage


def main():
    parser = argparse.ArgumentParser(description='Generate embedding features from comments')
    
    # Basic parameters
    parser.add_argument('--mode', choices=['product_level', 'comment_level'], 
                       default='comment_level', help='Aggregation mode')
    parser.add_argument('--date-cutoff', type=str, help='Date cutoff (YYYY-MM-DD)')
    parser.add_argument('--pipeline-version', type=str, help='Pipeline version for normalized text')
    parser.add_argument('--limit', type=int, help='Limit number of samples (for testing)')
    
    # Text source parameters
    parser.add_argument('--text-source', choices=['raw', 'norm'], default='norm',
                       help='Text source (raw or normalized)')
    
    # Sentence embedding parameters (Pipeline A)
    parser.add_argument('--embed-model', type=str, 
                       default='paraphrase-multilingual-MiniLM-L12-v2',
                       help='Sentence transformer model name')
    parser.add_argument('--pooling', choices=['mean', 'cls', 'max'], default='mean',
                       help='Pooling method for sentence embeddings')
    
    # Word embedding parameters (Pipeline B)
    parser.add_argument('--word-emb', choices=['fasttext', 'word2vec'], 
                       help='Word embedding type')
    parser.add_argument('--word-emb-path', type=str, help='Path to word embedding model')
    parser.add_argument('--idf-source', choices=['db', 'none'], default='none',
                       help='IDF source for SIF weighting')
    parser.add_argument('--sif-alpha', type=float, default=1e-3, help='SIF alpha parameter')
    parser.add_argument('--sif-remove-pc', type=int, default=1, help='Remove n principal components')
    
    # Processing parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for encoding')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto',
                       help='Device for processing')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 for sentence transformers')
    
    # Output parameters
    parser.add_argument('--outdir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--run-name', type=str, help='Custom run name (default: timestamp)')
    parser.add_argument('--with-y', action='store_true', help='Also generate y labels')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.text_source == 'norm' and not args.pipeline_version:
        raise ValueError("pipeline_version required for text_source='norm'")
    
    if args.word_emb and not args.word_emb_path:
        raise ValueError("word_emb_path required when using word embeddings")
    
    # Determine pipeline type
    use_sentence_emb = args.word_emb is None
    if use_sentence_emb:
        logger.info("Using sentence embedding pipeline")
    else:
        logger.info("Using word embedding pipeline")
    
    # Setup output directory
    if not args.run_name:
        args.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    outdir = os.path.join(args.outdir, args.run_name)
    os.makedirs(outdir, exist_ok=True)
    logger.info(f"Output directory: {outdir}")
    
    # Start timing
    start_time = datetime.now()
    
    # Connect to database
    conn = get_db_conn()
    
    try:
        # Fetch data
        if use_sentence_emb:
            # Pipeline A: Sentence embeddings
            data = fetch_texts(
                conn, 
                text_source=args.text_source,
                pipeline_version=args.pipeline_version,
                date_cutoff=args.date_cutoff,
                limit=args.limit
            )
            
            comment_ids = [row[0] for row in data]
            product_ids = [row[1] for row in data]
            capture_times = [row[2] for row in data]
            texts = [row[3] for row in data]
            
            # Load sentence transformer
            model, device = load_sentence_transformer(args.embed_model, args.device)
            
            # Encode sentences
            embeddings = encode_sentences(
                model, texts, args.batch_size, args.pooling, args.fp16
            )
            
            oov_ratio = None
            idf_coverage = None
            
        else:
            # Pipeline B: Word embeddings
            data = fetch_tokens(
                conn,
                pipeline_version=args.pipeline_version,
                date_cutoff=args.date_cutoff,
                limit=args.limit
            )
            
            comment_ids = [row[0] for row in data]
            product_ids = [row[1] for row in data]
            capture_times = [row[2] for row in data]
            tokens_list = [row[3] for row in data]
            
            # Load word embeddings
            word_model = load_word_embeddings(args.word_emb, args.word_emb_path)
            
            # Load IDF if requested
            idf_dict = None
            if args.idf_source == 'db' and args.pipeline_version:
                idf_dict = load_idf_from_db(conn, args.pipeline_version)
            
            # Encode tokens
            embeddings, oov_ratio, idf_coverage = encode_tokens(
                word_model, tokens_list, idf_dict, args.sif_alpha
            )
            
            # Remove principal components if requested
            if args.sif_remove_pc > 0:
                embeddings = remove_pc(embeddings, args.sif_remove_pc)
            
            # For preview, convert tokens back to text
            texts = [' '.join(tokens) for tokens in tokens_list]
        
        # End timing
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Save outputs
        n_samples, dim = embeddings.shape
        
        # Save embedding matrix
        embed_path = os.path.join(outdir, 'X_embed.npz')
        save_npz_compressed(embed_path, embeddings)
        
        # Create meta.csv
        meta_data = []
        for i, (comment_id, product_id, capture_time) in enumerate(zip(comment_ids, product_ids, capture_times)):
            if use_sentence_emb:
                # For sentence embeddings, count words in text
                text = texts[i] if i < len(texts) else ""
                n_tokens = len(text.split()) if text else 0
            else:
                # For word embeddings, count tokens from tokenization
                tokens = tokens_list[i] if i < len(tokens_list) else []
                n_tokens = len(tokens)
            meta_data.append((i, comment_id, product_id, capture_time, n_tokens))
        
        meta_path = os.path.join(outdir, 'meta.csv')
        save_csv(meta_path, meta_data, ['row_ix', 'comment_id', 'product_id', 'capture_time', 'n_tokens'])
        
        # Create sample preview
        preview_data = preview_rows(comment_ids, texts, embeddings)
        preview_path = os.path.join(outdir, 'sample_preview.csv')
        save_csv(preview_path, preview_data, ['comment_id', 'text_head', 'vec_head'])
        
        # Create manifest
        files = {
            'meta': 'meta.csv',
            'X_embed': 'X_embed.npz',
            'preview': 'sample_preview.csv'
        }
        
        manifest = create_manifest(
            run_name=args.run_name,
            start_time=start_time,
            end_time=end_time,
            device=device if use_sentence_emb else 'cpu',
            mode=args.mode,
            date_cutoff=args.date_cutoff,
            pipeline_version=args.pipeline_version,
            text_source=args.text_source if use_sentence_emb else None,
            embed_model=args.embed_model if use_sentence_emb else None,
            pooling=args.pooling if use_sentence_emb else None,
            word_emb=args.word_emb if not use_sentence_emb else None,
            idf_source=args.idf_source if not use_sentence_emb else None,
            sif_alpha=args.sif_alpha if not use_sentence_emb else None,
            sif_remove_pc=args.sif_remove_pc if not use_sentence_emb else 0,
            n_samples=n_samples,
            dim=dim,
            batch_size=args.batch_size,
            files=files,
            notes="local-only; no DB write"
        )
        
        manifest_path = os.path.join(outdir, 'manifest.json')
        save_manifest(manifest_path, manifest)
        
        # Print summary
        print_summary(n_samples, dim, embeddings, oov_ratio, idf_coverage, processing_time)
        
        logger.info(f"Embedding generation completed successfully!")
        logger.info(f"Output directory: {outdir}")
        
    finally:
        conn.close()


if __name__ == '__main__':
    main()
