# Feature Analysis Tool for train_with_embeddings.py

This tool provides comprehensive feature analysis capabilities for the `train_with_embeddings.py` script, supporting both traditional features (Dense, TF-IDF) and embedding features.

## Features

- **Dense Feature Analysis**: Analyzes distribution differences between y=0 and y=1 classes
- **Embedding Feature Analysis**: Evaluates embedding quality and class separation
- **PCA Visualization**: Creates 2D visualizations of the combined feature space
- **Comprehensive Reporting**: Generates detailed analysis reports with recommendations

## Usage

### Basic Usage

```bash
python Model/embeddings/feature_analysis_with_embeddings.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --vocab-mode global \
  --top-n 100 \
  --pipeline-version v1 \
  --exclude-products 8918452 \
  --embed-path Model/embeddings/outputs/product_level_raw_20250823 \
  --embed-mode append \
  --embed-scale none \
  --outdir Model/analysis_outputs_embed \
  --save-plots
```

### Parameters

#### Data Loading Parameters
- `--mode`: Analysis mode (`product_level` or `comment_level`)
- `--date-cutoff`: Date cutoff for training data (YYYY-MM-DD)
- `--pipeline-version`: Pipeline version for data loading
- `--vocab-mode`: Vocabulary mode (`global`)
- `--top-n`: Number of top features to use
- `--exclude-products`: Comma-separated list of product IDs to exclude
- `--keyword`: Single keyword filter (optional)

#### Embedding Parameters
- `--embed-path`: **Required** - Path to embedding directory containing:
  - `X_embed.npz`: Embedding matrix
  - `meta.csv`: Metadata with product_id/comment_id
  - `y.npy`: Target labels
  - `manifest.json`: Embedding metadata (optional)
- `--embed-mode`: Embedding integration mode
  - `append`: Use Dense + TF-IDF + Embedding features
  - `only`: Use only Dense + Embedding features (skip TF-IDF)
- `--embed-scale`: Embedding standardization
  - `none`: No standardization
  - `standardize`: Apply StandardScaler
- `--embed-dtype`: Embedding data type (`float32` or `float64`)

#### Output Parameters
- `--outdir`: Output directory for analysis results
- `--save-plots`: Save visualization plots (optional)

## Analysis Components

### 1. Dense Feature Analysis
- **Cohen's d**: Standardized mean difference between classes
- **Mutual Information**: Information-theoretic measure of feature-class relationship
- **Mann-Whitney U Test**: Statistical significance test
- **Feature Significance**: Identifies features with significant class differences

### 2. Embedding Feature Analysis
- **Distribution Statistics**: Mean, std, min, max of embedding vectors
- **Class Separation**: Fisher discriminant score
- **Center Distance**: Distance between class centers
- **Within-Class Variance**: Variance within each class

### 3. PCA Visualization
- **2D Projection**: Visualizes high-dimensional feature space
- **Class Separation Score**: Quantifies separation in 2D space
- **Explained Variance**: Shows how much variance is captured by first 2 components

### 4. Feature Sufficiency Scoring
The tool provides a comprehensive score (0-100) based on:
- **Feature Separation (40%)**: Quality of individual features
- **Embedding Separation (30%)**: Quality of embedding representations
- **Visualization Separation (30%)**: Overall class separation in reduced space

## Output Files

### Analysis Results
- `feature_analysis_embed_detailed_YYYYMMDD_HHMMSS.json`: Detailed analysis results
- `feature_visualization_embed_YYYYMMDD_HHMMSS_pca.png`: PCA visualization (if --save-plots)

### JSON Structure
```json
{
  "data_summary": {
    "total_samples": 4616,
    "positive_samples": 1234,
    "negative_samples": 3382,
    "imbalance_ratio": 2.74,
    "dense_features_count": 15,
    "tfidf_features_count": 100,
    "embedding_features_count": 384
  },
  "feature_analysis": {
    "feature_analysis": [...],
    "summary_stats": {
      "total_features": 15,
      "features_with_significant_diff": 12,
      "features_with_high_separation": 8
    }
  },
  "embedding_analysis": {
    "embedding_analysis": {
      "embedding_shape": [4616, 384],
      "embedding_stats": {...},
      "separation_metrics": {...}
    },
    "summary_stats": {...}
  },
  "pca_analysis": {
    "pca_results": {
      "explained_variance_ratio": [0.15, 0.08],
      "separation_score": 2.34
    }
  },
  "conclusions": {
    "feature_sufficiency_score": 75.2,
    "recommendations": [...]
  }
}
```

## Recommendations

The tool provides actionable recommendations based on analysis results:

- **Low Feature Separation**: Suggest adding more relevant features or feature engineering
- **Poor Embedding Quality**: Recommend adjusting embedding model or aggregation method
- **Insufficient Class Separation**: Suggest stronger features or different models
- **Overall Assessment**: Provide guidance on model complexity and optimization strategies

## Example Workflow

1. **Generate Embeddings**: Use `emb_build.py` to create embeddings
2. **Run Feature Analysis**: Use this tool to evaluate feature quality
3. **Review Recommendations**: Check the analysis results and recommendations
4. **Adjust Parameters**: Modify embedding or feature parameters if needed
5. **Train Model**: Use `train_with_embeddings.py` with optimized parameters

## Integration with train_with_embeddings.py

This tool uses the same data loading logic as `train_with_embeddings.py`, ensuring consistency between analysis and training. The embedding loading and alignment functions are identical, providing reliable analysis of the exact same data that will be used for training.
