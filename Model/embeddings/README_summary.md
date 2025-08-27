# Summary of Changes for train_with_embeddings.py

## Overview

This document summarizes the modifications made to support embedding features in the training pipeline, including changes to `emb_build.py` and the creation of `train_with_embeddings.py` and `feature_analysis_with_embeddings.py`.

## Files Modified/Created

### 1. Model/embeddings/emb_build.py (Modified)
**Purpose**: Generate embeddings with training/validation split and y labels

**Key Changes**:
- Added `--with-y` parameter to generate y labels
- Modified to use `date_cutoff` for training/validation split:
  - `<= date_cutoff`: Training data (used for embedding generation)
  - `> date_cutoff`: Validation data (used to determine y labels)
- Added `get_y_labels()` function to query sales changes
- Enhanced manifest to include y label information
- Saves `y.npy` file alongside embeddings

**Usage**:
```bash
python Model/embeddings/emb_build.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --pipeline-version v1 \
  --with-y \
  --outdir Model/embeddings/outputs
```

### 2. Model/embeddings/train_with_embeddings.py (Created)
**Purpose**: Training script that integrates embedding features with existing Dense and TF-IDF features

**Key Features**:
- **Two Embedding Modes**:
  - `append`: Dense + TF-IDF + Embedding
  - `only`: Dense + Embedding (skip TF-IDF)
- **Embedding Loading**: Loads pre-generated embeddings with y labels
- **Data Alignment**: Aligns embeddings with training dataset
- **Feature Assembly**: Combines different feature types using `scipy.sparse.hstack`
- **Consistent Workflow**: Maintains same CV, oversampling, and evaluation as `train.py`

**New Parameters**:
- `--embed-path`: Path to embedding directory
- `--embed-mode`: `append` or `only`
- `--embed-scale`: `none` or `standardize`
- `--embed-dtype`: `float32` or `float64`

**Usage**:
```bash
python Model/embeddings/train_with_embeddings.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --vocab-mode global \
  --top-n 100 \
  --algorithms xgboost \
  --fs-methods no_fs \
  --cv 10 \
  --oversample none \
  --embed-path Model/embeddings/outputs/product_level_raw_20250823 \
  --embed-mode append \
  --embed-scale none \
  --outdir Model/outputs_embed
```

### 3. Model/embeddings/feature_analysis_with_embeddings.py (Created)
**Purpose**: Comprehensive feature analysis tool supporting embedding features

**Key Features**:
- **Dense Feature Analysis**: Cohen's d, mutual information, statistical tests
- **Embedding Feature Analysis**: Distribution stats, Fisher discriminant score
- **PCA Visualization**: 2D projections with class separation metrics
- **Feature Sufficiency Scoring**: Comprehensive 0-100 score with recommendations
- **Consistent Data Loading**: Uses same logic as `train_with_embeddings.py`

**Usage**:
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

## Workflow

### 1. Generate Embeddings with Y Labels
```bash
python Model/embeddings/emb_build.py \
  --mode product_level \
  --date-cutoff 2025-06-25 \
  --pipeline-version v1 \
  --with-y \
  --outdir Model/embeddings/outputs
```

### 2. Analyze Feature Quality
```bash
python Model/embeddings/feature_analysis_with_embeddings.py \
  --embed-path Model/embeddings/outputs/[run_id] \
  --embed-mode append \
  --save-plots
```

### 3. Train Model with Embeddings
```bash
python Model/embeddings/train_with_embeddings.py \
  --embed-path Model/embeddings/outputs/[run_id] \
  --embed-mode append \
  --algorithms xgboost,svm \
  --cv 10
```

## Key Technical Changes

### Data Flow
1. **emb_build.py**: Generates embeddings for training data (≤ date_cutoff) and y labels based on validation data (> date_cutoff)
2. **train_with_embeddings.py**: Loads embeddings, aligns with training dataset, combines features, trains models
3. **feature_analysis_with_embeddings.py**: Analyzes feature quality and provides recommendations

### Feature Assembly
- **Dense Features**: Standardized with `StandardScaler(with_mean=False)`
- **TF-IDF Features**: Loaded as sparse matrices (skipped in `only` mode)
- **Embedding Features**: Loaded from `X_embed.npz`, optionally standardized
- **Combination**: Using `scipy.sparse.hstack` for efficient sparse matrix operations

### Data Alignment
- Embeds use same `product_id`/`comment_id` alignment logic
- Handles subset relationships gracefully
- Maintains data consistency across all tools

### Output Enhancements
- **Manifest Files**: Include embedding information and feature shapes
- **Analysis Reports**: Comprehensive JSON reports with recommendations
- **Visualizations**: PCA plots showing class separation

## Benefits

1. **Consistent Workflow**: All tools use same data loading and alignment logic
2. **Flexible Integration**: Support for different embedding modes and feature combinations
3. **Quality Assurance**: Feature analysis tool helps evaluate embedding quality
4. **Comprehensive Documentation**: Detailed usage examples and parameter descriptions
5. **Extensible Design**: Easy to add new embedding models or analysis methods

## File Structure

```
Model/embeddings/
├── emb_build.py                           # Generate embeddings with y labels
├── train_with_embeddings.py               # Training with embedding features
├── feature_analysis_with_embeddings.py    # Feature analysis tool
├── README_feature_analysis.md             # Feature analysis documentation
├── README_summary.md                      # This summary file
└── outputs/                               # Generated embeddings
    └── [run_id]/
        ├── X_embed.npz                    # Embedding matrix
        ├── meta.csv                       # Metadata
        ├── y.npy                          # Target labels
        └── manifest.json                  # Embedding metadata
```

## Next Steps

1. **Test with Real Data**: Run the complete workflow with actual embedding data
2. **Performance Optimization**: Optimize memory usage for large embedding matrices
3. **Additional Analysis**: Add more visualization and analysis methods
4. **Model Comparison**: Compare performance with and without embeddings
5. **Hyperparameter Tuning**: Optimize embedding and model parameters
