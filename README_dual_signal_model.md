# Dual-Signal Explosive Prediction Model

This document explains the architecture and results of our "Dual-Signal" model for predicting explosive products.

## 1. Core Concept: Dual-Signal Architecture

We combine two distinct types of signals to detect viral hits:

### A. Physical Signal (Review Kinematics)
**"How fast is it moving?"**
Treats comment volume as a physical object with velocity, acceleration, and jerk.
*   **Velocity ($v$)**: Raw comment count (Volume).
*   **Acceleration ($a$)**: Rate of change of volume ($v_1 - v_2$). Detects *growth*.
*   **Jerk ($j$)**: Rate of change of acceleration ($v_1 - 2v_2 + v_3$). Detects *explosive onset*.
*   **Early Bird Momentum**: Rewards high acceleration in low-volume items (`acc / log(volume)`).

### B. Algorithm Signal (Category Fit)
**"Does it fit the consensus?"**
Models the "Prototypicality" of the product based on user feedback.
*   **Category Fit Score**: Measures the cosine similarity between the product's aggregated comments and the centroid of its category.
*   **Hypothesis**: Viral hits receive "standard" positive feedback (High Fit), while non-hits have unique outliers or irrelevant noise (Low Fit).
*   **Quality Driven Momentum**: `Acceleration * Category Fit`. Filters out high-speed items that lack quality consensus.

## 2. Key Results

### Model Performance
*   **AUC**: **0.8498** (Best Performance).
*   **F1 Score**: **0.2826** (Improved).

### Feature Insights
1.  **Kinematics (Physical)**:
    *   **Crucial for False Negatives**: Missed hits (FN) have the **highest Acceleration (0.18)**.
    *   **Optimization**: We successfully forced the model to respect this signal using Monotonic Constraints and a Hybrid Rule (`Jerk > 0.1`).

2.  **Semantic Novelty (Algorithm)**:
    *   **High Baseline**: Most products have a high novelty score (~0.89).
    *   **Subtle Signal**: While not a silver bullet on its own, adding it contributed to the overall AUC improvement.

## 3. Production Strategy: Hybrid Rule

To maximize the capture of viral hits, we recommend a **Hybrid Strategy**:

1.  **Base Model**: XGBoost with Dual-Signal features (AUC 0.85).
2.  **Fast Lane Override**:
    *   **Rule**: If `Model Probability > 0.4` **AND** `Jerk > 0.1`
    *   **Action**: Force Prediction = **EXPLOSIVE (1)**.
    *   **Impact**: Boosts F1 score by **+1.7%** by catching rapidly accelerating items that the base model might be too conservative about.

## 4. Next Steps
*   **Deploy**: Push the `data_loader.py` (with Kinematics + Novelty) and `run_experiments.py` (with Constraints) to production.
*   **Monitor**: Watch the `Jerk` values of incoming products. High Jerk = High Alert.
