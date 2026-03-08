# Airbnb Superhost Prediction — Multimodal Classification

**Authors:** Jinyu Cong, Furkan Narin
**Date:** January 2026

## Overview

This project addresses the binary classification task of predicting whether an Airbnb host holds the **Superhost** status, using a multimodal approach that combines **textual** and **tabular** data from Paris listings.

The central research question is:

> *To what extent does combining heterogeneous modalities (text and tabular data) improve Superhost classification compared to using each source in isolation?*

## Dataset

The dataset is based on Paris Airbnb listings (`listings.csv`) scraped in September 2025, containing **79,540 samples** and **79 features**.

**Target variable:** `host_is_superhost` (binary: t / f)
**Class distribution:** 78.7% Non-Superhost / 21.3% Superhost (imbalanced)

### Selected Features

| Modality | Features |
|---|---|
| **Textual (4)** | `name`, `description`, `neighborhood_overview`, `host_about` |
| **Numerical (10)** | `host_response_rate`, `host_acceptance_rate`, `host_listings_count`, `number_of_reviews`, `reviews_per_month`, `review_scores_rating`, `review_scores_cleanliness`, `review_scores_communication`, `review_scores_location`, `review_scores_value` |
| **Categorical (4)** | `host_response_time`, `host_identity_verified`, `room_type`, `instant_bookable` |

## Methods

### Preprocessing

- **Text:** Missing values replaced with empty string `""`
- **Numerical:** Missing values replaced with `0.0`
- **Categorical:** Ordinal encoding; missing values treated as a separate category

### Models

Two model families are evaluated, each tested under four configurations:

1. **Unimodal Text** — trained on textual data only
2. **Unimodal Tabular** — trained on structured data only
3. **Multimodal Early Fusion** — concatenated feature vectors before training
4. **Multimodal Late Fusion** — prediction probabilities from unimodal models combined via a meta-model

#### Model 1: Logistic Regression (LR)

- **Text pipeline:** TF-IDF (unigrams + bigrams, 20,000 features) → Logistic Regression (2,000 iterations, balanced class weights)
- **Tabular pipeline:** StandardScaler + OneHotEncoder → Logistic Regression
- **Early Fusion:** TF-IDF → SVD (128 dims) concatenated with tabular features → LR
- **Late Fusion:** Probabilities from text LR and tabular LR → meta Logistic Regression (gating model)

#### Model 2: Mixture of Experts (MoE)

A neural network architecture with **6 expert MLPs** and a **gating MLP** that dynamically weights each expert's output:

```
Output = Σ gate_weight_i × expert_output_i
```

- Loss function: `BCEWithLogitsLoss` with `pos_weight=3.7` (ratio of negative/positive class)
- Optimizer: Adam (lr=0.001), trained for 500 epochs
- **Early Fusion:** Concatenated text (TF-IDF + SVD 128 dims) and tabular vectors fed into a single MoE
- **Late Fusion:** Probabilities from text MoE and tabular MoE fed into a meta-MoE

## Results

| Model | Method | Precision | Recall | F1 | Accuracy | AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | Text | 0.791 | 0.743 | 0.760 | 0.743 | 0.786 |
| | Tabular | 0.839 | 0.737 | 0.760 | 0.737 | 0.858 |
| | Early Fusion | 0.852 | 0.769 | 0.788 | 0.769 | 0.884 |
| | **Late Fusion** | **0.853** | **0.822** | **0.832** | **0.822** | **0.897** |
| **MoE** | Text | 0.756 | 0.768 | 0.761 | 0.768 | 0.708 |
| | Tabular | 0.855 | 0.828 | 0.836 | 0.828 | 0.864 |
| | **Early Fusion** | **0.865** | **0.843** | **0.850** | **0.843** | **0.913** |
| | Late Fusion | 0.857 | 0.832 | 0.840 | 0.832 | 0.867 |

### Key Findings

- **Multimodal > Unimodal:** Multimodal approaches consistently outperform unimodal ones across all configurations and both models, confirming that textual data provides complementary "weak signals" (tone, host engagement) that enrich the hard thresholds of tabular criteria.
- **Tabular > Text (in isolation):** Structured features (ratings, response rate) remain the strongest individual predictors (LR AUC 0.858 vs 0.786 for text).
- **Best overall model: MoE + Early Fusion** achieves the highest performance with **AUC 0.913** and **F1 0.850**. The MoE gating mechanism efficiently captures non-linear interactions between textual and tabular dimensions when combined at the feature level.
- **Best LR configuration: Late Fusion** (AUC 0.897, F1 0.832). The linear model benefits more from combining high-level predictions than from managing a concatenated feature space.
- **Class imbalance handling:** The positive class weight of 3.7 yielded strong recall (0.843 for MoE Early Fusion), detecting over 84% of true Superhosts without an explosion of false positives.

## Conclusion

Combining text and tabular modalities systematically outperforms single-source models for Superhost prediction. The **MoE Early Fusion** architecture is the optimal model of this study, while **Logistic Regression with Late Fusion** offers a strong and interpretable alternative.

## Project Structure

```
AirbnbMultimodal/
├── projet_multimodalite.ipynb   # Main notebook (EDA, preprocessing, models, evaluation)
├── rapport.pdf                  # Full project report
└── README.md
```

## Requirements

```
pandas, numpy, matplotlib, seaborn, missingno
torch
scikit-learn
```
