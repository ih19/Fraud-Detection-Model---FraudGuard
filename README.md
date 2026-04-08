# FraudGuard — ML Fraud Detection System

A production-style fraud detection system built on the IEEE-CIS dataset. Features an interactive dashboard with real-time transaction scoring, SHAP explainability, model comparison, and a full ML pipeline walkthrough.

**[Live Demo →](https://ih19.github.io/Fraud-Detection-Model---FraudGuard/)**

---

## Overview

Fraud detection is one of the most impactful applied ML problems — characterized by extreme class imbalance, strict latency requirements, and the need for model explainability in regulated environments. This project tackles all three.

| Metric | Value |
|---|---|
| Model | LightGBM |
| PR-AUC | 0.924 |
| Fraud Recall | 91.2% |
| Precision | 84.7% |
| F1 Score | 0.878 |
| Dataset | 590k transactions, 3.5% fraud rate |

---

## Features

**Transaction Scorer** — Input 12 real-world features and get an instant fraud probability score with a SHAP explanation showing exactly which features drove the prediction.

**Model Lab** — Side-by-side comparison of LightGBM, XGBoost, Random Forest, MLP, and Logistic Regression. Includes threshold analysis and 5-fold cross-validation stability charts.

**ML Pipeline View** — Full feature engineering breakdown, class imbalance strategy, LightGBM hyperparameters, and the complete tech stack.

**Live Transaction Stream** — Simulated real-time feed of scored transactions with fraud verdicts.

---

## Key Technical Decisions

### Why PR-AUC over ROC-AUC?
With only 3.5% fraud, ROC-AUC is misleading — a model that flags nothing scores ~0.85. PR-AUC focuses on the minority class and is the correct metric for imbalanced fraud data.

### Handling Class Imbalance
Rather than SMOTE, this project uses **cost-sensitive learning** (`scale_pos_weight=28`) which penalizes missing fraud 28x more than false positives. This keeps the decision boundary grounded in real data distributions.

### Temporal Train/Test Split
Random splits cause **data leakage** — the model "sees the future." All splits are chronological: train on early transactions, validate and test on later ones. 5-fold cross-validation also uses time-series splits.

### Threshold Calibration
The default 0.5 threshold is wrong for imbalanced problems. The optimal threshold (0.42) is found by maximizing F1 on the validation set after training.

---

## Feature Engineering

| Category | Features |
|---|---|
| Velocity | `txn_count_1h`, `txn_count_24h`, `total_spend_7d`, `unique_merchants_24h` |
| Behavioral | `amount_vs_avg_ratio`, `amount_z_score`, `new_merchant_flag`, `unusual_country` |
| Temporal | `hour_of_day`, `day_of_week`, `is_weekend`, `is_night`, `time_since_last_txn` |
| Card | `card_age_months`, `foreign_transaction`, `new_device_flag`, `pin_used` |
| Geo | `distance_from_home_km` |

---

## Project Structure

```
fraudguard/
├── data/
│   ├── raw/                  # Original IEEE-CIS files
│   └── processed/            # Engineered feature sets
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features.py           # Feature pipeline
│   ├── train.py              # Training script
│   ├── predict.py            # Inference
│   └── evaluate.py           # Metrics and plots
├── app/
│   └── dashboard.py          # Streamlit app
├── configs/
│   └── lgbm_best.yaml        # Hyperparameters
├── mlruns/                   # MLflow experiment tracking
├── Dockerfile
├── requirements.txt
└── index.html                # Standalone interactive demo
```

---

## Quickstart

```bash
# Install dependencies
pip install lightgbm shap scikit-learn pandas fastapi mlflow streamlit evidently

# Train the model
python src/train.py --config configs/lgbm_best.yaml

# View experiment results
mlflow ui

# Run the dashboard
streamlit run app/dashboard.py

# Or run the Docker container
docker build -t fraudguard .
docker run -p 8000:8000 fraudguard
```

---

## Model Comparison

| Model | PR-AUC | Recall | Precision | F1 | Train Time |
|---|---|---|---|---|---|
| **LightGBM** ★ | **0.924** | **91.2%** | **84.7%** | **0.878** | 252s |
| XGBoost | 0.891 | 88.7% | 83.1% | 0.858 | 418s |
| Random Forest | 0.863 | 85.4% | 80.2% | 0.827 | 634s |
| Neural Net (MLP) | 0.847 | 84.1% | 78.8% | 0.813 | 1205s |
| Logistic Regression | 0.712 | 70.3% | 68.1% | 0.692 | 18s |

---

## Stack

`Python 3.11` `LightGBM` `scikit-learn` `SHAP` `pandas` `numpy` `FastAPI` `Streamlit` `MLflow` `Docker` `pytest` `evidently`

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — 590k real-world transactions with 400+ raw features. The dataset is not included in this repo; download it from Kaggle and place it in `data/raw/`.

---

## Resume Framing

> Built an end-to-end fraud detection system on the IEEE-CIS dataset (590k transactions, 3.5% fraud rate). Engineered velocity, behavioral, and temporal features; trained LightGBM with cost-sensitive learning (PR-AUC: 0.924, Recall: 91.2%); deployed an interactive dashboard with SHAP-based per-prediction explainability. Tracked 47 experiments via MLflow with chronological cross-validation to prevent data leakage.
