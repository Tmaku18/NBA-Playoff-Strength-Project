# NBA "True Strength" Prediction

**Attention-Based Deep Set Network with Ensemble Validation**

Tanaka Makuvaza  
Georgia State University — Advanced Machine Learning  
January 27, 2026

---

## Overview
This project builds a **Multi-Modal Stacking Ensemble** to predict NBA **True Team Strength** using a Deep Set roster model plus a Hybrid tabular ensemble (XGBoost + Random Forest). The system targets **future outcomes** and identifies **Sleepers** versus **Paper Tigers** without circular evaluation.

---

## Key Design Choices
- **Target:** Future W/L (next 5) or Final Playoff Seed — **never** efficiency.
- **True Strength:** Latent **Z** from Deep Set penultimate layer; score mapped to percentile within conference.
- **No Net Rating leakage:** `net_rating` is excluded as a model input and never used for evaluation.
- **Stacking:** K-fold **OOF** predictions + level-2 Logistic Regression.
- **Game-level ListMLE:** lists per conference-date/week; evaluation season-end only.
- **Explainability:** SHAP on Model B only; Integrated Gradients or permutation ablation for Model A.

---

## Data Sources
- **nba_api** (official): play-by-play, player/team logs, tracking data.
- **Kaggle (Wyatt Walsh):** historical SOS/SRS (preferred over live scraping).
- **Basketball-Reference:** fallback only.

**Storage:** DuckDB preferred; SQLite allowed with pre-aggregation + indexing.

---

## Evaluation
- **Ranking:** NDCG, Spearman, MRR.
- **Future outcomes:** Brier score.
- **Sleeper detection:** ROC-AUC on playoff upsets.
- **Baselines:** rank-by-SRS and rank-by-Net-Rating.

---

## Outputs
- Predicted rank (1–15)
- True strength score (0–1)
- Fraud/Sleeper delta
- Ensemble agreement
- Roster dependence (attention weights)

---

## Reproducibility
- Set seeds for torch/numpy/sklearn.
- Persist OOF predictions to `outputs/oof_*.parquet`.
- Version datasets with hashes.

---

## Full Plan
See `.cursor/plans/Plan.md`.
