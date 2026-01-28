# NBA "True Strength" Prediction

**Attention-Based Deep Set Network with Ensemble Validation for NBA "True Strength" Prediction**

Tanaka Makuvaza  
Georgia State University — Advanced Machine Learning  
January 2026

---

## Overview

Traditional sports prediction models rely on team-level season averages, which hide the impact of individual players and non-linear roster interactions. This project implements a **hybrid set-based neural network** that treats a team as a **set of player vectors** rather than a single aggregate, combining:

- **Player embeddings** and **rolling-window stats** (last 10 / last 30 games)
- **Minutes-weighted attention** (“The Coach”) over the roster
- **ListMLE** (listwise) loss for rank and **MSE** for net rating
- A **stacking ensemble** with a **Random Forest** on team-level SOS, SRS, and Four Factors

The goal is to predict **“True Team Strength”** more accurately than raw standings and to flag **Sleepers** (under-ranked) and **Paper Tigers** (over-ranked).

---

## Data

- **Seasons**: 2015–16 through 2025–26 (post–Three-Point Revolution)
- **Sources**:
  - **nba_api**: games, player/team logs, rosters, tracking
  - **Basketball-Reference**: Strength of Schedule (SOS), Simple Rating System (SRS)
  - **Kaggle** (Wyatt Walsh `wyattowalsh/basketball`): historical validation and gap-filling

Data is merged into a **SQLite** database (`data/nba.db`).

---

## Models

| Component | Role |
|-----------|------|
| **Model A** | Deep Set: Player Encoder (shared MLP) → Minutes-Weighted Multi-Head Attention → Rank (ListMLE) + Net Rating (MSE) heads |
| **Model B** | Random Forest on team-level features (net rating, SOS, SRS, Four Factors) |
| **Meta-learner** | Stacking (e.g. `StackingRegressor`) or learned weighted average of A and B |

---

## Evaluation

- **NDCG** (ranking quality, emphasis on top)
- **Spearman** (predicted vs actual rank)
- **MRR** (“when we pick #1, how often are they truly #1?”)
- **MAE** on Net Rating (auxiliary)

---

## Outputs

Per team: **predicted rank**, **true strength score** (0–1), **predicted net rating**, **fraud/sleeper delta** (`actual_rank - predicted_rank`), **roster dependence** (from attention weights). Structured JSON for downstream analysis and viz.

---

## Repository Structure

```
mnu/
├── README.md
├── .cursor/plans/Plan.md    # Full implementation plan
├── config/
├── data/                    # raw, processed, nba.db
├── src/
│   ├── data/                # nba_api, bbref, kaggle, db, roster building
│   ├── features/            # rolling, four factors, team context
│   ├── models/              # Model A, B, ListMLE, stacking
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── viz/                 # accuracy, fraud/sleeper, SHAP, attention
├── scripts/                 # 1_download → 2_build_db → 3_build_rosters → 4_train → 5_evaluate → 6_run_inference
├── outputs/
└── tests/
```

---

## Run Order

1. `scripts/1_download_raw.py` — nba_api, BRef, Kaggle → `data/raw`
2. `scripts/2_build_db.py` — Schema + load → `data/nba.db`
3. `scripts/3_build_rosters.py` — Roster matrices, rolling stats, embedding index → `data/processed/`
4. `scripts/4_train.py` — Train Model A, Model B, meta-learner
5. `scripts/5_evaluate.py` — NDCG, Spearman, MRR, MAE → `outputs/`
6. `scripts/6_run_inference.py` — Predictions and JSON → `outputs/`

---

## Setup

- **Python**: 3.10+
- **Install**: `pip install -r requirements.txt` (once `requirements.txt` is added per the plan)
- **Config**: `config/defaults.yaml` for seasons, paths, model dims, etc.

---

## Plan

The full implementation plan (architecture, data pipeline, training, evaluation, viz, and todos) is in [`.cursor/plans/Plan.md`](.cursor/plans/Plan.md).
