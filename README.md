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
- **No Net Rating leakage:** `net_rating` is excluded as a model input and never used as a target or evaluation metric (allowed only in baselines).
- **Stacking:** K-fold **OOF** across **all training seasons**; level-2 **RidgeCV** on pooled OOF (not Logistic Regression).
- **Game-level ListMLE:** lists per conference-date/week; **torch.logsumexp** for numerical stability; hash-trick embeddings for new players.
- **Season config:** Hard-coded season date ranges in `defaults.yaml` to avoid play-in ambiguity.
- **Explainability:** SHAP on Model B only; Integrated Gradients or permutation ablation for Model A.

---

## Data Sources
- **nba_api** (official): play-by-play, player/team logs, tracking data.
- **Kaggle (Wyatt Walsh):** **primary** for SOS/SRS and historical validation.
- **Basketball-Reference:** **fallback** for SOS/SRS when Kaggle unavailable.
- **Proxy SOS:** If both are unavailable, compute from internal DB (e.g. opponent win-rate) and document.

**Storage:** DuckDB preferred; SQLite allowed with pre-aggregation + indexing.

---

## Evaluation
- **Ranking:** NDCG, Spearman, MRR.
- **Future outcomes:** Brier score.
- **Sleeper detection:** ROC-AUC on playoff upsets.
- **Baselines:** rank-by-SRS, rank-by-Net-Rating, **Dummy** (e.g. previous-season rank or rank-by-net-rating).

---

## Outputs (per run)
- Predicted rank (1–30, league-wide) and true strength score (0–1).
- Classification: **Sleeper** (under-ranked by standings), **Paper Tiger** (over-ranked), **Aligned**.
- Delta (actual rank − predicted rank) and ensemble agreement (Model A / XGB / RF ranks).
- Roster dependence (attention weights when available).
- `pred_vs_actual.png`: predicted vs actual rank scatter.

---

## How to Run the Pipeline

**Run order (production, real data only):**

1. **Setup:** `pip install -r requirements.txt`
2. **Config:** Edit `config/defaults.yaml` if needed (seasons, paths, model params). DB path: `data/processed/nba_build_run.duckdb`.
3. **Data:**  
   - `python -m scripts.1_download_raw` — fetch player/team logs via nba_api (writes to `data/raw/` as parquet).  
   - `python -m scripts.2_build_db` — build DuckDB from raw → `data/processed/nba_build_run.duckdb`, update `data/manifest.json`.
4. **Training (real DB):**  
   - `python -m scripts.3_train_model_a` — K-fold OOF → `outputs/oof_model_a.parquet`, then final model → `outputs/best_deep_set.pt`.  
   - `python -m scripts.4_train_model_b` — K-fold OOF → `outputs/oof_model_b.parquet`, then XGB + RF → `outputs/xgb_model.joblib`, `outputs/rf_model.joblib`.  
   - `python -m scripts.4b_train_stacking` — merge OOF parquets, RidgeCV → `outputs/ridgecv_meta.joblib`, `outputs/oof_pooled.parquet` (requires OOF from 3 and 4).
5. **Inference:** `python -m scripts.6_run_inference` — load DB and models, run Model A/B + meta → `outputs/run_001/predictions.json`, `outputs/run_001/pred_vs_actual.png`.
6. **Evaluation:** `python -m scripts.5_evaluate` — uses predictions from step 5 → `outputs/eval_report.json` (NDCG, Spearman, MRR, ROC-AUC upset).
7. **Explainability:** `python -m scripts.5b_explain` — SHAP on real team-context X, attention ablation on real list batch → `outputs/shap_summary.png`.

**Optional:** `python -m scripts.run_manifest` (run manifest); `python -m scripts.run_leakage_tests` (before training).

---

## Anti-Leakage Checklist

- [ ] **Time rule:** All features use only rows with `game_date < as_of_date` (strict t-1). Rolling stats use `shift(1)` before aggregation.
- [ ] **Roster:** Minutes and roster selection use only games before `as_of_date`.
- [ ] **Model B:** Feature set must **not** include `net_rating`. Enforced in `src.features.team_context.FORBIDDEN` and `train_model_b`.
- [ ] **ListMLE:** Targets are standings-to-date (win-rate), not season-end. Evaluation remains season-end.
- [ ] **Baselines only:** Net Rating is used only in `rank-by-Net-Rating` baseline, computed from off/def ratings, never as a model input.

---

## Report Assets (deliverables)

All paths under `outputs/` (or `config.paths.outputs`). Produced from real data when DB and models exist.

- `outputs/eval_report.json` — NDCG, Spearman, MRR, ROC-AUC upset (from script 5, using inference output).
- `outputs/run_001/predictions.json` — per-team predicted rank, true strength score, delta, classification (Sleeper/Paper Tiger/Aligned), ensemble diagnostics.
- `outputs/run_001/pred_vs_actual.png` — predicted vs actual rank scatter (from script 6).
- `outputs/shap_summary.png` — Model B (RF) SHAP summary on real team-context features (script 5b).
- `outputs/oof_pooled.parquet`, `outputs/ridgecv_meta.joblib` — stacking meta-learner and pooled OOF (script 4b).
- `outputs/oof_model_a.parquet`, `outputs/oof_model_b.parquet` — OOF from scripts 3 and 4 (Option A: K-fold, real data).
- `outputs/best_deep_set.pt`, `outputs/xgb_model.joblib`, `outputs/rf_model.joblib` — trained Model A and Model B.

---

## Reproducibility

- **Seeds:** `src.utils.repro.set_seeds(seed)` for random, numpy, torch.
- **Manifests:** `outputs/run_manifest.json` (config snapshot, git hash, data manifest hash). `data/manifest.json` (raw/processed hashes).
- **OOF:** `outputs/oof_pooled.parquet` and `ridgecv_meta.joblib`.
- **Season boundaries:** Hard-coded in `config/defaults.yaml` to avoid play-in ambiguity.

---

## Full Plan

See `.cursor/plans/Plan.md`.

---

## Implementation Roadmap
The full phased development plan and file-by-file checklist are in
`.cursor/plans/Plan.md` under **Development and Implementation Plan**.
