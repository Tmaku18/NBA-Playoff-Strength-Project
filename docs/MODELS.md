# All models in the project

This document lists **every model** used in the pipeline: core ensemble (A, B, C), optional team-stats models, stacking, and auxiliary models. Each output folder under `output/` is **model-based**; see [OUTPUT_FOLDER_NAMING.md](OUTPUT_FOLDER_NAMING.md) and [MODEL_LINEUP_AND_NEXT_STEPS.md](MODEL_LINEUP_AND_NEXT_STEPS.md) for roles and run commands.

---

## Core ensemble

| Model | Code / artifact | Config | Role |
|-------|------------------|--------|------|
| **Model A** | Deep Set (roster-aware); `src.models.deep_set_rank`, `set_attention`, `listmle_loss` or `ranking_surrogate_losses` | `model_a.*`, `training.listmle_target`, `training.loss_type` | Roster-based strength; trained with ListMLE or Spearman/RMSE surrogate. Checkpoint: `best_deep_set.pt`. **In ensemble.** |
| **Model B** | XGBoost on team-context features | `model_b.xgb.*` | Team-level ranking; OOF in stacking. Artifact: `xgb_model.joblib`. **In ensemble.** |
| **Model C** | Random Forest on team-context features | `model_b.rf.*`, `training.train_model_c` | Same features as B; analytics/comparison only when enabled. Artifact: `rf_model.joblib`. **Not in default ensemble.** |
| **Stacking / meta** | RidgeCV on OOF (A + B, plus optional confidence and standings-win-rate columns) | `stacking.use_confidence`, `stacking.use_standings`, `model_a.attention` (confidence) | Level-2 blend (not a base model; blends A + B, optionally anchored on standings win rate to date). Artifact: `ridgecv_meta.joblib`; per-conference `ridgecv_meta_E.joblib` / `ridgecv_meta_W.joblib` are fit on conference-only OOF rows. Meta input is 2-5 columns; inference detects the width. |

---

## Optional team-stats models (script 4)

Trained on the same team-context features as Model B/C; produce mean (± uncertainty) for ranking. Loaded at inference as **extra_models**; their scores appear in predictions and eval when present. **Not** part of the default ensemble (ensemble = A + B only). Labeled **Model D** through **Model G** for consistency with A/B/C.

| Model | Letter | Code / artifact | Config | Output folder (when used as primary) |
|-------|--------|------------------|--------|--------------------------------------|
| **Linear Regression** | **Model D** | `src.models.linear_regression_model` | `training.train_team_stats_linear: true`, `team_stats_models.linear.*` | `linreg_model.joblib` → `output/team_stats_linear_regression/` when outputs path set there. |
| **Bayesian Ridge** | **Model E** | `src.models.bayesian_ridge_model` | `training.train_team_stats_bayesian_ridge: true`, `team_stats_models.bayesian_ridge.*` | `bayesian_ridge_model.joblib` → `output/team_stats_bayesian_ridge/`. |
| **GPR** (Gaussian Process Regression) | **Model F** | `src.models.gpr_model` | `training.train_team_stats_gpr: true`, `team_stats_models.gpr.*` | `gpr_model.joblib` → `output/team_stats_gpr/`. |
| **GMM** (Gaussian Mixture Model) | **Model G** | `src.models.gmm_rank_model` | `training.train_team_stats_gmm: true`, `team_stats_models.gmm.*` | `gmm_rank_model.joblib` → `output/team_stats_gmm/`. |

---

## Auxiliary / standalone

| Model | Letter | Code / artifact | Config / script | Role |
|-------|--------|------------------|------------------|------|
| **Logistic Regression** (clone classifier) | **Model H** | `src.models.lr_model`; script `4c_train_classifier_clone` | `config/clone_classifier.yaml` | Binary classifier (playoff team vs not); Train/Val/Holdout. Report: `clone_classifier_report.json`. Not used in ranking ensemble. |
| **Calibration** | — | `src.models.calibration` | Used for probability calibration when enabled | Calibrates outputs (e.g. championship odds). |
| **Confidence** | — | `src.models.confidence` | `model_a.attention` (entropy/max weight), XGB tree variance | Per-instance confidence for Model A and B; optional 4-column stacking. |

---

## Model A variants (loss / objective)

Model A is always the same **architecture** (Deep Set + set attention); only the **loss** and optional **inputs** (e.g. standing rank, team stats) change. Output folders reflect the training setup:

| Training setup | Loss / target | Output folder examples |
|----------------|--------------|-------------------------|
| ListMLE | `listmle_loss`; target: standings or playoff outcome | `2_listmle`, `4_listmle`, `6_baseline`, `7_listmle`, `11_listmle_standing_rank` |
| Spearman surrogate | `ranking_surrogate_losses` (Spearman) | `8_spearman_surrogate`, `10_spearman_surrogate_standing_rank` |
| Top-weighted Spearman surrogate | `ranking_surrogate_losses` (weights ∝ 1/rank^`training.loss_top_weight_power`; up-weights ranks 1-4) | `8_spearman_surrogate/improved_topweighted_02-27` |
| RMSE surrogate | `ranking_surrogate_losses` (rank RMSE) | `13_rmse_surrogate`, `15_rmse_surrogate_standing_rank` |
| MAP | MAP estimator variant | `14_map_run`, `16_map_standing_rank` |

Config: `training.loss_type` (e.g. `listmle`, `spearman_surrogate`, `weighted_spearman_surrogate`, `rank_rmse_surrogate`), `training.listmle_target` (e.g. `final_rank`, `playoff_outcome`).

**Note (Feb 27, 2026):** the surrogate losses had an inverted soft-rank sign (best team got the lowest score; the meta had to flip Model A). Fixed in `src/models/ranking_surrogate_losses.py`; models trained before the fix keep the old inverted convention internally. See [PROJECT_STATE_AND_BEST_MODELS_02-27.md](PROJECT_STATE_AND_BEST_MODELS_02-27.md) → Implementation status.

---

## Summary table (all models)

| Letter | Name | In default ensemble? | Artifact(s) | Script / config |
|--------|------|------------------------|-------------|------------------|
| **A** | Deep Set | Yes | `best_deep_set.pt` | 3_train_model_a |
| **B** | XGBoost | Yes | `xgb_model.joblib` | 4_train_models_b_and_c |
| **C** | Random Forest | No (analytics only) | `rf_model.joblib` | 4_train_models_b_and_c (`training.train_model_c`) |
| — | RidgeCV meta | Yes | `ridgecv_meta.joblib` | 4b_train_stacking |
| **D** | Linear Regression (team-stats) | No (extra) | `linreg_model.joblib` | 4_train_models_b_and_c (`training.train_team_stats_linear`) |
| **E** | Bayesian Ridge | No (extra) | `bayesian_ridge_model.joblib` | 4_train_models_b_and_c (`training.train_team_stats_bayesian_ridge`) |
| **F** | GPR | No (extra) | `gpr_model.joblib` | 4_train_models_b_and_c (`training.train_team_stats_gpr`) |
| **G** | GMM | No (extra) | `gmm_rank_model.joblib` | 4_train_models_b_and_c (`training.train_team_stats_gmm`) |
| **H** | Logistic Regression (clone) | No | — | 4c_train_classifier_clone |

**Inference (script 6)** runs on **any subset of models**: it loads whatever artifacts exist under the outputs dir (Model A, B, C, meta, linreg, bayesian_ridge, gpr, gmm). When both A and B are present, ensemble uses the meta-learner; otherwise the ensemble is the mean of available model scores (each normalized to [0,1]). So you can run inference with only Model A, only XGB, only a team-stats model, or any combination. Extra models’ scores are included in predictions and evaluation for comparison.

**Memoization:** All training and inference steps use memoization for speed where possible (batch cache in script 3, feature cache in script 4, 5b, 4c, plot scripts, and inference). See [MEMOIZATION.md](MEMOIZATION.md).
