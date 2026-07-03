# Output folder naming (model-based)

Output folders under **`output/`** use **model-based names** so you can tell at a glance which model/run produced the results. For a **full list of all models** (A, B, C, stacking, team-stats, logistic regression) and their config/artifacts, see **[MODELS.md](MODELS.md)**.

| Folder | Model / meaning |
|--------|-----------------|
| 0_outputs | Generic / legacy outputs root. |
| 2_listmle | ListMLE runs (run_020/021/022). |
| 3_listmle | ListMLE baseline and Phase I sweeps. |
| 4_listmle | Production ListMLE (default in config). |
| 5_listmle | ListMLE outcome vs standings comparison. |
| 6_baseline | Phase 1 outcome baseline; **project baseline** for comparisons. |
| 7_listmle | ListMLE Optuna sweep (best Spearman combo 17). |
| **8_spearman_surrogate** | **Best** — Spearman-surrogate loss, canonical sweep (combo_0033, etc.). |
| 8_spearman_surrogate_sweep | Spearman-surrogate sweep (e.g. feature_subset_model_a batch). |
| 9_listmle | ListMLE sweep, Spearman objective. |
| 10_spearman_surrogate_standing_rank | Spearman surrogate + standing rank as input. |
| 11_listmle | ListMLE runs. |
| 11_listmle_standing_rank | ListMLE sweep + standing rank. |
| 12_baseline_standing_rank | Baseline-style config + standing rank (single run). |
| 13_legacy | Legacy outputs13. |
| 13_rmse_surrogate | Rank RMSE surrogate sweep. |
| 14_legacy | Legacy outputs14. |
| 14_map_run | MAP branch runs. |
| 15_rmse_surrogate_standing_rank | RMSE surrogate sweep + standing rank. |
| 16_map_standing_rank | MAP branch run + standing rank. |
| player_game | Per-game prediction artifacts. |
| team_stats_spearman_surrogate | Team-stats + Spearman surrogate. |
| team_stats_listmle | Team-stats + ListMLE (branch feature/team-stats-listmle). |
| team_stats_linear_regression | Team-stats track: Linear Regression (config: `training.train_team_stats_linear`). |
| team_stats_bayesian_ridge | Team-stats track: BayesianRidge (config: `training.train_team_stats_bayesian_ridge`). |
| team_stats_gpr | Team-stats track: Gaussian Process Regression, kernel sweep (config: `training.train_team_stats_gpr`). |
| team_stats_gmm | Team-stats track: Gaussian Mixture Model (config: `training.train_team_stats_gmm`). |
| logistic_regression | Logistic Regression (e.g. Model C–style diagnostics / binarized strength; `src/models/lr_model.py`). |

Configs set `paths.outputs` to these paths (e.g. `output/8_spearman_surrogate`, `output/4_listmle`).
