# Metrics used in the project

Documentation of metrics added for model inputs and evaluation (see NBA_ANALYST_METRICS and **outputs/ANALYSIS.md** for current run_026 results and playoff/rank metrics).

## Player rolling features (Model A)

- **TS% (True Shooting %):** `PTS / (2 * (FGA + 0.44 * FTA))` per rolling window (L10, L30). Computed in [src/features/rolling.py](../src/features/rolling.py); columns `ts_pct_L10`, `ts_pct_L30`.
- **Usage rate:** `FGA + 0.44 * FTA + TOV` per game over rolling window (L10, L30). Columns `usage_L10`, `usage_L30`.
- **Stat dim:** With TS% and usage, Model A stat_dim is 21 (18 base L10+L30 + 2 on_court_pm_approx + 1 pct_min_returning).

## Playoff contribution (evaluation)

- **Game Score (per player):** `PTS + 0.5*REB + 1.5*AST + 2*STL + 2*BLK - TOV` per game; aggregated by (player_id, team_id, season). Implemented in `compute_playoff_contribution_per_player` in [src/evaluation/playoffs.py](../src/evaluation/playoffs.py). Returns `contribution_per_game` and `total_contribution`.
- **BPM-style:** Not yet implemented; could be added as an alternative to Game Score using box-score regression (see NBA_ANALYST_METRICS).

## Rank-distance metrics (evaluation)

- **rank_mae**: Mean absolute error of predicted vs actual rank (1=best, 30=worst). Lower = better. Usage: `pred_vs_playoff_outcome_rank` (model predictions vs Playoff Outcome Rank) and `wl_record_standings_vs_playoff_outcome_rank` (baseline: W/L record standings vs Playoff Outcome Rank). Implemented in [src/evaluation/metrics.py](../src/evaluation/metrics.py).
- **rank_rmse**: Root mean squared error of rank predictions. Penalizes large errors more than MAE. Lower = better. Same usage as rank_mae.
- **Per-model MAE and RMSE:** Script 5 computes `rank_mae_pred_vs_playoff_outcome_rank` and `rank_rmse_pred_vs_playoff_outcome_rank` for the ensemble and for each model (Model A, XGB, RF) so each is scored against the same final outcome ranks.
- **Model vs standings comparison:** Evaluation compares each model to regular-season W/L standings on the **same** outcome ranks: standings MAE/RMSE vs outcome, each model’s MAE/RMSE vs outcome, and improvement (standings_error − model_error). Positive improvement = model better than standings.
- **Statistical significance:** Paired bootstrap over teams: resample teams with replacement, compute mean(standings_ae − model_ae) per replicate; 95% CI and p-value (H0: no improvement). Reported in `eval_report.json` under `model_vs_standings_comparison.significance` and in `ANALYSIS_*.md`.

## ListMLE and position-aware loss (Model A training)

- **ListMLE:** Default training loss for Model A: negative log-likelihood of the observed permutation under the Plackett–Luce model; implemented in [src/models/listmle_loss.py](../src/models/listmle_loss.py).
- **Position-aware ListMLE:** When `loss_type: listmle` and `training.listmle_position_aware: true`, the ListMLE loss is weighted by position (e.g. `1/log2(i+2)` for `listmle_position_discount: "log2"`), so top positions contribute more to the loss, aligning training with NDCG-style evaluation. Options: `log2` (NDCG-style), `linear`, or `none`. Default is off (`listmle_position_aware: false`).

## NDCG variants

- **ndcg / ndcg_at_30**: NDCG@30 — full ranking quality over all 30 teams. Main `ndcg` key uses k=30. Relevance = strength (higher = better). Implemented in [src/evaluation/metrics.py](../src/evaluation/metrics.py).
- **NDCG cutoff labels** (what each k represents):
  - **ndcg_at_4** — Conference Finals (top 4)
  - **ndcg_at_12** — Clinch Playoff (top 12)
  - **ndcg_at_16** — One Play-In Tournament (top 16)
  - **ndcg_at_20** — Qualify for Playoffs (top 20)
  - **ndcg_at_30** — full order (all 30)

## Calibration and Phase 3 metrics

- **ECE (Expected Calibration Error):** Implemented in [src/evaluation/metrics.py](../src/evaluation/metrics.py) as `ece(y_true, y_prob, n_bins=10)`. Use with championship probabilities vs one-hot champion label.
- **Platt scaling:** Two variants in [src/models/calibration.py](../src/models/calibration.py): (1) on meta-learner output, (2) on raw model outputs then combine. See [docs/PLATT_CALIBRATION_02-01.md](PLATT_CALIBRATION_02-01.md).
- **Pythagorean expectation:** Per comprehensive plan, Win_Pyth = PointsScored^13.91 / (PointsScored^13.91 + PointsAllowed^13.91). Regression candidate: Actual Win% − Pythagorean Win%. Not yet in team_context; add when needed.
