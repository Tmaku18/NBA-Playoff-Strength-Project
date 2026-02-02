---
name: All Update8 Next Steps
overview: "Phase 0: extend to L10+L30 stats (14 total), stat_dim=14. Then implement all five Update8 items: (1) hyperparameter sweep, (2) primary contributor fixes, (3) playoff performance learning design/stub, (4) actual playoff contribution, (5) NBA analyst metrics investigation."
todos: []
isProject: false
---

# All Update8 Next Steps

Implement all five items from [.cursor/plans/Update8.md](.cursor/plans/Update8.md) section 4.2.

---

## Phase 0: Extend to L10 + L30 stats (stat_dim=14) — do before anything else

**Goal:** Add the 7 L30 stats to the 7 L10 stats (14 total) and change the model to accept 14 stats so it can use both rolling windows.

**Current state:**

- [src/features/rolling.py](src/features/rolling.py): `PLAYER_STAT_COLS_L10` has 7 cols (pts_L10, reb_L10, ast_L10, stl_L10, blk_L10, tov_L10, availability_L10)
- [src/features/build_roster_set.py](src/features/build_roster_set.py): uses only L10 stat cols
- [src/training/data_model_a.py](src/training/data_model_a.py): `stat_dim=7`, uses `PLAYER_STAT_COLS_L10`
- [src/training/train_model_a.py](src/training/train_model_a.py): `stat_dim=7` hardcoded
- [src/models/deep_set_rank.py](src/models/deep_set_rank.py): `stat_dim` is a constructor parameter

**Changes:**

1. **rolling.py:** Add `PLAYER_STAT_COLS_L30` (pts_L30, reb_L30, ..., availability_L30) and `PLAYER_STAT_COLS_L10_L30` = L10 + L30 (14 cols). Ensure `compute_rolling_stats` / `get_player_stats_as_of_date` produce L30 cols when `training.rolling_windows` includes 30.
2. **build_roster_set.py:** Use `PLAYER_STAT_COLS_L10_L30` (or a config-driven list) for `stat_cols` when building roster tensors.
3. **data_model_a.py:** Use the 14 stat cols; set `stat_dim=14`.
4. **train_model_a.py:** Use `stat_dim=14` (or read from `config["model_a"]["stat_dim"]`).
5. **config/defaults.yaml:** Add `model_a.stat_dim: 14`.
6. **predict.py / load_models:** Pass `stat_dim=14` (or from config) when instantiating DeepSetRank.
7. **Prior-season baseline:** Update `get_prior_season_stats` to handle L30 stat names if needed (map L30 → base stats for aggregation).

**Files:** [src/features/rolling.py](src/features/rolling.py), [src/features/build_roster_set.py](src/features/build_roster_set.py), [src/training/data_model_a.py](src/training/data_model_a.py), [src/training/train_model_a.py](src/training/train_model_a.py), [src/inference/predict.py](src/inference/predict.py), [config/defaults.yaml](config/defaults.yaml).

---

## Phase 1: Full Hyperparameter Sweep

**Goal:** Expand [scripts/sweep_hparams.py](scripts/sweep_hparams.py) to run Model A epoch grid and Model B hyperparameter grid, write results to CSV/JSON, run in foreground with no timeout.

**Config additions** in [config/defaults.yaml](config/defaults.yaml):

- Add `sweep.model_a_epochs: [8, 12, 16, 20, 24, 28]` (or read from plan/refined_sweep)
- Add `sweep.model_b.grid` with XGB/RF candidates (e.g. `max_depth: [3, 4, 5]`, `learning_rate: [0.06, 0.08, 0.10]`, `n_estimators` for RF)
- Add `sweep.rolling_windows: [[5, 10], [10, 20], [10, 30], [15, 30]]` — test different rolling-window combos (L5/L10, L10/L20, L10/L30, L15/L30) for player stats in [src/features/rolling.py](src/features/rolling.py); config key `training.rolling_windows`

**Sweep logic:**

1. Load config; resolve `paths.outputs` → `outputs2/sweeps/<batch_id>/`
2. For each **rolling_windows** combo (e.g. [5,10], [10,30]): override `training.rolling_windows`, then for each Model A epoch value: run script 3 with overridden `model_a.epochs`, save `best_deep_set.pt` to sweep subdir (e.g. `sweep_roll_10_30_a_epochs_12/`), then for each Model B config: run 4, 4b, 6, 5; collect metrics (NDCG, Spearman, MRR, ROC-AUC, final loss)
3. **Per-model and ensemble metrics:** Compute and store metrics for Model A, XGB, RF, and ensemble separately. Use `deep_set_rank`, `xgboost_rank`, `random_forest_rank` from `ensemble_diagnostics` and `ensemble_score` from `prediction`. For each model, compute NDCG, Spearman, MRR, ROC-AUC (use negative rank or 31-rank as score proxy when only rank is available). Label the ensemble/combined metrics clearly as `ensemble` or `combined_prediction`.
4. Write `sweep_results.csv` and `sweep_results.json` with columns: `rolling_windows`, `model_a_epochs`, `model_b_max_depth`, `model_b_lr`, ... `ndcg_model_a`, `spearman_model_a`, `ndcg_xgb`, `spearman_xgb`, `ndcg_rf`, `spearman_rf`, `ndcg_ensemble`, `spearman_ensemble`, `mrr_ensemble`, `roc_auc_ensemble`, `loss` (and per-model mrr/roc_auc if applicable).
5. **Alternating metric priority:** When summarizing sweep results, identify and report the "best" config when optimizing for each metric (per model and ensemble): best by Spearman, best by NDCG, etc. Write `sweep_results_summary.json` with `best_by_spearman`, `best_by_ndcg`, etc. — each containing the top config and its metric values.
6. Subprocess or direct import: prefer importing `train_model_a`, `train_model_b` etc. to avoid full script re-runs; or run `python -m scripts.3_train_model_a` with env/config override. Scripts 3/4 read config from file, so use a temp config or CLI override.

**Implementation approach:** Create sweep config override (e.g. `--config-override` or write temp YAML), run `subprocess.run([sys.executable, "-m", "scripts.3_train_model_a"], env={...})` with modified config path, or refactor scripts 3/4 to accept optional config dict. Simpler: write per-combo config files under `batch_dir/configs/` and run scripts with `--config path`.

**Evaluation report structure** ([scripts/5_evaluate.py](scripts/5_evaluate.py), `eval_report.json`): Add per-model metric sections (`model_a`, `xgb`, `rf`) with ndcg, spearman, mrr, roc_auc for each. Rename/label the current `test_metrics` block as `ensemble` or `combined_prediction` metrics. **Group and order metrics:** ensemble at top, then model_a, then xgb, then rf. Structure: `test_metrics_ensemble`, `test_metrics_model_a`, `test_metrics_xgb`, `test_metrics_rf`, then `test_metrics_by_conference` keyed by model in that same order.

**Per-conference Spearman fix:** The per-conference Spearman should compare **predicted conference rank** vs **actual conference rank** (both 1–15 within East or West), not global rank vs ensemble score. Use `conference_rank` and `actual_conference_rank` from predictions — this fixes the negative per-conference Spearman. Same for per-conference NDCG: use conference ranks (relevance from actual, score from predicted or inverse rank) so the metric is meaningful within each conference.

**Files to touch:** [scripts/sweep_hparams.py](scripts/sweep_hparams.py), [scripts/5_evaluate.py](scripts/5_evaluate.py), [config/defaults.yaml](config/defaults.yaml), optionally [scripts/3_train_model_a.py](scripts/3_train_model_a.py) and [scripts/4_train_model_b.py](scripts/4_train_model_b.py) for `--config` support if not present.

---

## Phase 2: Primary Contributors Fixes

**Goal:** Ensure primary contributor information is displayed and calculated correctly (attention/IG path and fallbacks).

**Current behavior** ([src/inference/predict.py](src/inference/predict.py)):

- `roster_dependence.primary_contributors` from attention weights (top-10 by weight)
- Fallback when `attn_sum <= 0`: use top-k by raw weight even if ≤ 0
- `ig_contributors` added when Captum is installed and `ig_inference_top_k > 0` (per conference)
- `contributors_are_fallback` indicates attention fallback was used

**Likely issues to fix:**

1. **IG not run for all teams:** IG is run per-conference top-k (1 per conf by default). Expand to top-k per team or ensure at least one team per conference gets IG; or document that IG is conference-level.
2. **Empty primary_contributors when attention is all-zero:** Fallback exists but may still yield empty if all weights are NaN/inf. Add second fallback: use roster order (e.g. top-3 by minutes) when attention yields nothing.
3. **Display:** Ensure JSON outputs `primary_contributors` and `ig_contributors` are well-formatted; add to ANALYSIS.md or README how to interpret them.

**Files:** [src/inference/predict.py](src/inference/predict.py) (fallback logic), [outputs/ANALYSIS.md](outputs/ANALYSIS.md) or README (interpretation).

---

## Phase 3: Playoff Performance Learning (Design + Stub)

**Goal:** Design and stub a system to train how player performance changes in the playoffs. Full implementation can be phased.

**Approaches (from Update8):**

- Separate playoff head or fine-tuning on playoff logs (playoff-only ListMLE/regression)
- Regular-season model plus playoff adjustment (residual/linear layer)
- Single model with "playoff" flag or playoff-specific embeddings
- Learning over time: one season at a time (e.g. 2015–16, then 2016–17)

**Deliverables:**

1. Create [.cursor/plans/PlayoffPerformanceLearning.md](.cursor/plans/PlayoffPerformanceLearning.md) documenting approaches, pros/cons, and recommended path (e.g. start with residual layer).
2. Add `scripts/train_playoff_adjustment.py` stub: load `playoff_player_game_logs`, compute per-player playoff vs regular-season stat delta, write `outputs2/playoff_adjustment_stub.json` with placeholder structure. No training yet.

**Data:** [src/data/db_loader.py](src/data/db_loader.py) `load_playoff_data` returns `playoff_games`, `playoff_team_game_logs`, `playoff_player_game_logs`. Schema in [src/data/db_schema.py](src/data/db_schema.py).

---

## Phase 4: Actual Playoff Contribution

**Goal:** Calculate actual playoff contribution using playoff stats. Define formula and implement.

**Formula options:**

- Box-score: `pts + 0.5*reb + 1.5*ast + 2*stl + 2*blk - tov` per game (simple Game Score style)
- Per-possession: normalize by minutes/possessions
- Plus-minus: use `plus_minus` from `playoff_player_game_logs` (already in schema)
- BPM-style: would need regression; start simpler

**Implementation:**

1. Add `compute_playoff_contribution_per_player()` in [src/evaluation/playoffs.py](src/evaluation/playoffs.py): aggregate `playoff_player_game_logs` by `(player_id, team_id, season)`, compute per-game contribution (e.g. Game Score: `PTS + 0.5*REB + 1.5*AST + 2*STL + 2*BLK - TOV`), return DataFrame or dict.
2. Expose in predictions or new output: add `actual_playoff_contribution` (or similar) to analysis dict for playoff teams, or write `outputs2/<run_id>/playoff_contributions.json` keyed by season/team/player.
3. Document formula in Update8 or new plan.

**Files:** [src/evaluation/playoffs.py](src/evaluation/playoffs.py), [src/inference/predict.py](src/inference/predict.py) or new script.

---

## Phase 5: NBA Analyst Metrics Investigation

**Goal:** Investigate metrics NBA analysts use beyond basic box-score; note which could be useful as inputs or analysis.

**Research and document** in [.cursor/plans/Update8.md](.cursor/plans/Update8.md) or new `NBA_ANALYST_METRICS.md`:

- RAPM, BPM, VORP, PIPM, LEBRON, EPM, Win Shares
- Usage rate, TS%, on/off, net rating in playoffs
- Data availability: which can we compute from our DB (games, tgl, pgl) vs require external APIs?
- Recommendations: e.g. TS% and usage rate are computable from our data; BPM/VORP have published formulas; RAPM/EPM need large lineup datasets.

**Output:** Markdown doc with table: Metric | Description | Computable from our DB? | Useful for inputs/analysis?

---

## Summary of Deliverables


| Phase | Deliverable                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0     | L10+L30 stats (14 total), stat_dim=14 in model/config; rolling.py, build_roster_set, data_model_a, train_model_a, predict, config                                                                                                                                                                                                                                                                                    |
| 1     | Full `sweep_hparams.py` with rolling windows + Model A + Model B grids; per-model metrics (model_a, xgb, rf) and ensemble/combined metrics; CSV/JSON results; summary with best-by-metric; `5_evaluate.py` and `eval_report.json` updated with per-model sections (order: ensemble, model_a, xgb, rf), ensemble labeling, and per-conference Spearman/NDCG fix (use predicted vs actual conference rank, not global) |
| 2     | Primary contributor fallback (minutes-based when attention empty), IG coverage fix if needed                                                                                                                                                                                                                                                                                                                         |
| 3     | `PlayoffPerformanceLearning.md` plan + `train_playoff_adjustment.py` stub                                                                                                                                                                                                                                                                                                                                            |
| 4     | `compute_playoff_contribution_per_player()` + output in run or separate file                                                                                                                                                                                                                                                                                                                                         |
| 5     | NBA analyst metrics doc with computability and recommendations                                                                                                                                                                                                                                                                                                                                                       |


---

## Execution Order

1. **Phase 0 (L10+L30 stats, stat_dim=14)** — do first; all subsequent training uses 14 stats
2. Phase 1 (sweep) — highest impact, unblocks hyperparameter tuning
3. Phase 2 (primary contributors) — quick fixes
4. Phase 4 (playoff contribution) — uses existing playoff tables, independent
5. Phase 5 (NBA metrics doc) — documentation only
6. Phase 3 (playoff learning) — design + stub; full implementation in follow-up

