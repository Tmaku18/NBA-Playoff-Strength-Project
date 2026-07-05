# NBA "True Strength" Prediction

**Attention-Based Deep Set Network with Ensemble Validation**

Tanaka Makuvaza  
Georgia State University — Advanced Machine Learning  
January 28, 2026

---

## Overview
This project builds a **Multi-Modal Stacking Ensemble** to predict NBA **True Team Strength** using a Deep Set roster model (Model A) plus XGBoost (Model B), with optional team-stats models (Model D: Linear Regression, E: Bayesian Ridge, F: GPR, G: GMM), Model H (Logistic Regression clone classifier), and stacking (RidgeCV). The **ensemble** is Model A + Model B only; Model C (Random Forest) is optional and not in the default ensemble. **Full list of all models:** [docs/MODELS.md](docs/MODELS.md).

**Model lineup (baseline vs best vs experiments):**

| Role | Output | Description |
|------|--------|-------------|
| **Baseline** | **6_baseline** | Phase 1 outcome runs (ListMLE, playoff_outcome, rolling [15,30]; configs from 4_listmle). Reference for all comparisons. |
| **Best (fair eval)** | **8_spearman_surrogate / improved_07-03** | **Pipeline deep-dive retrain** (Jul 2026): season-scoped features, causal OOF, seed averaging, rank-transform meta, fair eval (6 checkpoints). **Pooled ensemble Spearman 0.750**, Model B 0.760. See **`output/8_spearman_surrogate/improved_07-03/ANALYSIS_02.md`**. |
| **Best (legacy sweep)** | **8_spearman_surrogate / combo_0033** | Spearman-surrogate loss, playoff_outcome, 40 Optuna trials. Best on **old** eval (single snapshot, corrupted features, unfair standings baseline). Superseded for production conclusions by **improved_07-03** under fair eval. |
| **MAP run** | **14_map_run** | MAP branch model tested (mixed): stronger top-focused NDCG/champion placement, weaker Spearman/rank error than 8_spearman_surrogate. Config: `config/outputs14_map_run.yaml`. See [docs/OUTPUTS14_MAP_ANALYSIS.md](docs/OUTPUTS14_MAP_ANALYSIS.md), [docs/OUTPUTS14_MAP_RUN.md](docs/OUTPUTS14_MAP_RUN.md). |
| **RMSE surrogate** | **13_rmse_surrogate** | **rank_rmse_surrogate** sweep; same setup as 8_spearman_surrogate, different loss. Config: `config/outputs13_sweep_rmse_surrogate.yaml`; compare to 8_spearman_surrogate. |
| **Baseline-style + standing rank** | **12_baseline_standing_rank** | Same as baseline config (ListMLE, final_rank, [15,30]) but **standing rank as input** (stat_dim 22). Single run: `config/outputs4_baseline_standing_rank.yaml` → `output/12_baseline_standing_rank/baseline_standing_rank`. Compare to baseline (6_baseline). |
| **ListMLE + standing (sweep)** | **11_listmle_standing_rank** | ListMLE sweep with standing rank (Optuna). Compare to **baseline (6_baseline)** and 10_spearman_surrogate_standing_rank. |

Outputs retention and gitignore: disposable output roots (all except **8_spearman_surrogate**) have their *files* gitignored; folders are kept via `.gitkeep` so future runs can overwrite. See **[docs/OUTPUTS_RETENTION_NOTES.md](docs/OUTPUTS_RETENTION_NOTES.md)** for the retention run summary and policy.

Each output folder has a **`MODEL.md`** that describes the model that produced those results. Folder names are **model-based** (e.g. `6_baseline`, `8_spearman_surrogate`, `7_listmle`). See [docs/OUTPUT_FOLDER_NAMING.md](docs/OUTPUT_FOLDER_NAMING.md).

**Output folder structure and configs:** All output roots live under a single parent **`output/`** with model-based subfolder names (e.g. `output/4_listmle`, `output/8_spearman_surrogate`). The default production outputs path in config is **`output/4_listmle`**. Branch configs align with best sweeps: **ListMLE branch** (`config/team_stats_listmle.yaml`) matches **7_listmle** best (combo 17; playoff_outcome, [10,30]); **Spearman surrogate branch** (`config/team_stats_spearman_surrogate.yaml`) matches **8_spearman_surrogate** best (combo_0033) plus team-stats (stat_dim 27, use_team_stats). Paths are set via `paths.outputs` in each config.

**All models (canonical list):** **[docs/MODELS.md](docs/MODELS.md)** — Model A (Deep Set), B (XGBoost), C (RF), RidgeCV meta, D (Linear Regression), E (Bayesian Ridge), F (GPR), G (GMM), H (Logistic Regression clone), calibration/confidence.

Full lineup and run commands: **[docs/MODEL_LINEUP_AND_NEXT_STEPS.md](docs/MODEL_LINEUP_AND_NEXT_STEPS.md)**. Official best configs: **[docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md)** and **[docs/OUTPUTS8_SWEEP_ANALYSIS_02-17.md](docs/OUTPUTS8_SWEEP_ANALYSIS_02-17.md)**. MAP findings: **[docs/OUTPUTS14_MAP_ANALYSIS.md](docs/OUTPUTS14_MAP_ANALYSIS.md)**. **Current-state analysis and best-model ranking (Feb 27, 2026):** **[docs/PROJECT_STATE_AND_BEST_MODELS_02-27.md](docs/PROJECT_STATE_AND_BEST_MODELS_02-27.md)**.

**Pipeline deep-dive (Jul 2026):** Fixed feature corruption, leakage, OOF validity, and evaluation fairness. First validated retrain: **`output/8_spearman_surrogate/improved_07-03/`** — pooled Spearman **0.750** (6 checkpoints, fair standings baseline). Full write-up: **[output/8_spearman_surrogate/improved_07-03/ANALYSIS_02.md](output/8_spearman_surrogate/improved_07-03/ANALYSIS_02.md)**. Config: **`config/8_spearman_improved.yaml`** (`team_rolling` + `injury` enabled for next run → `improved_07-05`).

```bash
# WSL — full retrain (RTX 4060; torch.compile auto-skipped when path has spaces)
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project"
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
python -m scripts.run_pipeline_from_model_a --config config/8_spearman_improved.yaml \
  --outputs output/8_spearman_surrogate/improved_07-05/outputs
```

**Legacy production defaults (4_listmle):** (1) **Standings** — `config/defaults.yaml` (Phase 3 fine NDCG@16 combo 18): listmle_target `final_rank` (playoff standings); combo path `output/4_listmle/sweeps/phase3_fine_ndcg16_final_rank/combo_0018/config.yaml`. (2) **Playoff-outcome** — `config/defaults_playoff_outcome.yaml` (Phase 5 combo 2): listmle_target `playoff_outcome`; use `--config config/defaults_playoff_outcome.yaml` for pipeline or inference. **Latest production run (4_listmle):** run_026 — ensemble NDCG@10 0.485, Spearman 0.477, playoff Spearman 0.467, NDCG@16 0.527 (eval: eos_final_rank; 141 train / 36 test dates). **ListMLE outcome vs standings:** In **output/5_listmle/** we compared training Model A (ListMLE) on `playoff_outcome` vs `final_rank` (standings); when evaluated on playoff outcome, **standings-trained** ListMLE matched or beat outcome-trained (e.g. ndcg_standing: Spearman 0.529, playoff Spearman 0.531 vs ndcg_outcome 0.491 / 0.475). See **`output/0_outputs/ANALYSIS.md`** for run_026 details. **Run 021/022** are baseline full-pipeline runs (output/2_listmle). **Sweep strategy:** Run separate Optuna sweeps with `--objective spearman`, `--objective ndcg4`, `--objective ndcg16`, `--objective playoff_spearman`, or `--objective rank_rmse`, then compare best configs. **output/4_listmle/** is the default production root (run_025, run_026); **output/5_listmle/** holds outcome vs standings comparison. See **`output/4_listmle/sweeps/SWEEP_PHASE1_ANALYSIS.md`**, **`docs/SWEEP_ANALYSIS.md`**, and **`output/2_listmle/run_022/RESULTS_AND_OUTPUTS_EXPLAINED.md`** for sweep and metric definitions.

**Best runs (historical):** **`docs/BEST_METRICS_02-15.md`** documents exact configs and numbers; **6_baseline** phase_1 runs are superseded by **8_spearman_surrogate** for official best.

---

## Branches (summary)

| Branch | Description |
|--------|-------------|
| **main** | Production baseline. **Standing rank** (`standing_rank_norm`, global 1–30) is an input feature for **Model B (XGB)** / **Model C (RF)** via team-context features; **Model A no longer uses standing rank**. Model A `stat_dim` is **24** (no standing). Planned (not yet implemented): conference-specific rank 1–15, train East/West separately, then use existing finals logic. See [docs/STANDING_RANK_FEATURE.md](docs/STANDING_RANK_FEATURE.md). |
| **feature/listmle-position-aware** | Legacy experiment branch (now archived): position-aware ListMLE (position discount in loss). Standing-rank input has been moved to `main`. See `docs/BRANCH_FEATURE_LISTMLE_POSITION_AWARE_SUMMARY.md`. |

---

## Key Design Choices
- **Target:** Future W/L (next 5) or Final Playoff Seed — **never** efficiency.
- **True Strength:** Model A produces a latent **Z** (penultimate layer); the **output** `ensemble_score` is the **ensemble** score (RidgeCV blend of A + XGB only) mapped to percentile (0–1 and 0–100). Model C (RF) is not trained in the pipeline by default; when present, it is for analytics/comparison only.
- **No Net Rating leakage:** `net_rating` is excluded as a model input and never used as a target or evaluation metric (allowed only in baselines).
- **Standing rank as input:** Current regular-season standing (rank 1–30) is an **input feature** for **team-context models** (Model B / Model C) via `standing_rank_norm`. Model A does **not** use standing rank. See [docs/STANDING_RANK_FEATURE.md](docs/STANDING_RANK_FEATURE.md) and § Branches.
- **Stacking:** K-fold **OOF** across **all training seasons**; level-2 **RidgeCV** on pooled OOF (2 columns: A + XGB, or **4 columns** when confidence is enabled: \( s_A, s_X, c_A, c_X \)).
- **Per-instance confidence:** When enabled, Model A confidence comes from **attention entropy** (high entropy = diffuse = high confidence) and **max attention weight** (high max = star-dependent = high risk = low confidence). XGB confidence comes from **tree-level prediction variance** (high variance = low confidence). The meta-learner is trained on these 4 inputs so the more confident model has higher effective weight per team. See [docs/CONFIDENCE_WEIGHTED_ENSEMBLE_OPTIONS.md](docs/CONFIDENCE_WEIGHTED_ENSEMBLE_OPTIONS.md).
- **Game-level ListMLE:** lists per conference-date/week; **torch.logsumexp** and input clamping for numerical stability; gradient clipping in Model A training; hash-trick embeddings for new players.
- **Model A training:** epochs configurable via `model_a.epochs` with optional validation-based early stopping (`early_stopping_*` in `defaults.yaml`). Learning rate and gradient clipping are configurable (`model_a.learning_rate`, `model_a.grad_clip_max_norm`). Set attention uses **σReparam** on Q/K/V projections (Zhai et al., [arXiv:2303.06296](https://arxiv.org/abs/2303.06296)) to bound attention logits and reduce entropy collapse.
- **Model A and attention collapse:** If training stops with "Model A is not learning" (flat loss), see [.cursor/plans/Attention_Report.md](.cursor/plans/Attention_Report.md) for investigation steps, references, and diagnostics. Enable `model_a.attention_debug: true` to log encoder/Z/scores, gradient norms, relevance, and player_stats; try different `learning_rate` or `grad_clip_max_norm` (e.g. 5.0) if flat loss may be due to over-clipping.
- **Model A AMP:** Automatic Mixed Precision (AMP) is enabled by default on CUDA (`model_a.use_amp: true`) for faster training; the ListMLE loss stays in float32. Disable with `model_a.use_amp: false` if numerical issues appear.
- **Attention analysis:** [docs/ANALYSIS_OF_ATTENTION_WEIGHTS.md](docs/ANALYSIS_OF_ATTENTION_WEIGHTS.md) — walkthrough of Model A attention, first working inference (run_023), key inferences (star-dominant vs. distributed), and a framework for tracking hyperparameters → metrics by model and conference. Updated with each run/sweep for comprehensive analysis by project end.
- **Season config:** Hard-coded season date ranges in `defaults.yaml` to avoid play-in ambiguity.
- **Explainability:** SHAP on Model B only; Integrated Gradients or permutation ablation for Model A.

---

## Data Sources
- **nba_api** (official): play-by-play, player/team logs, tracking data; **playoff** game logs (SeasonType=Playoffs) for validation. Play-In games are excluded from playoff win counts.
- **Kaggle (Wyatt Walsh):** **primary** for SOS/SRS and historical validation.
- **Basketball-Reference:** **fallback** for SOS/SRS when Kaggle unavailable.
- **Proxy SOS:** If both are unavailable, compute from internal DB (e.g. opponent win-rate) and document.

**Storage:** DuckDB preferred (regular-season + separate playoff tables: `playoff_games`, `playoff_team_game_logs`, `playoff_player_game_logs`).

**RAPTOR (FiveThirtyEight):** Optional. Place `modern_RAPTOR_by_team.csv` or `historical_RAPTOR_by_player.csv` in `docs/` or `data/`. Enable with `raptor.enabled: true` in config. Adds `raptor_offense_sum_top5` and `raptor_defense_sum_top5` to Model B team context. Download: `python -m scripts.1c_download_raptor` or from [GitHub](https://github.com/fivethirtyeight/data/tree/master/nba-raptor).

**LEBRON (future):** BBall Index proprietary metric. Manual CSV export from [BBall Index](https://www.bball-index.com/lebron-database/) or the free Google Sheets database. Place in `data/raw/lebron/` with schema: player_id or player_name, season, lebron_total, o_lebron, d_lebron. Loader and team_context wiring to be added when file present.

---

## Playoff performance rank (ground truth)
Used for training (optional) and evaluation when playoff data exists. **Phase 1:** Rank playoff teams by total playoff wins (desc). **Phase 2:** Tie-break by regular-season win %. **Phase 3:** Non-playoff teams are ranked 17–30 by regular-season win %. Config: `training.target_rank: standings | playoffs` (default `standings`). Playoff rank is computed from `playoff_team_game_logs` + `playoff_games` using **season date ranges** from `defaults.yaml`; it will be null if playoff data is missing or incomplete for the target season.

## Evaluation
- **Ranking:** NDCG, Spearman, MRR (mrr_top2 = champion+runner-up, mrr_top4 = conference finals), Precision@4, Precision@2 (planned). (Previously MRR used top_k=2 for two-conference “rank 1”).
- **Future outcomes:** Brier score.
- **Sleeper detection:** ROC-AUC on upsets (sleeper = actual rank worse than predicted rank); constant-label guard returns 0.5.
- **Playoff metrics** (when playoff data and predictions include post_playoff_rank): Spearman (predicted global rank vs playoff_final_results), NDCG@4 (final four), NDCG@10, Brier score on championship odds (one-hot champion vs predicted odds), rank_mae and rank_rmse (pred vs playoff_final_results; eos_standings vs playoff_final_results baseline). Section `playoff_metrics` in `eval_report.json`.
- **Report:** `eval_report.json` includes `notes` and, when applicable, `playoff_metrics`. **Train / validation / test accuracy** for all models (ensemble, Model A, Model B, Model C) are tracked as `train_metrics_*`, `val_metrics_*`, and `test_metrics_*` when inference is run with `inference.also_train_predictions: true` and `inference.also_validation_predictions: true` (ranking metrics: Spearman, NDCG, rank_mae, rank_rmse, playoff_metrics).
- **Baselines:** rank-by-SRS, rank-by-Net-Rating, **Dummy** (e.g. previous-season rank or rank-by-net-rating).

---

## Outputs (per run)
- **Conference rank** (1–15 within East/West), **ensemble score** (0–1 and 0–100; derived from the stacked ensemble, not Model A’s Z alone), **championship odds** (softmax with `output.odds_temperature`). Predicted strength (global rank 1–30) is used internally for evaluation but not exposed as an output key.
- **actual_conference_rank** (Actual Conference Rank) in outputs and plots is conference standing (1–15 within East or West), from standings-to-date at the inference target date. When available, `EOS_global_rank` (1–30) is included for global evaluation/classification.
- **post_playoff_rank** and **rank_delta_playoffs** (when playoff data exists for the target season).
- Classification: **Over-ranked**, **Under-ranked**, **Aligned**.
- Delta (actual conference rank − predicted league rank) and ensemble agreement (Model A / XGB / RF ranks).
- Roster dependence (attention weights; IG contributors when enabled via `output.ig_inference_top_k` and Captum). `contributors_are_fallback` indicates when attention weights were not usable.
- **Plots:** `pred_vs_actual.png` — two panels (East/West), Predicted Conference Rank vs Actual Conference Rank (1–15); `pred_vs_playoff_final_results.png` — predicted strength (global rank) vs playoff_final_results (1–30); `title_contender_scatter.png` — championship odds vs regular-season wins; `odds_top10.png` — top-10 championship odds bar chart.
- **Report assets:** **output/0_outputs/ANALYSIS.md** (current results run_026, split, ListMLE outcome vs standings). All output folders live under **`output/`**. **output/2_listmle/** run_020/021/022; **output/3_listmle/** baseline and Phase I sweeps; **output/4_listmle/** production (run_025, run_026); **output/5_listmle/** outcome vs standings; **output/6_baseline/** baseline; **output/7_listmle/** ListMLE sweep; **output/8_spearman_surrogate/** best (Spearman surrogate); **output/9_listmle/** ListMLE sweep; **output/10_spearman_surrogate_standing_rank/** Spearman+standing; **output/11_listmle_standing_rank/** ListMLE+standing; **output/12_baseline_standing_rank/** baseline+standing; **output/13_rmse_surrogate/** RMSE surrogate; **output/14_map_run/** MAP run. Each of **output/0_outputs/** through **output/14_map_run/** (and other model folders) may have a **`MODEL.md`** describing the model and how it differs from the previous output folder. See `output/4_listmle/sweeps/SWEEP_PHASE1_ANALYSIS.md` and `docs/SWEEP_ANALYSIS.md` for sweep analysis.

---

## How to Run the Pipeline

**Run order (production, real data only):**

1. **Setup:** `pip install -r requirements.txt`
2. **Config:** Edit `config/defaults.yaml` if needed (seasons, paths, model params, `build_db.skip_if_exists`, `inference.run_id`). **DB:** `config.paths.db` should point to a DB with playoff data (`playoff_games`, `playoff_team_game_logs`) for sweeps and `--objective playoff_spearman`. From any worktree, set env `NBA_DB_PATH` to that DB path so the sweep uses it (e.g. the main project DB). **Production runs use `output/4_listmle/`** (`config.paths.outputs`). The first run in an empty outputs folder can start at `run_001` (or use `inference.run_id_base`); with `inference.run_id: null` (default) runs auto-increment.
3. **Data:**  
   - `python -m scripts.1_download_raw` — fetch regular-season and playoff logs via nba_api (writes to `data/raw/`; reuses existing files when present).  
   - `python -m scripts.2_build_db` — build DuckDB from raw → path in `config.paths.db` (e.g. `nba_build.duckdb`), update `data/manifest.json`. If `build_db.skip_if_exists: true` (default) and the DB file already exists, the build is skipped to keep the current DB.
4. **Training (real DB):**  
   - `python -m scripts.3_train_model_a` — K-fold OOF → outputs dir `oof_model_a.parquet`, then final model → `best_deep_set.pt`.  
   - `python -m scripts.4_train_models_b_and_c` — K-fold OOF → outputs dir `oof_model_b.parquet`, then Model B (XGBoost) + Model C (RF) → `xgb_model.joblib`, `rf_model.joblib`.  
   - `python -m scripts.4b_train_stacking` — merge OOF parquets, RidgeCV → outputs dir `ridgecv_meta.joblib`, `oof_pooled.parquet` (requires OOF from 3 and 4).
5. **Inference:** `python -m scripts.6_run_inference` — load DB and models, run Model A/B + meta → outputs dir `<run_id>/predictions.json`, plots. With `inference.run_id: null`, run_id auto-increments (e.g. run_019, run_020 when using `output/2_listmle/` and `run_id_base: 19`) so each full pipeline run gets a new folder.
6. **Evaluation:** `python -m scripts.5_evaluate` — uses predictions from the latest (or configured) run_id → outputs dir `eval_report.json` (NDCG, Spearman, MRR, ROC-AUC upset).
7. **Explainability:** `python -m scripts.5b_explain` (uses `config/defaults.yaml`) or `python -m scripts.5b_explain --config path/to/config.yaml` — SHAP on Model B → `shap_summary.png`; attention ablation and Integrated Gradients (Model A) when Captum is installed → `ig_model_a_attributions.txt`. Use `--config` to run explain on a **sweep best combo** (e.g. `output/4_listmle/sweeps/phase3_fine_ndcg16_final_rank/combo_0018/config.yaml`).
8. **Clone classifier (optional):** `python -m scripts.4c_train_classifier_clone --config config/clone_classifier.yaml` — XGBoost binary classifier (playoff team vs not) on Train 2015–2022, Val 2023, Holdout 2024; outputs `clone_classifier_report.json` (AUC-ROC, Brier).
9. **Hyperparameter sweep:** `python -m scripts.sweep_hparams` — Runs full pipeline (3, 4, 4b, 6, 5) per combo; writes to **`output/4_listmle/sweeps/<batch_id>/`** (use `--config config/outputs4_phase1.yaml`). Use `--method optuna --n-trials N` for Bayesian tuning. **`--objective spearman|ndcg4|ndcg16|ndcg20|playoff_spearman|rank_rmse`** sets which metric Optuna optimizes (run separate sweeps for each objective and compare). Use semicolons in PowerShell to run sweeps sequentially; see [Outputs4 Phase I plan](.cursor/plans/archive/outputs4_phase_i_sweeps_2638c514.plan.md) §2.4. After the sweep, **explain (5b_explain) runs automatically** on the best combo unless `--no-run-explain`. Use `--dry-run` to preview combos, `--max-combos N` to limit (grid only).

**Optional:** `python -m scripts.run_manifest` (run manifest); `python -m scripts.run_leakage_tests` (before training); `python -m scripts.1b_download_injuries` (stub for injury data); `python -m scripts.1c_download_raptor` (RAPTOR CSV from FiveThirtyEight).

**Pipeline behavior:** Script 1 reuses raw files that already exist (no re-download). Script 2 skips rebuilding the DB when `build_db.skip_if_exists` is true and the DB file exists; set it to false to force a full rebuild from raw. With `inference.run_id: null`, inference writes to the next run folder (run_002, run_003, …) and evaluation uses the latest run.

**How to speed up sweeps:**
- **Parallelism:** Use `--n-jobs 4` (or higher on multi-core machines). Wall time ≈ (trials / n_jobs) × per-trial time (e.g. 20 trials at 15 min/trial with n_jobs=4 → ~75 min).
- **Fewer trials:** Start with `--n-trials 6` for exploratory sweeps; increase to 20 for Phase 1. Optuna TPE learns from fewer evaluations than grid.
- **Phase baseline:** Use `--phase baseline` with wide ranges and few trials to quickly explore; narrow ranges and increase trials for Phase 1.
- **Skip explain:** Use `--no-run-explain` to skip SHAP/IG on the best combo and save ~5–10 min per sweep.
- **Config caps:** In `baseline_max_features.yaml`, lower `max_lists_oof` and `max_final_batches` (e.g. 50) for faster Model A training at the cost of coverage.
- **Fixed rolling_windows:** Phase1, baseline, phase1_xgb, phase2_rf keep `rolling_windows: [10, 30]` only, maximizing batch cache hits. Test different rolling windows last with `--phase rolling` after main sweeps.

**Built-in optimizations (automatic):**

| Optimization | What it does | Speed effect |
|--------------|--------------|--------------|
| **AMP (Model A)** | Automatic Mixed Precision: model forward runs in float16 on CUDA; loss stays float32. Config: `model_a.use_amp: true` (default). | ~20–35% faster per epoch; ~1–2.5 min saved per trial on GPU. |
| **Batch cache (script 3)** | Caches built lists and batches (keyed by config + DB); reused when same listmle_target, rolling_windows, train_seasons, etc. Config: `paths.batch_cache` (default `data/processed/batch_cache`). Set to `null` to disable. | First trial builds; subsequent trials skip list/batch building → ~1–3 min saved per trial. 12-trial sweep: ~11–33 min saved. |
| **Feature cache (script 4, 5b, 4c, plot, inference)** | Caches `build_team_context_as_of_dates` output (keyed by config + DB + team_dates). Shared module: `src.features.feature_cache`. Config: `paths.feature_cache` (default `data/processed/feature_cache`). Set to `null` to disable. | Script 4, 5b_explain, 4c_train_classifier_clone, plot_feature_rank_* and inference (script 6) all use it; second and later runs skip re-building team-context features. |
| **DB load (in-process)** | `load_training_data` and `load_playoff_data` are memoized by (db_path, mtime) in `src.data.db_loader`. | Repeated calls in the same process (e.g. script 3 then 4 in same run) do not re-read the DB. |
| **Inference feature cache** | `run_inference_from_db` builds team-context features once for the union of all target specs and reuses `paths.feature_cache` (inf_* key). Re-running script 6 with same config/DB loads from cache. | Second and later inference runs skip re-building features; multiple target dates in one run share a single feature build. |

**Memoization (use for speed):** Keep `paths.batch_cache` and `paths.feature_cache` set in config (defaults enable both). All pipeline and analysis scripts use memoization where possible. See [docs/MEMOIZATION.md](docs/MEMOIZATION.md) for the full list.

**Combined effect:** A 12-trial sweep that used to take ~3 hours may complete in ~2–2.5 hours (~15–35% faster) with AMP + batch cache enabled on CUDA. With feature cache enabled, script 4, 5b, 4c, plot scripts, and inference are faster on repeated runs when config/DB/team_dates are unchanged.

**Perfect run checklist (workspace or worktree):**
- **DB:** `config.paths.db` must point to an existing DuckDB with `playoff_games` and `playoff_team_game_logs` (or set `inference.require_eos_final_rank: false` to allow standings-only).
- **Worktree / playoff plots:** Set `NBA_DB_PATH` env to the canonical DB path so inference (script 6) and sweep both use the DB that has playoff data (produces `eos_playoff_standings_vs_eos_global_rank.png`).
- **Outputs:** All scripts use `config.paths.outputs` (default `4_listmle`); scripts 3–6 write models, predictions, and plots there.

**Training notes:** Model A (script 3) subsamples conference-date lists for OOF and final training (`training.max_lists_oof`, `training.max_final_batches`) and `build_lists` subsamples dates (e.g. 200) for speed; use full list set by increasing these in config or adjusting `build_lists`. Configure training length and early stopping with `model_a.epochs`, `model_a.early_stopping_patience`, `model_a.early_stopping_min_delta`, `model_a.early_stopping_val_frac`.

---

## Anti-Leakage Checklist

- [ ] **Time rule:** All features use only rows with `game_date < as_of_date` (strict t-1). Rolling stats use `shift(1)` before aggregation.
- [ ] **Roster:** Minutes and roster selection use only games before `as_of_date`. Rosters use a **latest-team** map (player’s most recent team as of `as_of_date`) so traded players appear only on their current team; season boundaries from config scope games when building rosters.
- [ ] **Model B:** Feature set must **not** include `net_rating`. Enforced in `src.features.team_context.FORBIDDEN` and `train_model_b`.
- [ ] **ListMLE:** Targets configurable: `listmle_target: final_rank` (playoff standings = EOS reg-season rank; production default), `playoff_outcome` (playoff outcome = champion=1, runner-up=2), or `standings` (win-rate to date). Production uses `final_rank` (phase3 combo 18). Evaluation remains season-end.
- [ ] **Baselines only:** Net Rating is used only in `rank-by-Net-Rating` baseline, computed from off/def ratings, never as a model input.

---

## Report Assets (deliverables)

All paths under the configured outputs dir. **output/0_outputs/** holds ANALYSIS.md (current results). **output/2_listmle/** run_020/021/022; **output/3_listmle/** baseline sweeps; **output/4_listmle/** production baseline (run_025, run_026, sweeps, combo 18); **output/5_listmle/** ListMLE outcome vs standings; **output/6_baseline/** baseline; **output/7_listmle/** ListMLE sweep; **output/8_spearman_surrogate/** best; **output/9_listmle/** ListMLE sweep; **output/10_spearman_surrogate_standing_rank/** Spearman+standing; **output/11_listmle_standing_rank/** ListMLE+standing; **output/12_baseline_standing_rank/** baseline+standing; **output/13_rmse_surrogate/** RMSE surrogate; **output/14_map_run/** MAP run. **Each output folder has a `MODEL.md`** that describes the model that produced those results and the difference from the previous output (see [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](docs/MODEL_LINEUP_AND_NEXT_STEPS.md)). With `inference.run_id: null`, each pipeline run writes to a new folder; evaluation uses the latest run.

- `eval_report.json` — NDCG, Spearman, mrr_top2, mrr_top4, ROC-AUC upset, `notes`; per-model metrics (ensemble, model_a, xgb, rf); per-conference (predicted vs actual conference rank). When playoff data exists, `playoff_metrics`.
- `output/4_listmle/run_026/` — latest production run (EOS `eos_final_rank`): NDCG@10 0.485, Spearman 0.477, playoff Spearman 0.467, NDCG@16 0.527. `output/0_outputs/ANALYSIS.md` has full run_026 breakdown. `output/2_listmle/run_022/` — baseline; see `output/2_listmle/run_022/RESULTS_AND_OUTPUTS_EXPLAINED.md`.
- `output/<outputs_dir>/run_001/predictions.json` (e.g. `output/4_listmle/run_001/`) — per-team `predicted_strength` (rank), `conference_rank` (1–15), `championship_odds`, `ensemble_score` (0–1 percentile), `actual_conference_rank` (Actual Conference Rank), `EOS_global_rank` (1–30 when available), `post_playoff_rank`/`rank_delta_playoffs` (when playoff data exists), classification, ensemble diagnostics (model_agreement: High/Medium/Low), roster_dependence (attention + optional `ig_contributors`).
- `output/<outputs_dir>/run_001/pred_vs_actual.png` — two panels (East/West): predicted vs actual conference rank (1–15); grid lines, team-colored points, legend.
- `output/<outputs_dir>/run_001/pred_vs_playoff_final_results.png` — predicted global rank (1–30) vs playoff_final_results (1–30).
- `output/<outputs_dir>/run_001/title_contender_scatter.png` — championship odds vs regular-season wins (proxy).
- `output/<outputs_dir>/run_001/odds_top10.png` — top-10 championship odds bar chart.
- `output/<outputs_dir>/shap_summary.png` — Model B (XGBoost) SHAP summary on real team-context features (script 5b).
- `output/<outputs_dir>/ig_model_a_attributions.txt` — Model A Integrated Gradients top-5 player indices by attribution L2 norm (script 5b; requires Captum).
- `output/<outputs_dir>/oof_pooled.parquet`, `ridgecv_meta.joblib` — stacking meta-learner and pooled OOF (script 4b).
- `output/<outputs_dir>/clone_classifier_report.json`, `clone_xgb_classifier.joblib` — clone classifier (script 4c).
- `output/<outputs_dir>/oof_model_a.parquet`, `oof_model_b.parquet` — OOF from scripts 3 and 4 (Option A: K-fold, real data).
- `output/<outputs_dir>/best_deep_set.pt`, `xgb_model.joblib`, `rf_model.joblib` — trained Model A, Model B (XGBoost), and Model C (RF).
- `docs/ANALYSIS_OF_ATTENTION_WEIGHTS.md` — Analysis of Model A attention weights: architecture walkthrough, run_023 inferences (star-dominant vs. distributed), hyperparameter/metric tracking framework, conference vs. league-wide. Updated with each run/sweep.
- `docs/SWEEP_ANALYSIS.md` — Phase 2 & Phase 3 sweep analysis: best combos, Optuna importances, progression from run_022 to Phase 3 fine NDCG@16; production default (combo 18).
- `docs/METRIC_MATRIX_EXPLORATION_PLAN.md` — 8-sweep matrix (playoff standings vs playoff outcome × Spearman/NDCG); Phase 2 exploration plan.

---

## Reproducibility

- **Seeds:** `src.utils.repro.set_seeds(seed)` for random, numpy, torch.
- **Manifests:** outputs dir `run_manifest.json` (config snapshot, git hash, data manifest hash). `data/manifest.json` (raw/processed hashes; `db_path` is stored relative to project root for portability).
- **OOF:** `outputs/oof_pooled.parquet` and `ridgecv_meta.joblib`.
- **Season boundaries:** Hard-coded in `config/defaults.yaml` to avoid play-in ambiguity.

### WSL / GPU and reproducibility

Run training and sweeps in **WSL** for CUDA. **`torch.compile` is skipped automatically** when the project or venv path contains spaces (inductor linker limitation on this OneDrive path); training still uses the GPU with AMP.

Running in **WSL (Ubuntu)** with GPU vs **Windows** can yield different results due to CUDA/cuDNN versions, RNG, or numerical precision. To compare:

- Fix seeds (`repro.seed` in config) and run the same config on both environments.
- Compare first-epoch train loss and attention debug stats (`attn_sum_mean`, `attn_max_mean`, `attn_grad_norm`) from `python -m scripts.debug_model_a --dummy` or script 3 with `--attention-debug`.
- Document environment (OS, Python, PyTorch, CUDA) for the run that reproduces run_021-style attention; use the debugger and attention_debug output to confirm attention is learning in both environments before trusting sweep results.

---

## Recent implementation (standing rank + uncertainty intervals)

**Standing rank as input:** `standing_rank_norm` (global 1–30, normalized) is used as an input feature for **team-context models** (Model B / Model C) via `src/features/team_context.py`. **Model A does not use standing rank** (removed from roster-set features); Model A `stat_dim` is **24** in `config/defaults.yaml`. Doc: [docs/STANDING_RANK_FEATURE.md](docs/STANDING_RANK_FEATURE.md).

**Predicted rank intervals:** `predictions.json` now includes `predicted_strength_low/high` (and plus/minus) for the **ensemble**, and `model_*_rank_low/high` for Models A/B/C, plus any extra team-stats models when enabled. Evaluation writes interval coverage/width into `eval_report.json` under `uncertainty_metrics`. See [docs/UNCERTAINTY_INTERVALS.md](docs/UNCERTAINTY_INTERVALS.md).

**Earlier (Update2):**

Implemented per [.cursor/plans/archive/Update2.md](.cursor/plans/archive/Update2.md): **IG batching fix** (Captum auxiliary tensors expanded to match batched inputs); **latest-team roster** (players only on current team as of `as_of_date`); **EOS_global_rank** (1–30) for evaluation/classification; **manifest db_path** stored relative to project root; **conference plot** uses only valid conference ranks (no global fallback); **attention/IG** sanitized so `predictions.json` is valid JSON (`allow_nan=False`); **ensemble agreement** High/Medium/Low with scaled thresholds and handling of missing models; **ensemble_score** percentile reaches 0.0/1.0. Optional: `output.ig_inference_top_k` and `output.ig_inference_steps` control IG in inference outputs.

---

## Planned improvements (next phase)

Planned per [.cursor/plans/archive/centralize_training_config_attention_eval_expansion.plan.md](.cursor/plans/archive/centralize_training_config_attention_eval_expansion.plan.md):

1. **Centralize training loops** — Move core logic (batching, loss, early stopping) into a class-based `BaseTrainer` in `src/training/`. Scripts handle only CLI and orchestration.
2. **Configuration-driven architecture** — Pass a `ModelAConfig` object to model constructors; save architecture DNA with checkpoints to prevent inference mismatches.
3. **Attention debugging as integrated hook** — PyTorch hooks or logging callback to monitor attention entropy; auto-restart on collapse when enabled.
4. **Attention stability metrics** — Track attention variance and entropy across epochs; `train_history.json`; programmatic collapse detection.
5. **Championship win-rate calibration** — Brier score, ECE, reliability diagram for championship odds vs actual outcomes.
6. **Comparative explainability (RFX-Fuse)** — Validate DeepSet high-attention players against RFX-Fuse historical archetypes.
7. **Strength of Schedule (SoS) normalization** — Enable SOS/SRS; add "Net Rating Adjusted for Opponent" feature.

**Analysis tracking:** [docs/ANALYSIS_OF_ATTENTION_WEIGHTS.md](docs/ANALYSIS_OF_ATTENTION_WEIGHTS.md) is the living document for attention analysis and hyperparameter→metric tracking. Each run/sweep will update it with config, metrics, and attention patterns (conference and league-wide) for comprehensive analysis by project end.

---

## Full Plan

See `.cursor/plans/Plan.md`. Planned extensions: [.cursor/plans/archive/Update1.md](.cursor/plans/archive/Update1.md) through [.cursor/plans/archive/Update8.md](.cursor/plans/archive/Update8.md).

**Comprehensive expansion plan:** [.cursor/plans/archive/comprehensive_feature_and_evaluation_expansion.plan.md](.cursor/plans/archive/comprehensive_feature_and_evaluation_expansion.plan.md) — data evolution (First–Fourth Order metrics), Four Factors, SRS, Pythagorean, Elo, Massey, RAPM-lite, Bayesian Optimization (Optuna) for XGBoost, lineup continuity, fatigue, momentum, calibration (ECE, Platt Scaling), playoff residual model, and hyperparameter tuning strategy with NBA-specific XGBoost ranges.

**Hyperparameter testing evolution:** [docs/HYPERPARAMETER_TESTING_EVOLUTION.md](docs/HYPERPARAMETER_TESTING_EVOLUTION.md) — methodology changes from initial sweeps to grid, Optuna, successive halving, and phased Model B grids; decision steps, pros/cons, and research on optimization speed (with references). **Sweep analysis:** [docs/SWEEP_ANALYSIS.md](docs/SWEEP_ANALYSIS.md) — Phase 2/Phase 3 results, best combos, and production default.

---

## Implementation Roadmap

The full phased development plan is in `.cursor/plans/Plan.md`. Update plans: Update1–Update8. **Comprehensive plan:** `.cursor/plans/archive/comprehensive_feature_and_evaluation_expansion.plan.md` for feature engineering, tuning, and playoff residual architecture.
