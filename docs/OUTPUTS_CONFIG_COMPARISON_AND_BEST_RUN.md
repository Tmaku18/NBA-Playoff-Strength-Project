# Outputs config comparison and running comparable to best run

**Purpose:** Clarify which runs use which config, why many are **not** comparable to the official best run, and how to run **comparable to best** so you don’t have to guess.

---

## 1. Official best run (reference)

**Path:** `output/8_spearman_surrogate/sweeps/20260217_042955/combo_0033/config.yaml`  
**Doc:** [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md)

| Setting | Best run (combo_0033) |
|--------|------------------------|
| **training** | `loss_type: spearman_surrogate`, `listmle_target: playoff_outcome`, `rolling_windows: [10, 30]`, `train_seasons`: 2015-16..2022-23 |
| **model_a** | `stat_dim: 21`, `epochs: 15`, no team_stats, no standing_rank |
| **model_b.xgb** | `n_estimators: 250`, `max_depth: 6`, `learning_rate: 0.07963763436288537`, `subsample: 0.8`, `colsample_bytree: 0.7` |
| **model_b.rf** | `n_estimators: 200`, `min_samples_leaf: 5` |
| **seasons** | 12 seasons (2014-15 through 2025-26) in config |

Best Spearman 0.777, playoff_spearman 0.802, rank_rmse 5.78.

---

## 2. Where configs come from

- **Pipeline (`run_pipeline_from_model_a`):** With `--config <file>`, it loads **`config/defaults.yaml`** then **deep-merges** your config on top. So anything you don’t set comes from **current** `defaults.yaml`.
- **Sweep (`sweep_hparams`):** Always loads **`config/defaults.yaml`** first, then deep-merges `--config` (if given). So every sweep batch is built from **current defaults** plus overlay.

So: **defaults.yaml is the global base.** If defaults change (seasons, loss, rolling, model_a/model_b), all overlay-based runs and sweeps change too.

---

## 3. Current `config/defaults.yaml` vs best run

**defaults.yaml is not the best-run config.** It is the old production/default setup (outputs4-style):

| Setting | defaults.yaml (current) | Best run (combo_0033) |
|---------|--------------------------|------------------------|
| paths.outputs | output/4_listmle | (sweep path) |
| training.loss_type | **listmle** | spearman_surrogate |
| training.listmle_target | **final_rank** | playoff_outcome |
| training.rolling_windows | **[15, 30]** | [10, 30] |
| model_a.stat_dim | **24** | 21 |
| model_a.epochs | **27** | 15 |
| model_b.xgb.n_estimators | **229** | 250 |
| model_b.xgb.max_depth | **5** | 6 |
| model_b.xgb.learning_rate | **0.072** | 0.0796… |

So any run that uses “defaults + overlay” without explicitly fixing these will **not** be comparable to the best run.

---

## 4. Outputs and whether they match best run

### 4.1 outputs8 (canonical best)

| What | Config source | Comparable to best? |
|------|----------------|---------------------|
| **output/8_spearman_surrogate/sweeps/20260217_042955/** (combo_0032, combo_0033, combo_0038, etc.) | Sweep from **2026-02-17** (batch 20260217_042955); base config at that time matched Spearman-surrogate, playoff_outcome, rolling [10,30], 12 seasons. Optuna chose combo_0033 for Spearman. | **Yes.** This *is* the best run. |

### 4.2 outputs8_spearman_surrogate (new batch)

| What | Config source | Comparable to best? |
|------|----------------|---------------------|
| **output/8_spearman_surrogate_sweep/sweeps/20260226_233831/** | Sweep run with **current** defaults + sweep overlay. Base = defaults (listmle, final_rank, rolling [15,30], stat_dim 24, epochs 27, etc.) plus **feature_subset_model_a** phase (Optuna over Model A stat columns). Configs have 50-year seasons, `player_stat_cols`, stat_dim 15, epochs 26, different XGB params. | **No.** Different experiment (Model A feature selection). Different base (current defaults), different loss/target/rolling if overlay didn’t override everything. |

So the **new** outputs8_spearman_surrogate batch is **not** a replication of the best run; it’s a different sweep (feature_subset_model_a) on top of current defaults.

### 4.3 outputs_team_stats_spearman_surrogate (run_001 / run_026)

| What | Config source | Comparable to best? |
|------|----------------|---------------------|
| **output/team_stats_spearman_surrogate** (single run) | Pipeline with `--config config/team_stats_spearman_surrogate.yaml` → **defaults + overlay**. Overlay sets: `paths.outputs`, `loss_type: spearman_surrogate`, `listmle_target: playoff_outcome`, `rolling_windows: [10, 30]`, `model_a.stat_dim: 27`, `model_a.use_team_stats: true`. Everything else comes from **defaults** (e.g. epochs **27**, XGB **229/5/0.072**). | **No.** Same loss/target/rolling, but: stat_dim **27** and **use_team_stats: true** (experiment), and **epochs 27**, **model_b** from defaults (not best-run 15 / 250/6/0.0796). |

So the team_stats run was **not** “best run + team stats”; it was “defaults + team_stats overlay,” which is not comparable.

### 4.4 outputs13_rmse_surrogate

| What | Config source | Comparable to best? |
|------|----------------|---------------------|
| **output/13_rmse_surrogate/sweeps/rmse_surrogate_40/** | Sweep with **rank_rmse_surrogate** loss. Optuna chose different model_a/model_b (e.g. combo_0004: stat_dim 22, epochs 22, XGB 258/4/0.08). | **No.** Intentionally different **loss** (RMSE surrogate) and different hyperparameters. |

### 4.5 outputs4, outputs7, etc.

- **outputs4:** Various phases; base was older defaults (listmle, final_rank or playoff_outcome, rolling [15,30] or similar). Not the Spearman-surrogate best run.
- **outputs7:** ListMLE sweep (same loss family as older runs), rolling [10,30]; not Spearman-surrogate. See [OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md](OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md).

So: **only output/8_spearman_surrogate/sweeps/20260217_042955 (e.g. combo_0033) is the official best-run config.**

---

## 5. How many runs have different configs?

- **Same as best run:** Only runs that explicitly use **`output/8_spearman_surrogate/sweeps/20260217_042955/combo_0033/config.yaml`** (or an exact copy of that config).
- **Different by design:** All of outputs13 (RMSE surrogate), outputs_team_stats (team-stats experiment), outputs8_spearman_surrogate 20260226 (feature_subset_model_a), and other sweeps that use a different loss/phase/base.
- **Different because base = defaults:** Any pipeline run that used only an overlay (e.g. team_stats_spearman_surrogate) got **defaults** for everything not in the overlay (epochs, model_b, etc.), so not comparable.

So: **most recent runs you have are not comparable to the best run** — either they use a different loss/experiment or they use current defaults as base instead of the best-run config.

---

## 6. Why you have to rerun (and how to run comparable to best)

- **Why rerun:** Runs like outputs_team_stats and new outputs8_spearman_surrogate batches were built from **defaults + overlay**. Defaults do not match the best run (different loss, target, rolling, epochs, model_b). So those runs are not comparable.
- **How to run comparable to best:**
  1. Use the **best-run config file** as the single source of truth for training/inference:
     ```bash
     python -m scripts.run_pipeline_from_model_a --config "outputs8/sweeps/20260217_042955/combo_0033/config.yaml" --outputs "output/8_spearman_surrogate_sweep/official_best_spearman"
     ```
     Do **not** pass an overlay that gets merged with defaults; pass the **full** combo_0033 config (and only override `paths.outputs` via `--outputs` if needed).
  2. For a **new experiment** (e.g. team stats) that you want comparable to best: start from the **best-run config**, not defaults. For example:
     - Copy `outputs8/sweeps/20260217_042955/combo_0033/config.yaml` to something like `config/best_run_plus_team_stats.yaml`.
     - Change only what the experiment needs: e.g. `paths.outputs`, `model_a.stat_dim`, `model_a.use_team_stats: true`, `model_a.team_stats_cols`.
     - Run with `--config config/best_run_plus_team_stats.yaml` and no merge with defaults (or use a script that loads combo_0033 and then applies only the small overlay).

That way, “comparable to best” means: same loss, target, rolling, epochs, and model_b as combo_0033, with only the intended change (e.g. team stats).

---

## 7. Summary table

| Output / run | Config base | Same as best run? | Notes |
|--------------|-------------|-------------------|--------|
| output/8_spearman_surrogate/sweeps/20260217_042955/combo_0033 | Original sweep base (Spearman-surrogate, [10,30], etc.) | Yes | Official best. |
| output/8_spearman_surrogate_sweep/sweeps/20260226_233831 | defaults.yaml + feature_subset_model_a | No | Different phase and base. |
| output/team_stats_spearman_surrogate (run_001/026) | defaults.yaml + team_stats overlay | No | Different stat_dim, team_stats, epochs, model_b. |
| output/13_rmse_surrogate | defaults + RMSE sweep | No | Different loss and hyperparams. |
| config/defaults.yaml | — | No | output/4_listmle-style (listmle, final_rank, [15,30], etc.). |

**Bottom line:** To avoid having to rerun for comparability, use **`output/8_spearman_surrogate/sweeps/20260217_042955/combo_0033/config.yaml`** (or a minimal overlay on top of that) for any run you want comparable to the best run.
