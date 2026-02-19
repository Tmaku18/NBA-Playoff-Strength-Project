# Best Metrics & Config Comparison

Best configs per metric and side-by-side comparison of all "best runs," including **listmle_target** (outcome vs standings).

---

## Current official best (outputs8)

**Official best configs** for playoff-outcome metrics are from the **outputs8** sweep (batch 20260217_042955, Spearman-surrogate loss, 40 trials). For the full cross-run comparison (outputs4, outputs6 phase_1, outputs7, outputs8) and exact config paths, see **[OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md)**.

**Planned: outputs9 sweep** — outputs9 will use the same sweep setup as outputs7/8 but with **ListMLE** (outputs6-style config) instead of Spearman surrogate. See OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md.

---

## Listmle target (outcome vs standings)

| `listmle_target`   | Meaning |
|--------------------|--------|
| **`playoff_outcome`** | Model A ListMLE is trained to rank by **playoff finish** (1 = champion, 2 = runner-up, …). Eval: pred vs playoff outcome rank. |
| **`final_rank`**      | Model A ListMLE is trained to rank by **W/L standings** (regular-season). Eval: pred vs standings-derived rank. |

- **WSL sweep** (`outputs4/sweeps/wsl_playoff_spearman`): all combos use `listmle_target: playoff_outcome`.
- **Phase 6** (`outputs4/sweeps/phase6_ndcg16_playoff_narrow`): `listmle_target: playoff_outcome`; eval uses playoff_final_results.
- **Default** (`config/defaults.yaml`): `listmle_target: final_rank`. For playoff-outcome training use a combo config or `config/defaults_playoff_outcome.yaml`.

---

## Best config per metric

### Official best (outputs8)

| Metric | Config path | Key value |
|--------|-------------|-----------|
| **Spearman** | `outputs8/sweeps/20260217_042955/combo_0033/config.yaml` | 0.777 |
| **playoff_spearman** | `outputs8/sweeps/20260217_042955/combo_0038/config.yaml` | 0.854 |
| **rank_mae / rank_rmse** | `outputs8/sweeps/20260217_042955/combo_0033/config.yaml` | 4.80 / 5.78 |
| **NDCG@4, NDCG@16, NDCG@20, NDCG@30** | `outputs8/sweeps/20260217_042955/combo_0032/config.yaml` | NDCG@30 0.522 |

See [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md) for the full list and cross-run comparison.

### Historical best (pre-outputs8)

| Metric | Sweep | Combo | Listmle target | Config path | Ensemble (test) |
|--------|-------|-------|----------------|-------------|------------------|
| **Spearman** | WSL playoff_spearman | 18 | playoff_outcome | `outputs4/sweeps/wsl_playoff_spearman/combo_0018/config.yaml` | Spearman 0.532, NDCG 0.611 |
| **NDCG / NDCG@30** | WSL playoff_spearman | 10 | playoff_outcome | `outputs4/sweeps/wsl_playoff_spearman/combo_0010/config.yaml` | NDCG 0.824, Spearman 0.466 |
| **Playoff Spearman** | WSL playoff_spearman | 14 | playoff_outcome | `outputs4/sweeps/wsl_playoff_spearman/combo_0014/config.yaml` | Playoff ρ 0.523, NDCG 0.608 |
| **Rank RMSE** (vs playoff outcome) | WSL playoff_spearman | 18 | playoff_outcome | same as Spearman (combo_0018) | RMSE 8.37 |
| **Rank MAE** (vs playoff outcome) | WSL playoff_spearman | 14 | playoff_outcome | same as Playoff Spearman (combo_0014) | MAE 6.53 |
| **Spearman** (Phase 6) | Phase 6 NDCG16 | 5 | playoff_outcome | `outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0005/config.yaml` | Spearman 0.536, NDCG 0.486 |
| **NDCG** (Phase 6) | Phase 6 NDCG16 | 10 | playoff_outcome | `outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0010/config.yaml` | NDCG 0.531, Spearman 0.503 |
| **NDCG@4** (Phase 6) | Phase 6 NDCG16 | 6 | playoff_outcome | `outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0006/config.yaml` | NDCG@4 0.506, NDCG@16 0.542 |

*Standings eval = same training (playoff_outcome), but reported metric is correlation/RMSE vs W/L standings rank.*

**Model C (RF):** The pipeline does not train Model C by default (`training.train_model_c: false`). Ensemble = A + B only. When present (e.g. older runs), Model C metrics are kept in reports for analytic comparison only.

---

## Run pipeline with a best config

From project root (override output dir so sweep combo dir is not overwritten):

```powershell
# Best Spearman (WSL)
python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0018/config.yaml" --outputs "outputs4/wsl_best_spearman"

# Best NDCG (WSL)
python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0010/config.yaml" --outputs "outputs4/wsl_best_ndcg"

# Best Rank RMSE / Playoff Spearman (WSL) — combo_0018 and combo_0014
python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0018/config.yaml" --outputs "outputs4/wsl_best_rmse"
python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0014/config.yaml" --outputs "outputs4/wsl_best_playoff_spearman"
```

---

## outputs6 layout and run commands (phase_1, run_028+)

Future runs write under **outputs6**. Run IDs start at **run_028** when the pipeline is invoked with an output path that contains `outputs6`.

**Folder layout:**

```
outputs6/
  phase_<N>/                    # N = 1, 2, 3...
    outcome/                    # listmle_target = playoff_outcome
      best_spearman/
      best_ndcg4/
      best_ndcg12/
      best_ndcg16/
      best_ndcg30/
      best_rmse/
    standings/                  # listmle_target = final_rank (W/L standings)
      best_spearman/
      best_ndcg4/
      best_ndcg12/
      best_ndcg16/
      best_ndcg30/
      best_rmse/
```

When `--outputs` contains:

- `"outputs6"`: pipeline sets `inference.run_id` = `run_028` and `run_id_base` = 28
- `"outputs7"`: pipeline sets `inference.run_id` = `run_029` and `run_id_base` = 29
- `"outputs8"`: pipeline sets `inference.run_id` = `run_030` and `run_id_base` = 30
- `"standings"`: pipeline sets `training.listmle_target` = `final_rank`

### outputs7 / outputs8: sweep roots (new)

Both **outputs7** and **outputs8** follow the same sweep method as outputs4/outputs6: `scripts/sweep_hparams.py` deep-merges a config overlay on top of `config/defaults.yaml`, writes sweeps under `<outputsX>/sweeps/<batch_id>/combo_*/outputs/`, and you can then run the full pipeline into `outputs7/phase_<N>/...` or `outputs8/phase_<N>/...` using the winning combo’s `config.yaml`.

- **outputs7** (RMSE-optimized): run sweeps with `--objective rank_rmse` (minimize) using `config/outputs7_sweep_rmse.yaml`.
- **outputs8** (Spearman-optimized): run sweeps with `--objective spearman` (or `playoff_spearman`) using `config/outputs8_sweep_spearman.yaml`.

Example WSL sweep commands (Optuna; 4 parallel jobs):

```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project" ; export PYTHONPATH="$PWD" ; python -m scripts.sweep_hparams --config config/outputs7_sweep_rmse.yaml --method optuna --objective rank_rmse --n-trials 20 --listmle-target playoff_outcome --n-jobs 4
```

```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project" ; export PYTHONPATH="$PWD" ; python -m scripts.sweep_hparams --config config/outputs8_sweep_spearman.yaml --method optuna --objective spearman --n-trials 20 --listmle-target playoff_outcome --n-jobs 4
```

**Training toward Spearman or rank RMSE (different branches)**  
Model A can train with a **differentiable surrogate** for Spearman or rank RMSE instead of ListMLE. Use `training.loss_type`: `listmle` (default), `spearman_surrogate`, or `rank_rmse_surrogate`. Two branches and config overlays:

- **Branch `feature/train-spearman-surrogate`**: train Model A with Spearman-surrogate loss. Use config `config/outputs8_train_spearman_surrogate.yaml` (sets `loss_type: spearman_surrogate`, `paths.outputs: outputs8`). Merge with defaults or defaults_playoff_outcome when running the pipeline.
- **Branch `feature/train-rank-rmse-surrogate`**: train Model A with rank-RMSE-surrogate loss. Use config `config/outputs7_train_rank_rmse_surrogate.yaml` (sets `loss_type: rank_rmse_surrogate`, `paths.outputs: outputs7`).

Both branches contain the same code (both losses); the branch name and overlay indicate which objective to use. Optional: `training.loss_tau` (default 1.0) controls the soft-rank temperature for surrogate losses.

**Single-line WSL commands (phase_1; 3 jobs in parallel, 5 threads per job):**

Outcome – batch 1 (best_spearman, best_ndcg4, best_ndcg12):

```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project" ; export PYTHONPATH="$PWD" OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 ; ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0018/config.yaml" --outputs "outputs6/phase_1/outcome/best_spearman" ) & ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0006/config.yaml" --outputs "outputs6/phase_1/outcome/best_ndcg4" ) & ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0006/config.yaml" --outputs "outputs6/phase_1/outcome/best_ndcg12" ) ; wait
```

Outcome – batch 2 (best_ndcg16, best_ndcg30, best_rmse):

```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project" ; export PYTHONPATH="$PWD" OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 ; ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0016/config.yaml" --outputs "outputs6/phase_1/outcome/best_ndcg16" ) & ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0010/config.yaml" --outputs "outputs6/phase_1/outcome/best_ndcg30" ) & ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0018/config.yaml" --outputs "outputs6/phase_1/outcome/best_rmse" ) ; wait
```

Standings – batch 1 (best_spearman, best_ndcg4, best_ndcg12; listmle_target=final_rank via path):

```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project" ; export PYTHONPATH="$PWD" OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 ; ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0018/config.yaml" --outputs "outputs6/phase_1/standings/best_spearman" ) & ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0006/config.yaml" --outputs "outputs6/phase_1/standings/best_ndcg4" ) & ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0006/config.yaml" --outputs "outputs6/phase_1/standings/best_ndcg12" ) ; wait
```

Standings – batch 2:

```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project" ; export PYTHONPATH="$PWD" OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 ; ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0016/config.yaml" --outputs "outputs6/phase_1/standings/best_ndcg16" ) & ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0010/config.yaml" --outputs "outputs6/phase_1/standings/best_ndcg30" ) & ( python -m scripts.run_pipeline_from_model_a --config "outputs4/sweeps/wsl_playoff_spearman/combo_0018/config.yaml" --outputs "outputs6/phase_1/standings/best_rmse" ) ; wait
```

For **phase_2**, replace `phase_1` with `phase_2` in the `--outputs` paths.

### outputs6 phase_1 outcome results (run_028)

All six outcome runs completed (WSL). **Important:** `eval_report.json` reports multiple “Spearman” variants:

- **Overall Spearman** = `test_metrics_ensemble.spearman` (computed using the report’s relevance definition; tends to be the largest number).
- **Playoff Spearman** = `test_metrics_ensemble.playoff_metrics.spearman_pred_vs_playoff_outcome_rank` (predicted global order vs playoff outcome rank).
- **Standings Spearman** = `test_metrics_ensemble.spearman_standings` (predicted order vs regular-season W/L standings) — expected to be **negative** when we train/eval for playoff outcome.

The notes below focus on the **“best runs we saw in-chat”** where **Overall Spearman \(\gtrsim 0.7\)**, all from `outputs6/phase_1/outcome/.../eval_report.json`.

| outputs6 folder | Config used | Overall Spearman | Playoff Spearman | Ensemble NDCG@30 | Ensemble rank RMSE (vs outcome) |
|---|---|---:|---:|---:|---:|
| `best_rmse` | `outputs4/sweeps/wsl_playoff_spearman/combo_0018/config.yaml` | **0.749499** | **0.725473** | 0.413982 | 6.126445 |
| `best_ndcg16` | `outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0016/config.yaml` | **0.738376** | **0.715239** | 0.450277 | 6.260991 |
| `best_ndcg30` | `outputs4/sweeps/wsl_playoff_spearman/combo_0010/config.yaml` | **0.721468** | **0.717019** | 0.425699 | 6.460134 |
| `best_ndcg12` | `outputs4/sweeps/phase6_ndcg16_playoff_narrow/combo_0006/config.yaml` | **0.713014** | **0.685428** | 0.400204 | 6.557438 |

**Config notes (exact settings that produced each run)**

Common to all four runs:

- **training.listmle_target**: `playoff_outcome` (outcome ranking)
- **training.rolling_windows**: `[15, 30]`
- **training.n_folds**: `5`, **training.train_frac**: `0.75`
- **training.use_prior_season_baseline**: `true` (lookback `365` days)
- **elo.enabled / massey.enabled / raptor.enabled / motivation.enabled**: `true`
- **injury.enabled / team_rolling.enabled / sos_srs.enabled**: `false`

Per-run deltas (what differs across the configs):

- **`best_rmse`** (`combo_0018`, WSL sweep)
  - **Model A**: `epochs: 27`; attention **multi-temp enabled** with `temperature: 3`, `temperatures: [1, 5, 10]`, `multi_temp_base_weights: {1: 0.85, 5: 1.0, 10: 0.7}`; `use_amp: true`; confidence weights `entropy_weight: 0.5`, `max_weight_weight: 0.5`
  - **Model B (XGB)**: `n_estimators: 220`, `max_depth: 5`, `learning_rate: 0.085397...`, `subsample: 0.8`, `colsample_bytree: 0.7`, `early_stopping_rounds: 20`
  - **Stacking**: `use_confidence: true`

- **`best_ndcg16`** (`combo_0016`, Phase 6 narrow)
  - **Model A**: `epochs: 28` (no explicit attention override in this combo config)
  - **Model B (XGB)**: `n_estimators: 230`, `max_depth: 5`, `learning_rate: 0.083699...`, `subsample: 0.8`, `colsample_bytree: 0.7`, `early_stopping_rounds: 20`

- **`best_ndcg30`** (`combo_0010`, WSL sweep)
  - **Model A**: `epochs: 24`; same **multi-temp attention** block as `combo_0018`
  - **Model B (XGB)**: `n_estimators: 231`, `max_depth: 5`, `learning_rate: 0.087824...`, `subsample: 0.8`, `colsample_bytree: 0.7`, `early_stopping_rounds: 20`
  - **Stacking**: `use_confidence: true`

- **`best_ndcg12`** (`combo_0006`, Phase 6 narrow)
  - **Model A**: `epochs: 24` (no explicit attention override in this combo config)
  - **Model B (XGB)**: `n_estimators: 233`, `max_depth: 5`, `learning_rate: 0.080557...`, `subsample: 0.8`, `colsample_bytree: 0.7`, `early_stopping_rounds: 20`

**Quick interpretation**

- These are **high correlations** (Spearman ~0.71–0.75) for the “overall” Spearman metric used by the report, and **Playoff Spearman** is also strong (~0.69–0.73).
- But **rank RMSE vs outcome (~6.1–6.6)** is still worse than the **W/L standings baseline RMSE (~4.45)** in these same reports (`rank_rmse_wl_record_standings_vs_playoff_outcome_rank`), so these runs are “good at ordering” but not necessarily “good at exact rank error” relative to the baseline.

### Model vs standings significance (p-value) — outputs6 phase_1

Evaluation reports include a **paired bootstrap** over teams: resample teams, compute mean(standings MAE − model MAE); **p_value_model_better_than_standings** = proportion of bootstrap samples where that mean ≤ 0. So **high p-value** ⇒ we cannot reject the null that the model is no better than W/L standings.

| Phase 1 metric folder | p-value range (model better than standings) |
|-----------------------|--------------------------------------------|
| **best_spearman**     | 0.9975 – **1.0**                           |
| **best_ndcg4**        | 0.9965 – **1.0**                           |
| **best_ndcg12**       | 0.993 – **1.0**                            |
| **best_ndcg16**       | 0.992 – **1.0**                            |
| **best_rmse**         | 0.9905 – **1.0**                           |
| **best_ndcg30**       | 0.9845 – **0.9995** (lowest in phase_1)    |

**Interpretation:** Across all outputs6 phase_1 outcome runs, p-values are **~0.98–1.0**. We do **not** have statistically significant evidence that any model (A, B, or ensemble) is better than regular-season W/L standings at predicting playoff outcome rank; the test says the observed improvement could easily be chance. Source: `outputs6/phase_1/outcome/<metric>/eval_report.json` (and run-level `run_028_02-15/eval_report.json`), key `significance.p_value_model_better_than_standings` per model.

### East vs West (run_028, best_spearman)

Per-conference metrics live in **`test_metrics_by_conference`** in `eval_report.json` (keys `"E"`, `"W"`). Example from `outputs6/phase_1/outcome/best_spearman/run_028/eval_report.json`:

| Conference | NDCG | Spearman | Ensemble MAE vs outcome | Model A MAE | Model B MAE |
|------------|------|----------|--------------------------|-------------|-------------|
| East (E)   | 0.546 | 0.54    | 6.93                     | 8.73        | 7.47        |
| West (W)   | 0.636 | 0.81    | 4.67                     | 5.13        | 5.73        |

Interpretation: West has higher NDCG and Spearman and lower rank error in this run; ensemble and Model A both do better in the West. For full East/West semantics and caveats, see [MODEL_A_VS_B_02-15.md](MODEL_A_VS_B_02-15.md) §7 and [ANALYSIS_OF_ATTENTION_WEIGHTS_02-03.md](ANALYSIS_OF_ATTENTION_WEIGHTS_02-03.md) §5.

### Analysis: outputs6 phase_2 outcome (best_ndcg4 run_028)

Run path: **`outputs6/phase_2/outcome/best_ndcg4/run_028`**. EOS source: **eos_final_rank** (playoff outcome rank). Ensemble = A + B only; Model C present for comparison.

**Test metrics (ensemble)**

| Metric | Value |
|--------|-------|
| **Spearman** | **0.740** |
| **Playoff Spearman** (pred vs playoff outcome rank) | **0.702** |
| **Kendall τ** | 0.549 |
| **NDCG@30** | 0.394 |
| **NDCG@4** (Conference Finals) | 0.050 |
| **Rank MAE** (pred vs outcome) | 5.40 |
| **Rank RMSE** (pred vs outcome) | 6.24 |
| **Standings baseline** (W/L vs outcome) | MAE 3.13, RMSE 4.45 |
| **precision@4** / **precision@8** | 0 / 0.625 |
| **champion_rank** / **champion_in_top_4** | 9 / 0 |
| **Brier (champ odds)** / **ECE (champ odds)** | 0.032 / 0.0 |

**Model vs standings (same outcome ranks)**  
Ensemble and both A and B have **worse** MAE/RMSE than W/L standings (Δ MAE ≈ −2.27, Δ RMSE ≈ −1.79). Bootstrap p-values (model better than standings): ensemble 0.998, A 1.0, B 0.9965, C 1.0 — no evidence that any model beats the standings.

**East vs West (phase_2 best_ndcg4)**  
West has stronger correlation and lower error: East NDCG 0.596, Spearman 0.68, Kendall 0.52, ensemble MAE 6.2; West NDCG 0.569, Spearman **0.81**, Kendall **0.62**, ensemble MAE **4.6**.

**Takeaways**  
- Strong overall and playoff Spearman (~0.74 / 0.70) and good precision@8 (0.625); NDCG@4 is low (0.05), champion not in top 4.  
- Ranking quality (order) is good; rank error is still worse than standings baseline.  
- Use `run_028/confusion_matrix_ranking_top16.png` and `eval_report.json` → `confusion_matrices_ranking_top16` for top-16 ordering detail; full narrative in `run_028/ANALYSIS_03.md`.

---

## Side-by-side: config (best runs)

Same structure across WSL best-by-metric runs. Phase 6 differs mainly in Model B (XGB) params and some Model A epochs.

| Config item | WSL best Spearman (18) | WSL best NDCG (10) | WSL best Playoff ρ (14) | WSL best RMSE standings (6) | Phase 6 best Spearman (5) | Phase 6 best NDCG (10) |
|-------------|------------------------|--------------------|--------------------------|-----------------------------|----------------------------|-------------------------|
| **listmle_target** | playoff_outcome | playoff_outcome | playoff_outcome | playoff_outcome | playoff_outcome | playoff_outcome |
| **rolling_windows** | [15, 30] | [15, 30] | [15, 30] | [15, 30] | [15, 30] | [15, 30] |
| **model_a.epochs** | 27 | 24 | 27 | 27 | 26 | 25 |
| **model_a.attention_heads** | 4 | 4 | 4 | 4 | 4 | 4 |
| **model_a.attention.temperature** | 3 | 3 | 3 | 3 | (default) | (default) |
| **model_a.attention.temperatures** | [1, 5, 10] | [1, 5, 10] | [1, 5, 10] | [1, 5, 10] | — | — |
| **model_a.attention.multi_temp_base_weights** | {1:0.85, 5:1, 10:0.7} | same | same | same | — | — |
| **model_b.xgb.n_estimators** | 220 | 231 | 220 | 228 | 245 | 241 |
| **model_b.xgb.max_depth** | 5 | 5 | 5 | 5 | 5 | 5 |
| **model_b.xgb.learning_rate** | 0.0854 | 0.0878 | 0.0861 | 0.0891 | 0.0802 | 0.0821 |
| **model_b.xgb.subsample** | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| **model_b.xgb.colsample_bytree** | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |

*Model B = XGB only (single model). Phase 6 combo configs may omit explicit attention block; they inherit from sweep/defaults.*

---

## Side-by-side: test metrics (ensemble)

All metrics below are **test** (2023-24, 2024-25). WSL uses playoff_outcome_rank; Phase 6 uses playoff_final_results (equivalent notion).

| Metric | WSL Spearman (18) | WSL NDCG (10) | WSL Playoff ρ (14) | WSL RMSE stand (6) | Phase 6 Spearman (5) | Phase 6 NDCG (10) |
|--------|-------------------|---------------|--------------------|--------------------|----------------------|-------------------|
| **Ensemble Spearman** | 0.532 | 0.466 | 0.527 | 0.515 | 0.536 | 0.503 |
| **Ensemble NDCG** | 0.611 | **0.824** | 0.608 | 0.603 | 0.486 | 0.531 |
| **Ensemble NDCG@4** | 0.464 | **0.696** | **0.500** | 0.465 | 0.464 | 0.464 |
| **Ensemble NDCG@16** | 0.541 | **0.753** | 0.538 | 0.532 | 0.531 | 0.533 |
| **Ensemble Rank RMSE** (vs outcome) | **8.37** | 8.95 | 8.42 | 8.52 | 8.34 | 8.63 |
| **Ensemble Rank MAE** (vs outcome) | 6.60 | 6.93 | **6.53** | 6.60 | **6.53** | 6.60 |
| **Playoff Spearman** (pred vs outcome) | 0.520 | 0.468 | **0.523** | 0.518 | **0.540** | 0.527 |
| **Model B (XGB) Spearman** | 0.624 | **0.684** | **0.660** | 0.588 | **0.632** | 0.617 |
| **Model B (XGB) NDCG** | 0.409 | **0.416** | **0.431** | 0.384 | 0.197 | 0.214 |

---

## Comparison: standings vs outcome (same run, two evals)

All WSL configs are **trained** with `listmle_target: playoff_outcome`. The tables below show the **same** ensemble predictions evaluated two ways:

- **Eval vs outcome**: predicted rank vs **playoff outcome rank** (1 = champion, 2 = runner-up, …).
- **Eval vs standings**: predicted rank vs **W/L standings rank** (regular-season order).

So “outcome” = how well we predict playoff finish; “standings” = how well we match regular-season order.

### Quick comparison (ensemble, same run — two evals)

| Run | Spearman (vs outcome) | Spearman (vs standings) | Rank RMSE (vs outcome) | Rank RMSE (vs standings) | NDCG@30 (vs outcome) | NDCG@30 (vs standings) |
|-----|------------------------|-------------------------|-------------------------|---------------------------|------------------------|------------------------|
| WSL best Spearman (18) | **0.532** | −0.508 | **8.37** | 8.59 | 0.611 | 0.591 |
| WSL best NDCG (10) | 0.466 | −0.469 | 8.95 | 8.92 | **0.824** | **0.824** |
| WSL best Playoff ρ (14) | **0.527** | −0.515 | **8.42** | 8.52 | 0.608 | 0.596 |

### WSL best Spearman (combo 18) — outcome vs standings eval

| Metric | Eval vs outcome | Eval vs standings |
|--------|------------------|--------------------|
| **Spearman** | **0.532** | −0.508 |
| **Rank RMSE** | **8.37** | 8.59 |
| **Rank MAE** | 6.60 | — |
| **NDCG** | 0.611 | — |
| **NDCG@4** | 0.464 | 0.443 |
| **NDCG@16** | 0.541 | 0.496 |
| **NDCG@30** | 0.611 | 0.591 |
| **Playoff Spearman** | 0.520 | — |

### WSL best NDCG (combo 10) — outcome vs standings eval

| Metric | Eval vs outcome | Eval vs standings |
|--------|------------------|--------------------|
| **Spearman** | 0.466 | −0.469 |
| **Rank RMSE** | 8.95 | 8.92 |
| **Rank MAE** | 6.93 | — |
| **NDCG** | **0.824** | **0.824** |
| **NDCG@4** | **0.696** | **0.682** |
| **NDCG@16** | **0.753** | **0.730** |
| **NDCG@30** | **0.824** | **0.824** |
| **Playoff Spearman** | 0.468 | — |

### WSL best Playoff Spearman (combo 14) — outcome vs standings eval

| Metric | Eval vs outcome | Eval vs standings |
|--------|------------------|--------------------|
| **Spearman** | **0.527** | −0.515 |
| **Rank RMSE** | **8.42** | 8.52 |
| **Rank MAE** | **6.53** | — |
| **NDCG** | 0.608 | — |
| **NDCG@4** | **0.500** | 0.444 |
| **NDCG@16** | 0.538 | 0.501 |
| **NDCG@30** | 0.608 | 0.596 |
| **Playoff Spearman** | **0.523** | — |

### Takeaway

- **Trained for outcome**: Ensemble Spearman **vs standings** is negative (−0.47 to −0.52): predictions tuned for playoff finish do not follow regular-season order.
- **Trained for outcome**: Rank RMSE is similar whether we eval vs outcome or vs standings (~8.4–8.9).
- **Combo 10**: NDCG is high under **both** evals (0.82); this config ranks well on both playoff finish and standings.
- Use **eval vs outcome** to measure playoff-prediction quality; use **eval vs standings** to see alignment with W/L order.

---

## Alternate listmle: outcome vs standings (training target)

| Listmle target | Use case | Best WSL combo (if evaluated same way) | Note |
|----------------|----------|----------------------------------------|------|
| **playoff_outcome** | Rank by playoff finish; eval vs playoff outcome rank | Spearman/NDCG/RMSE: 18, 10, 14 | All WSL best configs above use this. |
| **final_rank** | Rank by W/L standings | Not in WSL sweep (WSL is all playoff_outcome) | Use `config/defaults.yaml` or `config/outputs5_regular_*.yaml` for standings-based training. |

To run with **standings** (final_rank) as the training target instead of playoff outcome, use a config that sets `listmle_target: final_rank` (e.g. `config/defaults.yaml` or `config/outputs5_regular_spearman.yaml`) and optionally merge in Model A/B params from a best combo.

---

## Analysis: local runs (wsl_best_spearman, wsl_best_ndcg, wsl_best_rmse)

Comparison of the three pipelines run locally with WSL best configs (combo 18 → Spearman & RMSE, combo 10 → NDCG). Test metrics from `outputs4/wsl_best_*/eval_report.json`.

### Side-by-side test metrics (local runs)

| Metric | wsl_best_spearman (combo 18) | wsl_best_ndcg (combo 10) | wsl_best_rmse (combo 18) |
|--------|-----------------------------|--------------------------|---------------------------|
| **Ensemble Spearman** | **−0.72** | **−0.73** | **−0.72** |
| **Ensemble NDCG** | 0.30 | 0.29 | 0.29 |
| **Ensemble NDCG@4** | ~0 | ~0 | ~0 |
| **Ensemble Rank RMSE** | 16.05 | 16.09 | 16.05 |
| **Ensemble Rank MAE** | 14.2 | 14.2 | 14.1 |
| **Model A Spearman** | **0.46** | **0.47** | **0.46** |
| **Model A NDCG** | **0.60** | **0.61** | **0.60** |
| **Model A NDCG@4** | 0.46 | **0.50** | 0.46 |
| **Model A Rank RMSE** | 9.03 | 8.93 | 8.97 |
| **Model B (XGB) Spearman** | **0.62** | **0.68** | **0.62** |
| **Model B (XGB) NDCG** | 0.41 | **0.42** | 0.41 |
| **Model B Rank RMSE** | **7.51** | **6.88** | **7.51** |
| **Model C (RF) Spearman** | 0.31 | 0.31 | 0.31 |

### Findings

1. **Ensemble is broken on these runs.** Ensemble Spearman is strongly negative (−0.72 to −0.73) and ensemble NDCG is low (~0.29–0.30), while **Model A** and **Model B** both have positive Spearman (0.46–0.68) and reasonable NDCG (0.41–0.61). So the stacking/meta-learner (Ridge or blend) is combining A and B in a way that inverts or cancels the good signal.
2. **Model A and Model B are healthy.** Model A: Spearman 0.46–0.47, NDCG 0.60–0.61. Model B (XGB): Spearman 0.62–0.68, NDCG 0.41–0.42, Rank RMSE 6.88–7.51. These are in line with the WSL sweep expectations; the issue is the ensemble, not the base models.
3. **Best NDCG config (combo 10)** gives the strongest Model B (Spearman 0.68, Rank RMSE 6.88) and slightly better Model A NDCG@4 (0.50). Best Spearman/RMSE config (combo 18) gives the same Model B twice (Spearman 0.62, RMSE 7.51) in both wsl_best_spearman and wsl_best_rmse, as expected (same config).
4. **Recommendation:** Until stacking is fixed (e.g. check Ridge weights, confidence features, or OOF alignment), use **Model A alone** or **Model B alone** for predictions, or a simple fixed blend (e.g. 0.5×A + 0.5×B) instead of the learned meta-learner. Re-running the pipeline in WSL with the same configs may reproduce the original sweep ensemble metrics (0.52–0.53 Spearman); the local Windows run may differ due to split/seed or stacking fit.

### Comparison to WSL sweep (expected vs local)

| Run | Expected (WSL sweep) | Local (Windows) |
|-----|----------------------|------------------|
| Combo 18 ensemble Spearman | 0.532 | −0.72 |
| Combo 18 Model A Spearman | 0.532 | 0.46 |
| Combo 10 ensemble NDCG | 0.824 | 0.29 |
| Combo 10 Model A NDCG | 0.824 | 0.61 |

The large gap between expected and local **ensemble** metrics, with Model A/B staying reasonable, points to stacking or inference-time blending, not to Model A/B training.

---

## Source files

- WSL sweep summary: `outputs4/sweeps/wsl_playoff_spearman/sweep_results_summary.json`
- Phase 6 sweep summary: `outputs4/sweeps/phase6_ndcg16_playoff_narrow/sweep_results_summary.json`
- Combo configs: `outputs4/sweeps/<batch_id>/combo_XXXX/config.yaml`
- Local run evals: `outputs4/wsl_best_spearman/eval_report.json`, `outputs4/wsl_best_ndcg/eval_report.json`, `outputs4/wsl_best_rmse/eval_report.json`
