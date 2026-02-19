# outputs8 Sweep Analysis vs Previous Best

**Sweep:** outputs8/sweeps/20260217_042955  
**Trials:** 40 (Optuna, n_jobs=3, phase=baseline)  
**Objective:** spearman (primary)  
**Last updated:** 2026-02-17

---

## 1. Overview

**These outputs8 configs are the current official best per metric.** Full cross-run comparison and official config list are in [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md).

outputs8 baseline sweep improves strongly over previous best (outputs4 Phase 3 fine NDCG@16) on Spearman, playoff Spearman, and rank error metrics, with a trade-off on NDCG at low cutoffs.

| Metric | outputs8 Best | Previous Best (outputs4) | Δ |
|--------|---------------|--------------------------|---|
| **Spearman** | **0.777** (combo 33) | 0.557 | **+0.220** |
| **playoff_spearman** | **0.854** (combo 38) | 0.568 | **+0.286** |
| **rank_mae** | **4.80** (combo 33/38) | 6.33 | **-1.53** |
| **rank_rmse** | **5.78** (combo 33) | 8.15 | **-2.37** |
| NDCG@16 | 0.522 (combo 32) | 0.550 | -0.028 |
| NDCG@4 | 0.349 (combo 32) | 0.506 | -0.157 |

---

## 2. Best Configs by Objective (outputs8)

| Objective | Combo | Key Metric | Config Path |
|-----------|-------|------------|-------------|
| spearman | 33 | spearman 0.777, playoff_spearman 0.802 | combo_0033 |
| ndcg4 / ndcg16 / ndcg20 / ndcg30 | 32 | ndcg@30 0.522, spearman 0.737 | combo_0032 |
| playoff_spearman | 38 | playoff_spearman **0.854** | combo_0038 |
| rank_mae / rank_rmse | 33 | rank_mae 4.80, rank_rmse 5.78 | combo_0033 |
| spearman_standings | 1 | — | combo_0001 |
| ndcg4_standings | 9 | — | combo_0009 |
| ndcg16_standings | 33 | — | combo_0033 |
| ndcg30_standings | 20 | — | combo_0020 |
| rank_rmse_standings | 38 | — | combo_0038 |

---

## 3. Detailed Metrics by Best Combo

### Combo 33 — Best Spearman / rank_mae / rank_rmse
- **spearman:** 0.777  
- **playoff_spearman:** 0.802  
- **rank_mae:** 4.80 | **rank_rmse:** 5.78  
- ndcg@30: 0.438 | ndcg@4: 0.042  
- champion_in_top_4: 0.0 | champion_rank: 7  
- HPs: rolling [10,30], model_a_epochs 15, max_depth 6, lr 0.080, n_xgb 250, n_rf 200

### Combo 38 — Best playoff_spearman
- **playoff_spearman:** **0.854**  
- spearman: 0.766  
- rank_mae: 4.80 | rank_rmse: 5.92  
- ndcg@30: 0.440 | ndcg@4: 0.040  
- champion_in_top_4: 0.0 | champion_rank: 7  
- HPs: rolling [10,30], model_a_epochs 19, max_depth 6, lr 0.070, n_xgb 297, n_rf 200

### Combo 32 — Best NDCG (4/16/20/30)
- **ndcg@30:** 0.522 | ndcg@16: 0.522 | ndcg@4: 0.349  
- spearman: 0.737 | playoff_spearman: 0.751  
- rank_mae: 5.07 | rank_rmse: 6.28  
- champion_in_top_4: **1.0** | champion_rank: **3**  
- HPs: rolling [10,30], model_a_epochs 15, max_depth 5, lr 0.056, n_xgb 231, n_rf 200

---

## 4. Comparison to Previous Best (outputs4 Phase 3)

Previous best from [SWEEP_ANALYSIS_02-08.md](SWEEP_ANALYSIS_02-08.md): Phase 3 fine NDCG@16 combo 18.

| Metric | outputs8 combo 33 | outputs8 combo 38 | outputs8 combo 32 | outputs4 phase3 (combo 18) |
|--------|-------------------|-------------------|-------------------|----------------------------|
| Spearman | **0.777** | 0.766 | 0.737 | 0.557 |
| playoff_spearman | 0.802 | **0.854** | 0.751 | 0.568 |
| rank_mae | **4.80** | **4.80** | 5.07 | 6.33 |
| rank_rmse | **5.78** | 5.92 | 6.28 | 8.15 |
| NDCG@16 | 0.438 | 0.440 | **0.522** | 0.550 |
| NDCG@4 | 0.042 | 0.040 | 0.349 | 0.506 |

**Improvements:**
- Spearman: +0.22 (combo 33)  
- playoff_spearman: +0.29 (combo 38)  
- rank_mae: -1.53 | rank_rmse: -2.37  

**Regressions:**
- NDCG@4 and NDCG@16 are lower in outputs8 Spearman-focused combos (33, 38). Combo 32 restores NDCG@16 to ~0.52 but still below outputs4’s 0.55.

**Config differences:**  
outputs8 uses `listmle_target: playoff_outcome`, rolling [10,30], and a baseline pipeline. outputs4 Phase 3 used `listmle_target: final_rank` (standings), rolling [15,30]. Same test seasons (2023-24, 2024-25) per config.

---

## 4b. What’s different about outputs8 (vs outputs7)

The main pipeline difference is **training loss**:

| | outputs7 (sweep 20260217_235940) | outputs8 (sweep 20260217_042955) |
|---|----------------------------------|----------------------------------|
| **Config** | `config/outputs7_sweep_*.yaml` (no loss override) | `config/outputs8_sweep_spearman.yaml` |
| **`training.loss_type`** | **listmle** (default) | **spearman_surrogate** |
| **Meaning** | Model A trained with ListMLE (likelihood of correct order) | Model A trained with a differentiable Spearman surrogate; optimization targets the same metric we evaluate |

So in outputs8, the listwise stage is **optimized for Spearman** (and the sweep objective is Spearman), which matches the evaluation metric. In outputs7, the listwise stage used ListMLE, so the training objective did not directly target Spearman. That alignment (loss = evaluation metric) is the main reason outputs8 reaches higher Spearman and playoff_spearman. Other settings (rolling [10,30], `listmle_target: playoff_outcome`, test seasons, data) are the same; hyperparameter differences (e.g. max_depth, lr) are from the Optuna search on top of that.

---

## 5. Optuna Study (outputs8)

- **objective:** spearman  
- **best_value:** 0.777 (combo 33)  
- **best_params:** rolling_windows [10,30], model_a_epochs 15, max_depth 6, lr 0.080, n_estimators_xgb 250, n_estimators_rf 200, subsample 0.8, colsample_bytree 0.7, min_samples_leaf 4  

---

## 6. Recommendations

1. **Spearman / rank error:** Use **combo 33** — best Spearman (0.777), rank_mae, and rank_rmse.
2. **playoff_spearman:** Use **combo 38** — best playoff outcome correlation (0.854).
3. **NDCG / top-order:** Use **combo 32** — best NDCG and only config with champion in top 4.
4. **Production default:** Prefer **combo 33** or **combo 38** for overall playoff strength; use combo 32 if top-4 ordering is the main goal.

---

## 7. outputs7 Comparison

outputs7 (sweep 20260217_235940, 40 trials) was compared in [OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md](OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md). **outputs8 beats outputs7** on every primary metric: Spearman (0.777 vs 0.765), playoff_spearman (0.854 vs 0.742), rank_mae (4.80 vs 5.13), rank_rmse (5.78 vs 5.94), NDCG@30 (0.522 vs 0.487). Same pipeline family; gains are from better hyperparameters found in the outputs8 sweep.

---

## 8. Planned: outputs9

A future **outputs9** sweep will use the same mechanics as outputs7/8 but **ListMLE** (outputs6-style config) instead of Spearman surrogate. See [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md).

---

## 9. Related Files

- [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md) — official best configs and cross-run comparison  
- `outputs8/sweeps/20260217_042955/sweep_results_summary.json`  
- `outputs8/sweeps/20260217_042955/best_config_by_objective.json`  
- `outputs8/sweeps/20260217_042955/sweep_results.csv`  
- [OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md](OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md) — outputs4 / outputs7 / outputs8 three-way comparison  
- [SWEEP_ANALYSIS_02-08.md](SWEEP_ANALYSIS_02-08.md) — outputs4 Phase 2/3 analysis  
