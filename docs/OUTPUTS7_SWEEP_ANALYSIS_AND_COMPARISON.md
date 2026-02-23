# outputs7 Sweep Analysis & Comparison (outputs4 / outputs7 / outputs8)

**outputs7 Sweep:** outputs7/sweeps/20260217_235940  
**Trials:** 40 (Optuna, n_jobs=2, phase=baseline)  
**Last updated:** 2026-02-17

---

## 1. Overview

outputs7 sits between the previous best (outputs4 Phase 3) and outputs8: it clearly beats outputs4 on Spearman, playoff Spearman, and rank error, but **outputs8 beats outputs7** on every primary metric. Same pipeline family (listmle_target: playoff_outcome, rolling [10,30], same test seasons).

---

## 2. Three-Way Comparison: outputs4 vs outputs7 vs outputs8

| Metric | outputs4 (prev best) | outputs7 Best | outputs8 Best | Best source |
|--------|----------------------|---------------|---------------|-------------|
| **Spearman** | 0.557 | 0.765 (combo 17) | **0.777** (combo 33) | outputs8 |
| **playoff_spearman** | 0.568 | 0.742 (combo 30) | **0.854** (combo 38) | outputs8 |
| **rank_mae** | 6.33 | 5.13 (combo 1) | **4.80** (combo 33) | outputs8 |
| **rank_rmse** | 8.15 | 5.94 (combo 17) | **5.78** (combo 33) | outputs8 |
| NDCG@30 | — | 0.487 (combo 16) | **0.522** (combo 32) | outputs8 |
| NDCG@16 | 0.550 | 0.465 (combo 17) | 0.522 (combo 32) | outputs4 (raw NDCG@16) |

**Summary:**
- **outputs7 vs outputs4:** outputs7 wins on Spearman (+0.21), playoff_spearman (+0.17), rank_mae (-1.2), rank_rmse (-2.2).
- **outputs8 vs outputs7:** outputs8 wins on Spearman (+0.01), playoff_spearman (+0.11), rank_mae (-0.33), rank_rmse (-0.16), NDCG@30 (+0.04).

---

## 3. outputs7 Best Configs by Objective

| Objective | Combo | Key Metric | Notes |
|-----------|-------|------------|--------|
| spearman | 17 | spearman 0.765, playoff_spearman 0.736 | rank_rmse 5.94 |
| ndcg4 / ndcg20 / ndcg30 | 16 | ndcg@30 0.487, spearman 0.733 | champion_in_top_4: 1.0 |
| ndcg16 | 36 | — | — |
| playoff_spearman | 30 | playoff_spearman **0.742** | spearman 0.737 |
| rank_mae | 1 | rank_mae **5.13** | spearman 0.749 |
| rank_rmse | 17 | rank_rmse **5.94** | same as best Spearman combo |

---

## 4. outputs7 Best Combo Details

### Combo 17 — Best Spearman / rank_rmse
- **spearman:** 0.765 | **playoff_spearman:** 0.736  
- **rank_mae:** 5.27 | **rank_rmse:** 5.94  
- ndcg@30: 0.466 | ndcg@4: 0.062  
- champion_rank: 5 | champion_in_top_4: 0  
- HPs: rolling [10,30], model_a_epochs 14, max_depth 3, lr 0.066, n_xgb 245, n_rf 200  

### Combo 30 — Best playoff_spearman
- **playoff_spearman:** 0.742  
- spearman: 0.737 | rank_mae: 5.47 | rank_rmse: 6.28  
- ndcg@30: 0.430 | ndcg@4: 0.054  
- HPs: model_a_epochs 10, max_depth 5, lr 0.081, n_xgb 220  

### Combo 16 — Best NDCG
- **ndcg@30:** 0.487 | ndcg@4: 0.346  
- spearman: 0.733 | playoff_spearman: 0.708  
- rank_mae: 5.47 | rank_rmse: 6.32  
- champion_in_top_4: **1.0** | champion_rank: 4  
- HPs: model_a_epochs 15, max_depth 3, min_samples_leaf 6, n_xgb 246  

### Combo 1 — Best rank_mae
- **rank_mae:** 5.13 | rank_rmse: 6.14  
- spearman: 0.749 | playoff_spearman: 0.728  
- HPs: model_a_epochs 8, max_depth 6, n_xgb 273  

---

## 5. Why outputs8 Beats outputs7

Likely drivers (same pipeline, different sweep samples):

1. **Hyperparameter search:** outputs8 sweep (20260217_042955) found stronger combos (e.g. combo 33: max_depth 6, lr 0.080; combo 38: higher playoff_spearman with more epochs/n_xgb).
2. **Model A contribution:** outputs7 combo 17 has Model A with ndcg 0.597 and spearman 0.48 (ensemble still 0.765); outputs8’s best combos rely more on Model B (XGB) for the top ensemble numbers.
3. **Objective spread:** outputs8 has a combo (38) that specifically pushes playoff_spearman to 0.854; outputs7’s best playoff_spearman is 0.742 (combo 30).

Same data split and eval (2023–24, 2024–25 test), so gains are from config/optimization, not data.

---

## 6. Recommendations

1. **Official best is outputs8** — use outputs8 configs for production (best Spearman, playoff_spearman, rank_mae, rank_rmse). See [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md) and [OUTPUTS8_SWEEP_ANALYSIS_02-17.md](OUTPUTS8_SWEEP_ANALYSIS_02-17.md).
2. **outputs7** is a valid baseline to compare against (e.g. ablation or pipeline changes); prefer outputs8 when choosing a single best run.
3. **outputs4** remains the historical reference; outputs7 and outputs8 both improve on it.
4. **Planned: outputs9** — a future sweep will mirror the outputs7/8 setup with **ListMLE** instead of Spearman surrogate. See OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md.

---

## 7. Related Files

- [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md) — official best configs and cross-run comparison
- `outputs7/sweeps/20260217_235940/sweep_results_summary.json`
- `outputs7/sweeps/20260217_235940/best_config_by_objective.json`
- [OUTPUTS8_SWEEP_ANALYSIS_02-17.md](OUTPUTS8_SWEEP_ANALYSIS_02-17.md) — outputs8 vs previous best
- [SWEEP_ANALYSIS_02-08.md](SWEEP_ANALYSIS_02-08.md) — outputs4 Phase 2/3
