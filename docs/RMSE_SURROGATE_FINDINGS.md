# RMSE surrogate (13_rmse_surrogate) — consolidated findings

**Status:** **Not recommended for production.** Use **Spearman surrogate** (8_spearman_surrogate) instead. This doc records all findings so the approach can be retired or revisited later.

---

## 1. What was tried

- **13_rmse_surrogate:** Optuna sweep with **rank_rmse_surrogate** loss (Model A trained to minimize rank RMSE between predicted and playoff-outcome rank). Same pipeline and evaluation as 8_spearman_surrogate (Spearman surrogate), but training objective = rank RMSE.
- **Sweep:** 40 trials, batch `rmse_surrogate_40`, objective minimize rank_rmse. Best combo by rank_rmse = **combo_004** (22 Model A epochs).
- **Singular run:** Same combo_004 hyperparameters with **80 epochs** max and early_stopping_patience 10 (`config/outputs13_combo004_high_epochs.yaml`) to test whether more epochs helped.

---

## 2. Sweep results (rmse_surrogate_40)

| Metric | 13_rmse_surrogate best (combo_004, 22 epochs) | 8_spearman_surrogate best (Spearman surrogate) |
|--------|--------------------------------------|------------------------------------|
| Spearman | -0.16 | **0.777** |
| playoff_spearman | -0.30 | **0.854** |
| rank_mae | 11.0 | **4.80** |
| rank_rmse | 13.21 | **5.78** |
| NDCG@30 | 0.28 | **0.52** |

- No combo in the RMSE surrogate sweep achieved **positive** correlation with playoff outcome.
- Ensemble was **worse** than W/L standings (rank_mae 11 vs standings ~3.1).
- **Conclusion:** RMSE surrogate underperformed Spearman surrogate on every primary metric. See [OUTPUTS13_RMSE_SURROGATE_SWEEP.md](OUTPUTS13_RMSE_SURROGATE_SWEEP.md) and `output/13_rmse_surrogate/sweeps/rmse_surrogate_40/ANALYSIS_01.md`.

---

## 3. Singular run (80 epochs, combo_004)

| Season | Spearman | playoff_spearman | rank_mae | rank_rmse |
|--------|----------|------------------|----------|-----------|
| 2023-24 | -0.21 | -0.38 | 10.73 | 13.48 |
| 2024-25 | -0.04 | -0.14 | 9.80 | 12.49 |

- **vs 22-epoch combo_004:** Mixed — 2023-24 slightly worse, 2024-25 better (near-zero Spearman, better rank error). More epochs did **not** consistently improve and did not close the gap to Spearman surrogate.
- **vs 8_spearman_surrogate:** Still a large gap (8_spearman_surrogate Spearman 0.78, rank_mae 4.8).
- **vs W/L standings:** Ensemble still worse than standings on both seasons.
- See `output/13_rmse_surrogate/combo_004_high_epochs/ANALYSIS_01_singular_run.md`.

---

## 4. Why RMSE surrogate underperformed (hypotheses)

- **Objective vs evaluation:** Minimizing rank RMSE in training did not yield good Spearman or rank_mae on the test set; the surrogate may optimize a different shape of the ranking surface than what correlates with playoff outcome.
- **Epochs:** 22-epoch sweep best was weak; 80-epoch single run improved one season but not the other. Training loss was not fully logged in all runs, so we could not confirm whether 22 epochs was underfitting.
- **Comparison:** Spearman surrogate (8_spearman_surrogate) directly targets correlation and consistently outperformed RMSE surrogate on correlation and rank error. For this pipeline and data, **Spearman surrogate is the right choice.**

---

## 5. Recommendation

- **Production / best config:** Use **8_spearman_surrogate** (see [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md)). Do **not** use 13_rmse_surrogate (RMSE surrogate) for production.
- **Branch cleanup:** The **feature/train-rank-rmse-surrogate** branch (and any 13_rmse_surrogate-only branch) can be considered **unproductive** and is a candidate for deletion after this doc is in main. All findings are preserved here and in OUTPUTS13_RMSE_SURROGATE_SWEEP.md plus the ANALYSIS_01 files under `output/13_rmse_surrogate/`.

---

## 6. References

- [OUTPUTS13_RMSE_SURROGATE_SWEEP.md](OUTPUTS13_RMSE_SURROGATE_SWEEP.md) — setup, commands, sweep and singular-run summary  
- `output/13_rmse_surrogate/sweeps/rmse_surrogate_40/ANALYSIS_01.md` — sweep analysis  
- `output/13_rmse_surrogate/combo_004_high_epochs/ANALYSIS_01_singular_run.md` — 80-epoch run analysis  
- [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md) — official best (8_spearman_surrogate)
