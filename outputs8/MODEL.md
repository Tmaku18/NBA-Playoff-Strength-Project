# outputs8 — Model and role (best)

**Role:** **Official best** for playoff-outcome evaluation. Spearman-surrogate loss (not ListMLE), 40 Optuna trials, `listmle_target: playoff_outcome`, rolling [10,30]. Best Spearman 0.777 (combo_0033), best playoff_spearman 0.854 (combo_0038), best rank_mae/rank_rmse (combo_0033), best NDCG@30 (combo_0032).

**Model:** **Spearman-surrogate** loss for Model A (soft rank correlation objective) + Model B (XGB) + stacking. Sweep batch 20260217_042955 under `outputs8/sweeps/20260217_042955/combo_*/`. No standing rank as input in this sweep (stat_dim 21 or baseline feature set).

**Difference from outputs7:** outputs7 = ListMLE sweep; outputs8 = **Spearman-surrogate** sweep (same Optuna setup, different loss). outputs8 beats outputs7 on every primary metric (Spearman, playoff_spearman, rank_mae, rank_rmse, NDCG@30). This is the **production best** until RMSE-surrogate or other experiments are run and compared.

**See also:** [docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](../docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md), [docs/OUTPUTS8_SWEEP_ANALYSIS_02-17.md](../docs/OUTPUTS8_SWEEP_ANALYSIS_02-17.md), [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md).
