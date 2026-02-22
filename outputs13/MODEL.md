# outputs13_rmse_surrogate — Model and role (RMSE surrogate)

**Folder name:** `outputs13_rmse_surrogate` (rename from `outputs13` to match; see [OUTPUT_FOLDER_NAMING.md](../docs/OUTPUT_FOLDER_NAMING.md).)

**Role:** **Rank RMSE surrogate** sweep. Same methodology as outputs8 (40 Optuna trials, playoff_outcome, rolling [10,30]) but Model A trains with **rank_rmse_surrogate** loss. Optuna typically minimizes **rank_rmse** (pred vs playoff outcome rank). Use to compare best rank_rmse, Spearman, and playoff_spearman vs outputs8 (Spearman surrogate).

**Model:** **rank_rmse_surrogate** loss (Model A) + Model B (XGB) + stacking. Sweep batch e.g. `rmse_surrogate_40` under `outputs13/sweeps/<batch_id>/combo_*/`. No standing rank as input unless overridden (defaults have stat_dim 22; sweep uses same baseline as outputs8).

**Difference from outputs10:** outputs10 = Spearman surrogate **with standing rank as input**. outputs13 = **RMSE surrogate** (no standing rank in baseline config), dedicated folder so RMSE-surrogate results are separate from Spearman-surrogate (outputs8) and ListMLE (outputs7/9).

**See also:** [docs/OUTPUTS13_RMSE_SURROGATE_SWEEP.md](../docs/OUTPUTS13_RMSE_SURROGATE_SWEEP.md), [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md), [docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](../docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md).
