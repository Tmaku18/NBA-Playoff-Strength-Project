# outputs15_rmse_surrogate_standing_rank — Model and role

**Folder name:** `outputs15_rmse_surrogate_standing_rank`. See [OUTPUT_FOLDER_NAMING.md](../docs/OUTPUT_FOLDER_NAMING.md).

**Role:** **Rank RMSE surrogate** sweep with **standing rank as input** (Model A/B/C). Same methodology as outputs13 (40 Optuna trials, playoff_outcome, rolling [10,30], rank_rmse_surrogate loss) but with **standing_rank_norm** in the feature set (stat_dim 22, use_standing_rank true). Compare to outputs13_rmse_surrogate (no standing) and to outputs10_spearman_surrogate_standing_rank (Spearman surrogate + standing).

**Model:** **rank_rmse_surrogate** loss (Model A) + Model B (XGB) + stacking, with **standing_rank_norm** in roster and team-context features. Sweep writes to `outputs15_rmse_surrogate_standing_rank/sweeps/<batch_id>/combo_*/`. Config: `config/outputs15_sweep_rmse_surrogate_standing_rank.yaml`.

**Difference from outputs13:** outputs13 = RMSE surrogate **without** standing rank. outputs15 = **RMSE surrogate + standing rank** as input. Isolates the effect of standing rank for the RMSE-surrogate objective.

**Run in WSL (from project root):**
```bash
export PYTHONPATH="$PWD"
python -m scripts.sweep_hparams --config config/outputs15_sweep_rmse_surrogate_standing_rank.yaml --method optuna --objective rank_rmse --n-trials 40 --n-jobs 4 --listmle-target playoff_outcome --batch-id rmse_surrogate_standing_rank_40
```

**See also:** [docs/OUTPUTS15_RMSE_SURROGATE_STANDING_RANK.md](../docs/OUTPUTS15_RMSE_SURROGATE_STANDING_RANK.md), [docs/OUTPUTS13_RMSE_SURROGATE_SWEEP.md](../docs/OUTPUTS13_RMSE_SURROGATE_SWEEP.md), [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md).
