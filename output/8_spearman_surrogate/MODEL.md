# outputs8_spearman_surrogate — Model and role (best)

**Folder name:** `outputs8_spearman_surrogate` (rename from `outputs8` to match; see [OUTPUT_FOLDER_NAMING.md](../docs/OUTPUT_FOLDER_NAMING.md).)

**Role:** **Official best** for playoff-outcome evaluation. Spearman-surrogate loss (not ListMLE), 40 Optuna trials, `listmle_target: playoff_outcome`, rolling [10,30]. Best Spearman 0.777 (combo_0033), best playoff_spearman 0.854 (combo_0038), best rank_mae/rank_rmse (combo_0033), best NDCG@30 (combo_0032).

**Model:** **Spearman-surrogate** loss for Model A (soft rank correlation objective) + Model B (XGB) + stacking. Sweep batch 20260217_042955 under `outputs8/sweeps/20260217_042955/combo_*/`. No standing rank as input in this sweep (stat_dim 21 or baseline feature set).

**Difference from outputs7:** outputs7 = ListMLE sweep; outputs8 = **Spearman-surrogate** sweep (same Optuna setup, different loss). outputs8 beats outputs7 on every primary metric (Spearman, playoff_spearman, rank_mae, rank_rmse, NDCG@30). This is the **production best** until RMSE-surrogate or other experiments are run and compared.

**Run in WSL (from project root):**
```bash
export PYTHONPATH="$PWD"
python -m scripts.sweep_hparams --config config/outputs8_sweep_spearman.yaml --method optuna --objective spearman --n-trials 40 --n-jobs 4 --listmle-target playoff_outcome
```

**See also:** [docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](../docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md), [docs/OUTPUTS8_SWEEP_ANALYSIS_02-17.md](../docs/OUTPUTS8_SWEEP_ANALYSIS_02-17.md), [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md).

---

## Improved runs (Feb 27, 2026)

The 7 ranked improvements from [docs/PROJECT_STATE_AND_BEST_MODELS_02-27.md](../docs/PROJECT_STATE_AND_BEST_MODELS_02-27.md) are implemented:

- **`improved_02-27/`** — combo_0033 HPs re-run with the **fixed surrogate-loss sign** (the pre-fix loss inverted Model A: best team got the lowest score, and the meta compensated with a negative coefficient), **standings win rate as an extra stacking column**, **confidence-weighted stacking**, lifted caps (250 lists) + early stopping (patience 4), consistent higher-is-better meta target, and true per-conference metas. Config: `config/8_spearman_improved.yaml`.
- **`improved_topweighted_02-27/`** — same, plus `weighted_spearman_surrogate` loss (weights ∝ 1/rank) targeting the weak top-end (ndcg@4 0.042 / precision@4 0.0 in combo_0033). Config: `config/8_spearman_improved_topweighted.yaml`.
- **Flag ablation:** `python -m scripts.sweep_hparams --phase flags --config config/8_spearman_improved.yaml --n-jobs 3` (8 combos toggling sos_srs / team_rolling / injury).

**Caution:** pre-fix checkpoints (including combo_0033's `best_deep_set.pt` and metas) keep the inverted Model A convention internally; do not mix pre-fix models with post-fix metas.

---

## Pipeline deep-dive retrain (`improved_07-03`, Jul 2026)

First full retrain after correcting feature corruption, leakage, OOF/stacking validity, and evaluation fairness. Config: `config/8_spearman_improved.yaml`.

**Key code changes:** season-scoped aggregations; causal Massey/SRS; RAPTOR carry-forward; deterministic training + 3-seed Model A averaging; causal expanding-window OOF; multi-temp OOF/inference parity; 3-column rank-transform meta (no confidence cols); fair eval (`standings_to_date_rank`, 3 checkpoints/season).

**Results (pooled 6 checkpoints):**

| Metric | improved_07-03 | improved_02-27 run_026 |
|--------|------------------|-------------------------|
| Ensemble Spearman | **0.750** | 0.524 |
| Model B Spearman | **0.760** | 0.522 |
| Model A Spearman | 0.547 | 0.566 |
| Standings MAE (fair) | 3.97 | 3.13 (unfair EOS baseline) |

**Analysis:** [improved_07-03/ANALYSIS_02.md](improved_07-03/ANALYSIS_02.md)

**Next run:** `team_rolling` + `injury` enabled in config (flag-ablation winners) → `output/8_spearman_surrogate/improved_07-05/outputs`. See README for WSL command.
