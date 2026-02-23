# outputs9_listmle_spearman — Model and role

**Folder name:** `outputs9_listmle_spearman` (rename from `outputs9` to match; see [OUTPUT_FOLDER_NAMING.md](../docs/OUTPUT_FOLDER_NAMING.md).)

**Role:** ListMLE sweep with **Spearman as Optuna objective** (same evaluation target as outputs8, but Model A trained with ListMLE loss, not Spearman surrogate). 40 trials, `listmle_target: playoff_outcome`, rolling [10,30]. Standing rank as input (stat_dim 22) when using defaults.

**Model:** ListMLE (Model A) + Model B + stacking. Sweep batch e.g. `outputs9_listmle_spearman` under `outputs9/sweeps/outputs9_listmle_spearman/combo_*/`. Best Spearman ~0.477 (combo_16); best NDCG@4 ~0.475 (combo_18). Lower than outputs8 on Spearman/playoff_spearman; stronger on NDCG@4 than outputs8’s best Spearman combo.

**Difference from outputs8:** outputs8 = **Spearman-surrogate** loss (best overall metrics). outputs9 = **ListMLE** loss with same Optuna objective (spearman). Directly compares ListMLE vs surrogate: surrogate (outputs8) wins on correlation and rank error; ListMLE (outputs9) can do better on NDCG at top-4.

**Run in WSL (from project root):**
```bash
export PYTHONPATH="$PWD"
python -m scripts.sweep_hparams --config config/outputs9_sweep_listmle.yaml --method optuna --objective spearman --n-trials 40 --n-jobs 4 --listmle-target playoff_outcome
```

**See also:** [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md), [docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](../docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md), `scripts/aggregate_outputs9.py` for eval summary.
