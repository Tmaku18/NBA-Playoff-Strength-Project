# outputs11_listmle_standing_rank — Model and role

**Folder name:** `outputs11_listmle_standing_rank` (rename from `outputs11` to match; see [OUTPUT_FOLDER_NAMING.md](../docs/OUTPUT_FOLDER_NAMING.md).)

**Role:** ListMLE sweep with **standing rank as input** (Model A/B/C). Same standing-rank feature as outputs10, but training loss = **listmle** (not Spearman surrogate). Compare to **baseline (outputs6)** and to outputs10 (Spearman surrogate + standing).

**Model:** ListMLE (Model A) + Model B + stacking, with **standing_rank_norm** in features (stat_dim 22). Sweep writes to `outputs11/sweeps/<batch_id>/combo_*/`. Config: `config/outputs11_sweep_listmle_standing_rank.yaml`. `listmle_target: playoff_outcome`; typical objective: spearman.

**Difference from outputs10:** outputs10 = Spearman surrogate + standing rank. outputs11 = **ListMLE** + standing rank (Optuna sweep). Compare both to **baseline (outputs6)** (no standing) to isolate effect of standing and of loss type.

**Run in WSL (from project root):**
```bash
export PYTHONPATH="$PWD"
python -m scripts.sweep_hparams --config config/outputs11_sweep_listmle_standing_rank.yaml --method optuna --objective spearman --n-trials 40 --n-jobs 4 --listmle-target playoff_outcome --batch-id listmle_standing_rank_40
```

**See also:** [docs/OUTPUTS11_LISTMLE_STANDING_RANK.md](../docs/OUTPUTS11_LISTMLE_STANDING_RANK.md), [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md).
