# outputs11: ListMLE with standing rank as input

**Purpose:** Run an Optuna sweep with the **ListMLE** loss (same as outputs9) and **standing rank as an input feature** for Model A and Model B (same as outputs10). Compare to **baseline (outputs6)**, outputs10 (Spearman-surrogate + standing), and outputs8 (Spearman-surrogate, no standing).

---

## Setup

- **Config:** `config/outputs11_sweep_listmle_standing_rank.yaml`
- **Training:** `loss_type: listmle`, `listmle_target: playoff_outcome`, `listmle_position_aware: false`
- **Input:** Standing rank via defaults (`stat_dim: 22` includes `standing_rank_norm` in Model A; Model B uses `TEAM_CONTEXT_FEATURE_COLS` with `standing_rank_norm`)
- **Output root:** `outputs11/`. Sweep writes to `outputs11/sweeps/<batch_id>/`

---

## How to run (WSL)

From project root with `PYTHONPATH` set:

```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
python -m scripts.sweep_hparams --config config/outputs11_sweep_listmle_standing_rank.yaml --method optuna --objective spearman --n-trials 40 --n-jobs 4 --listmle-target playoff_outcome --batch-id listmle_standing_rank_40
```

Other objectives (same as other sweeps):

- `--objective playoff_spearman`
- `--objective ndcg4` or `ndcg16`
- `--objective rank_rmse`

---

## After the sweep

- **Results:** `outputs11/sweeps/listmle_standing_rank_40/sweep_results_summary.json`, `sweep_results.csv`, `optuna_study.json`
- **Best config:** `outputs11/sweeps/listmle_standing_rank_40/combo_<NN>/config.yaml` (combo index from `best_optuna_trial` or `best_by_spearman`)
- If inference fails with missing `multi_temp_aggregation`, ensure `src/models/multi_temp_aggregation.py` exists (added for outputs10).

---

## Comparison matrix

| Batch    | Role / loss         | Standing rank | Best for |
|----------|---------------------|--------------|----------|
| **outputs6** | **Baseline** (listmle) | No          | Reference for comparisons |
| outputs8 | spearman_surrogate  | No          | Spearman / playoff_spearman (0.77 / 0.85) |
| outputs9 | listmle             | Yes (defaults) | ListMLE best Spearman ~0.48 (combo 16) |
| outputs10| spearman_surrogate  | Yes         | Standing + surrogate; Model A inverted |
| outputs11| listmle             | Yes         | ListMLE + standing (this sweep) |
