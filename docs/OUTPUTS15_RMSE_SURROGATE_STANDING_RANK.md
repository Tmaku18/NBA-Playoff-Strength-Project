# outputs15: Rank RMSE surrogate sweep with standing rank

**Purpose:** Run an Optuna sweep with **rank_rmse_surrogate** loss and **standing rank as an input feature** (Model A/B/C). Same as outputs13_rmse_surrogate but with `standing_rank_norm` in the feature set (stat_dim 22, use_standing_rank true). Compare to outputs13 (RMSE surrogate, no standing), outputs10 (Spearman surrogate + standing), and outputs8 (Spearman surrogate, no standing).

---

## Setup

- **Config:** `config/outputs15_sweep_rmse_surrogate_standing_rank.yaml`
- **Training:** `loss_type: rank_rmse_surrogate`, `listmle_target: playoff_outcome`, rolling [10,30] (defaults)
- **Input:** Standing rank via `model_a.stat_dim: 22`, `model_a.use_standing_rank: true` (Model B uses defaults with standing_rank_norm)
- **Output root:** `outputs15_rmse_surrogate_standing_rank/`. Sweep writes to `outputs15_rmse_surrogate_standing_rank/sweeps/<batch_id>/`

---

## How to run (WSL)

From project root with `PYTHONPATH` set:

```bash
export PYTHONPATH="$PWD"
python -m scripts.sweep_hparams --config config/outputs15_sweep_rmse_surrogate_standing_rank.yaml --method optuna --objective rank_rmse --n-trials 40 --n-jobs 4 --listmle-target playoff_outcome --batch-id rmse_surrogate_standing_rank_40
```

- **rank_rmse** = minimize; other objectives (spearman, playoff_spearman, ndcg*) = maximize.
- Other objectives: `--objective spearman`, `--objective playoff_spearman`, `--objective ndcg4`, `--objective ndcg16`.

---

## After the sweep

- **Results:** `outputs15_rmse_surrogate_standing_rank/sweeps/<batch_id>/sweep_results_summary.json`, `sweep_results.csv`, `optuna_study.json`
- Compare to **outputs13_rmse_surrogate** (no standing), **outputs10_spearman_surrogate_standing_rank** (Spearman + standing), **outputs8_spearman_surrogate** (Spearman, no standing).

---

## Comparison

| Output | Loss | Standing rank | Use case |
|--------|------|---------------|----------|
| outputs8_spearman_surrogate | spearman_surrogate | No | Best Spearman / playoff_spearman (current best) |
| outputs10_spearman_surrogate_standing_rank | spearman_surrogate | Yes | Spearman surrogate + standing rank |
| outputs13_rmse_surrogate | rank_rmse_surrogate | No | RMSE surrogate; compare to outputs8 |
| **outputs15_rmse_surrogate_standing_rank** | **rank_rmse_surrogate** | **Yes** | RMSE surrogate + standing rank (this sweep) |
