# outputs13: Rank RMSE surrogate sweep

**Purpose:** Run an Optuna sweep with **rank_rmse_surrogate** loss (minimize rank RMSE between predicted and playoff-outcome rank). Same pipeline and evaluation as outputs8 (Spearman surrogate), but training objective = rank RMSE instead of Spearman. Compare best rank_rmse, Spearman, and playoff_spearman to outputs8 to see whether optimizing for RMSE improves rank error or trades off vs correlation.

---

## Setup

- **Config:** `config/outputs13_sweep_rmse_surrogate.yaml`
- **Training:** `loss_type: rank_rmse_surrogate`, `listmle_target: playoff_outcome`, rolling [10,30] (defaults)
- **Output root:** `outputs13_rmse_surrogate/`. Sweep writes to `outputs13_rmse_surrogate/sweeps/<batch_id>/`

---

## How to run (WSL)

From project root with `PYTHONPATH` set:

```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
python -m scripts.sweep_hparams --config config/outputs13_sweep_rmse_surrogate.yaml --method optuna --objective rank_rmse --n-trials 40 --n-jobs 4 --listmle-target playoff_outcome --batch-id rmse_surrogate_40
```

- **rank_rmse** = minimize (Optuna direction `minimize`); other objectives (spearman, playoff_spearman, ndcg*) = maximize.
- Other objectives for the same sweep config: `--objective spearman`, `--objective playoff_spearman`, `--objective ndcg4`, `--objective ndcg16`.

---

## Sweep results (rmse_surrogate_40)

- **Best by Optuna objective (rank_rmse):** **combo_004** — rank_rmse 13.21, Spearman -0.16, playoff_spearman -0.30. No combo achieved positive correlation; RMSE surrogate underperformed Spearman surrogate on all primary metrics.
- **Full analysis:** [outputs13_rmse_surrogate/sweeps/rmse_surrogate_40/ANALYSIS_01.md](../outputs13_rmse_surrogate/sweeps/rmse_surrogate_40/ANALYSIS_01.md)
- **Conclusion:** outputs8 (Spearman surrogate) remains the official best; RMSE surrogate sweep is not recommended for production.

---

## Single run: best combo with more epochs (combo_004 high epochs)

If you suspect the sweep underperformed because **Model A epochs were too low** (loss still decreasing at 22), run the same best combo with a larger epoch budget:

- **Config:** `config/outputs13_combo004_high_epochs.yaml` (combo_004 params, **80 epochs** max, **early_stopping_patience: 10**).
- **Command (from project root):**
  ```bash
  export PYTHONPATH="$PWD"
  python -m scripts.run_pipeline_from_model_a --config config/outputs13_combo004_high_epochs.yaml
  ```
- **Outputs:** `outputs13_rmse_surrogate/combo_004_high_epochs/`. Script 3 writes **`training_loss.csv`** (epoch, train_loss, val_loss) in that folder so you can inspect the curve; also watch stdout for `epoch N loss=X`.

---

## After the sweep

- **Results:** `outputs13_rmse_surrogate/sweeps/rmse_surrogate_40/sweep_results_summary.json`, `sweep_results.csv`, `optuna_study.json`
- **Best by rank_rmse:** `summary["best_by_rank_rmse"]`; also `best_by_spearman`, `best_by_playoff_spearman` for comparison
- Compare to **outputs8_spearman_surrogate** (Spearman surrogate): rank_rmse, Spearman, playoff_spearman, rank_mae

---

## Comparison

| Output   | Loss                  | Optuna objective (typical) | Use case |
|----------|------------------------|----------------------------|----------|
| outputs8_spearman_surrogate | spearman_surrogate    | spearman                   | Best Spearman / playoff_spearman (current best) |
| outputs13_rmse_surrogate | rank_rmse_surrogate   | rank_rmse                  | Best rank_rmse; compare to outputs8_spearman_surrogate on full metric set |
