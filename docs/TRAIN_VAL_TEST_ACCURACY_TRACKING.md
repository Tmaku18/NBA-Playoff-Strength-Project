# Train / validation / test accuracy tracking

**Purpose:** Record how to enable and read **training**, **validation**, and **test** ranking metrics for the ensemble and for each model (Model A, Model B, Model C).

---

## Enabling

In your config (e.g. `config/defaults.yaml` or a sweep overlay), set:

```yaml
inference:
  also_train_predictions: true    # write train_predictions.json (last train date)
  also_validation_predictions: true  # write val_predictions.json (last 20% of train dates)
```

Then run the full pipeline (scripts 3 → 4 → 4b → 6 → 5). Inference (script 6) will write:

- **Test:** `predictions.json` and/or `predictions_<season>.json` (unchanged).
- **Train:** `train_predictions.json` when `also_train_predictions: true`.
- **Validation:** `val_predictions.json` when `also_validation_predictions: true` (validation = last 20% of training dates by date).

Evaluation (script 5) fills `eval_report.json` with:

- `test_metrics_ensemble`, `test_metrics_model_a`, `test_metrics_model_b`, `test_metrics_model_c`
- `train_metrics_*` (same structure) when `train_predictions.json` exists
- `val_metrics_*` (same structure) when `val_predictions.json` exists

Each of these can include `playoff_metrics` (Spearman, NDCG@4, rank_mae, rank_rmse, etc.) when applicable, and `*_by_conference` for per-conference metrics.

---

## Report layout

In `eval_report.json`:

| Key | When present |
|-----|----------------------|
| `test_metrics_ensemble` | Always (from test predictions). |
| `test_metrics_model_a`, `test_metrics_model_b`, `test_metrics_model_c` | When test predictions include per-model scores. |
| `train_metrics_ensemble`, `train_metrics_model_a`, … | When `train_predictions.json` exists. |
| `val_metrics_ensemble`, `val_metrics_model_a`, … | When `val_predictions.json` exists. |
| `notes.eval_on` | `"test"`, `"test+train"`, `"test+val"`, or `"test+train+val"`. |
| `notes.train_val_test_accuracy` | Short description of train/val/test metrics. |

Per-model metrics use the same structure: scalars (e.g. `spearman`, `ndcg_at_4`, `rank_mae`, `rank_rmse`) and, when playoff data exist, a nested `playoff_metrics` object.

---

## Validation definition

Validation predictions use the **last 20% of training dates** (by sorted date). The inference run for validation uses the **last date** in that 20% so validation is a single snapshot, comparable to the single-date train and test snapshots.

This matches the idea of a temporal holdout within the training window (as in script 4’s XGB validation holdout).

---

## Related

- **Config:** `config/defaults.yaml` → `inference.also_train_predictions`, `inference.also_validation_predictions`.
- **README:** Evaluation section describes train/val/test accuracy and these flags.
- **Script 5:** `scripts/5_evaluate.py` computes and writes all `*_metrics_*` into `eval_report.json`.
