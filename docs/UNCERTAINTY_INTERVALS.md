# Uncertainty intervals (predicted rank ranges)

This project outputs **rank intervals** (not just point ranks) so you can interpret predictions as **\(rank \pm range\)** and evaluate calibration (coverage vs width).

## Where intervals appear

### `predictions.json`

- **Ensemble (primary):** under `prediction`
  - `predicted_strength` (point rank)
  - `predicted_strength_low`, `predicted_strength_high` (interval)
  - `predicted_strength_minus`, `predicted_strength_plus` (distance to bounds)

- **Per-model diagnostics:** under `ensemble_diagnostics`
  - `model_a_rank`, `model_a_rank_low`, `model_a_rank_high`
  - `model_b_rank`, `model_b_rank_low`, `model_b_rank_high`
  - `model_c_rank`, `model_c_rank_low`, `model_c_rank_high`
  - `extra_model_ranks` (when enabled): `{model_name: {rank_low, rank_high}}`

## How intervals are computed

Intervals are produced via **Monte Carlo rank sampling** (`uncertainty.method: mc_rank_interval`):

1. Each model provides a **score distribution** per team: mean score + score std.
2. We draw `mc_samples` score samples for each team in a list and compute sampled **rank** each time.
3. We report the central interval with tail mass `alpha` (e.g. `alpha=0.1` → 90% interval).

### Score std sources

- **Model A:** heuristic from attention confidence: `std = (1 - conf_a) * conf_to_std_scale`
- **Model B (XGB):** per-tree std from `predict_with_uncertainty` (`src/models/xgb_model.py`)
- **Model C (RF):** per-tree std across `rf.estimators_` (when available)
- **Extra team-stats models:** model-native predictive std when available (BayesianRidge / GPR / GMM-supervised mixture), or residual std (LinearRegression).

## Evaluation

`scripts/5_evaluate.py` writes `uncertainty_metrics` to `eval_report.json`:

- `coverage`: fraction of teams where `EOS_global_rank` lies in `[low, high]`
- `mean_width`: average `(high - low)` interval width
- `n`: number of evaluated teams (with actual rank + interval present)

## Config knobs

In `config/defaults.yaml`:

- `uncertainty.enabled`
- `uncertainty.mc_samples` (default 200)
- `uncertainty.alpha` (default 0.1 → 90% interval)
- `uncertainty.score_std_floor`
- `uncertainty.conf_to_std_scale` (Model A heuristic scaling)

