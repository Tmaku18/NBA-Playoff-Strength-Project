---
name: Sweep Analyze Report Compare
overview: Run two hyperparameter sweeps (rolling windows not varied, then varied), analyze results, write a standalone report and a comparative analysis, and refine sweep configs for each setting based on findings. All runs in foreground with no time limit.
todos: []
isProject: false
---

# Sweep, Analyze, Report, and Comparative Analysis Plan

## Definitions

- **Rolling windows "off" (baseline):** Sweep with **no variation** in rolling windows — use a single value, e.g. `[[10, 30]]`, so all combos share the same player L10/L30-style features. Focus is on Model A epochs and Model B hyperparameters.
- **Rolling windows "on":** Sweep **with variation** in rolling windows — use the full list (e.g. `[[5, 10], [10, 20], [10, 30], [15, 30]]`) so the effect of window choice is included in the search.

No code change is required for "off": the pipeline already supports any list of window pairs; we only change the sweep config.

## Current sweep setup

- **Config:** [config/defaults.yaml](config/defaults.yaml) — `sweep.rolling_windows: [[5, 10], [10, 20], [10, 30], [15, 30]]`, `model_a_epochs: [8, 16, 24, 28]`, and a large Model B grid (depth, lr, n_estimators_xgb/rf, etc.), `include_clone_classifier: true`.
- **Script:** [scripts/sweep_hparams.py](scripts/sweep_hparams.py) — builds the Cartesian product of rolling_list × epochs × model_b params, runs 3 → 4 → 4b → 6 → 5 (and 4c if clone included), writes `outputs3/sweeps/<batch_id>/sweep_results.csv`, `sweep_results_summary.json`, and `sweep_config.json`.
- **Combo count (current):** 4 rolling × 4 epochs × 3 depth × 3 lr × 4 n_xgb × 3 n_rf × 1×1×1 = **1,728** combos. With clone classifier and no `--max-combos`, runtime will be very long (days). Plan below uses a **reduced grid** for the first pass so results and refinement are feasible; you can scale up afterward.

## Phase 1 — Sweep with rolling windows "off" (baseline)

1. **Set baseline sweep config**
  - In `config/defaults.yaml`, set `sweep.rolling_windows` to a single entry: `[[10, 30]]`.
  - Optionally reduce the grid for a first pass (e.g. `model_a_epochs: [16, 28]`, one or two values per Model B dimension, and/or `include_clone_classifier: false`) to get results in a few hours; document the reduced grid in the report.
2. **Run sweep (foreground)**
  - From repo root: `python -m scripts.sweep_hparams` (no `--batch-id` so a timestamped batch is used).
  - Let it run to completion (no background, no timeout). Outputs: `outputs3/sweeps/<batch_id_off>/` with `sweep_results.csv`, `sweep_results_summary.json`, and per-combo `combo_XXXX/outputs/`.
3. **Analyze results**
  - Load `sweep_results.csv` and `sweep_results_summary.json`.
  - Key metrics (from [scripts/sweep_hparams.py](scripts/sweep_hparams.py) `_collect_metrics`): `test_metrics_ensemble_spearman`, `test_metrics_ensemble_ndcg`, and optionally `test_metrics_ensemble_playoff_metrics`, `test_metrics_model_a_*`, `test_metrics_xgb_*`, `test_metrics_rf_*`.
  - Identify: best combo by Spearman and by NDCG; sensitivity to epochs, max_depth, learning_rate, n_estimators; any failures (column `error` if present).
4. **Write Phase 1 report**
  - Create a short report (e.g. `outputs3/sweeps/<batch_id_off>/sweep_report_rolling_off.md` or `docs/SWEEP_REPORT_ROLLING_OFF.md`) containing:
    - Sweep config used (rolling fixed at [10, 30], grid sizes).
    - Combo count and any failures.
    - Best configs (by Spearman and NDCG) and their metrics.
    - Brief interpretation: which hyperparameters helped most, recommended ranges for future sweeps (rolling-off).

## Phase 2 — Sweep with rolling windows "on" (varied)

1. **Turn rolling windows "on" in sweep**
  - In `config/defaults.yaml`, set `sweep.rolling_windows` to the full list: `[[5, 10], [10, 20], [10, 30], [15, 30]]`.
  - Optionally refine the rest of the grid using Phase 1 findings (e.g. narrow epochs or Model B to the best region) to keep combo count manageable.
2. **Run sweep again (foreground)**
  - `python -m scripts.sweep_hparams` (new timestamped batch). Outputs: `outputs3/sweeps/<batch_id_on>/`.
3. **Analyze results**
  - Same as Phase 1, plus: best rolling window pair(s); correlation between window choice and ensemble/model metrics.
4. **Write comparative analysis**
  - Create a comparative report (e.g. `docs/SWEEP_COMPARATIVE_ANALYSIS.md` or `outputs3/sweeps/SWEEP_COMPARATIVE_ANALYSIS.md`) that:
    - Compares Phase 1 vs Phase 2: best Spearman/NDCG, variance across combos, failure rates.
    - Discusses impact of varying rolling windows: whether any window pair consistently wins, and how much metrics change vs baseline.
    - Summarizes recommended hyperparameter ranges for **rolling-off** vs **rolling-on** sweeps.

## Phase 3 — Refine hyperparameter sweeps

1. **Document refined configs**
  - Using the comparative analysis, write down two refined sweep presets:
    - **Rolling off:** e.g. `rolling_windows: [[10, 30]]`, narrowed `model_a_epochs` and Model B ranges that performed best in Phase 1.
    - **Rolling on:** e.g. `rolling_windows: [[5, 10], [10, 20], [10, 30], [15, 30]]` (or a subset if one pair dominated), plus narrowed epochs and Model B from Phase 2.
  - Store these either in:
    - The comparative report (recommended), or
    - Comments or a small block in [config/defaults.yaml](config/defaults.yaml) (e.g. “refined sweep – rolling off” vs “refined sweep – rolling on”), or
    - Optional: separate YAML snippets under `config/` (e.g. `sweep_rolling_off.yaml`, `sweep_rolling_on.yaml`) that override only the `sweep` section when passed to a wrapper.
2. **No background runs**
  - All `sweep_hparams` runs are executed in the foreground; no daemon or background mode. If the full grid is too large, use `--max-combos N` for a first pass and document N in the reports.

## Execution order summary


| Step | Action                                                                                                                                                               |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | Set `sweep.rolling_windows: [[10, 30]]` (and optionally reduce grid); run `python -m scripts.sweep_hparams`.                                                         |
| 2    | Analyze Phase 1 outputs; write `sweep_report_rolling_off.md`.                                                                                                        |
| 3    | Set `sweep.rolling_windows: [[5, 10], [10, 20], [10, 30], [15, 30]]` (and optionally refine other params from Phase 1); run `python -m scripts.sweep_hparams` again. |
| 4    | Analyze Phase 2 outputs; write comparative report.                                                                                                                   |
| 5    | Document refined hyperparameter sweeps for rolling-off and rolling-on in the comparative report (and optionally in config).                                          |


## Files to create or modify

- **Modify:** [config/defaults.yaml](config/defaults.yaml) — toggle `sweep.rolling_windows` and optionally other sweep keys between Phase 1 and Phase 2; optionally add refined preset comments after Phase 3.
- **Create:** Phase 1 report (e.g. `docs/SWEEP_REPORT_ROLLING_OFF.md` or under `outputs3/sweeps/<batch_id_off>/`).
- **Create:** Comparative report (e.g. `docs/SWEEP_COMPARATIVE_ANALYSIS.md`).
- **Use (read-only):** [scripts/sweep_hparams.py](scripts/sweep_hparams.py), [scripts/5_evaluate.py](scripts/5_evaluate.py) — no changes required for this plan.

## Optional: Notion

If the project is connected to Notion, after the reports exist you can add a short summary and links to the Phase 1 report and the comparative analysis on the relevant Notion page (e.g. NBA True Strength — Project Master), with the robot icon convention applied.