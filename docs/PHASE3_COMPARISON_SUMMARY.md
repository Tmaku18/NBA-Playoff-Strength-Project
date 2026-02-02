# Phase 3: Comparison of Rolling On vs Rolling Off Runs

**Plan:** [.cursor/plans/Hyperparameter_Testing.md](../.cursor/plans/Hyperparameter_Testing.md)  
**Full analysis:** [docs/SWEEP_COMPARATIVE_ANALYSIS.md](SWEEP_COMPARATIVE_ANALYSIS.md)

## Runs compared

| Run | Batch | Combo | rolling_windows | Params (summary) |
|-----|--------|-------|-----------------|------------------|
| **Run A (rolling on)** | `outputs3/sweeps/phase2_rolling_on` | combo_0000 | [5, 10] | epochs 16, max_depth 4, lr 0.08, n_xgb 250, n_rf 200 |
| **Run B (rolling off)** | `outputs3/sweeps/phase1_rolling_off` | combo_0000 | [10, 30] | epochs 16, max_depth 4, lr 0.08, n_xgb 250, n_rf 200 |

Same primary metric “best” in both batches: best_by_spearman, best_by_ndcg, best_by_rank_mae all point to combo_0000 in each.

## Metric comparison (same control values)

| Metric | Run A (rolling on) | Run B (rolling off) |
|--------|--------------------|----------------------|
| Ensemble Spearman | 0.590 | 0.590 |
| Ensemble NDCG | 0.150 | 0.150 |
| rank_mae_pred_vs_playoff | 6.07 | 6.07 |
| rank_mae_standings_vs_playoff | 3.13 | 3.13 |
| Failures | 0 | 0 |

**Variance:** None; metrics are identical across combos in both phases.

## Conclusion

- **Rolling on vs rolling off:** No measurable difference in this setup. [5, 10] and [10, 30] produced the same test metrics.
- **Reason:** Model A did not differentiate (flat loss, all-zero attention); ensemble is effectively Model B (XGB + RF) only.
- **Recommendation:** Treat both runs as equivalent for now. Refine “rolling on vs off” sweeps **after** Model A is fixed and contributes (non-zero attention, decreasing loss). See docs/SWEEP_COMPARATIVE_ANALYSIS.md refined presets for the next sweep when ready.

## Params used for each run (for “most recent update”)

- **Run A (rolling on):** `training.rolling_windows: [5, 10]`, `model_a.epochs: 16`, model_b as above.
- **Run B (rolling off):** `training.rolling_windows: [10, 30]`, `model_a.epochs: 16`, model_b as above.

No single “rolling off preferred” or “rolling on preferred” conclusion; defer until Model A is validated.
