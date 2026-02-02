# Hyperparameter Sweep — Comparative Analysis (Rolling Off vs Rolling On)

**Phase 1 batch:** `outputs3/sweeps/phase1_rolling_off`  
**Phase 2 batch:** `outputs3/sweeps/phase2_rolling_on`

## Summary

- **Phase 1 (rolling off):** 2 combos — rolling fixed at `[10, 30]`, epochs [16, 28].  
- **Phase 2 (rolling on):** 4 combos — rolling `[5, 10]` and `[10, 30]`, epochs [16, 28].

All six combos produced **identical** test metrics. Model A did not differentiate: training loss was flat (27.9 every epoch) and attention debug showed all-zero attention at inference. The ensemble is effectively driven by Model B (XGB + RF) only; rolling window choice and epoch count had no measurable effect in this setup.

## Metric comparison

| Metric | Phase 1 (2 combos) | Phase 2 (4 combos) |
|--------|-------------------|--------------------|
| Ensemble Spearman | 0.590 (both) | 0.590 (all 4) |
| Ensemble NDCG | 0.150 (both) | 0.150 (all 4) |
| rank_mae_pred_vs_playoff | 6.07 (both) | 6.07 (all 4) |
| rank_rmse_pred_vs_playoff | 7.84 (both) | 7.84 (all 4) |
| rank_mae_standings_vs_playoff | 3.13 (both) | 3.13 (all 4) |
| rank_rmse_standings_vs_playoff | 4.45 (both) | 4.45 (all 4) |
| Failures | 0 | 0 |

- **Variance:** Zero across combos in both phases.  
- **Rolling window effect:** None in this run; [5, 10] and [10, 30] gave the same metrics.  
- **Rank-distance interpretation:** Predicted rank is on average ~6.07 positions off actual EOS playoff rank; reg-season standings baseline is ~3.13 positions off. So the ensemble is roughly twice as far from playoff outcome as the standings baseline on this test set.

## Impact of varying rolling windows

- **Phase 1 vs Phase 2:** No difference in best or worst metrics.  
- **Best rolling window:** No winner — [5, 10] and [10, 30] tied (same numbers).  
- **Conclusion:** Until Model A is fixed and contributes (non-zero attention, decreasing loss), rolling window variation and epoch choice will not show up in ensemble metrics. Refining sweeps for “rolling off” vs “rolling on” is deferred until Model A is validated.

## Refined hyperparameter sweep presets

Based on these runs, use the following once Model A is contributing and you want to scale up:

### Rolling off (baseline)

- **rolling_windows:** `[[10, 30]]` only.  
- **model_a_epochs:** [16, 28] (expand to [8, 16, 24, 28] when tuning).  
- **model_b:** Keep max_depth [4], learning_rate [0.08], n_estimators_xgb [250], n_estimators_rf [200] as a first grid; then add depth [3, 5], lr [0.05, 0.10], n_xgb [200, 300], n_rf [150, 250] if needed.  
- **include_clone_classifier:** false for speed; set true for full evaluation.

### Rolling on (varied windows)

- **rolling_windows:** `[[5, 10], [10, 20], [10, 30], [15, 30]]` (or start with `[[5, 10], [10, 30]]` for a smaller sweep).  
- **model_a_epochs:** Same as rolling off.  
- **model_b:** Same as rolling off.  
- **include_clone_classifier:** false for speed.

### Recommendation

1. **Fix/validate Model A** so that loss decreases and attention is non-zero at inference; then re-run Phase 1 and Phase 2 (or a larger grid) to see real sensitivity to rolling windows and epochs.  
2. **Rank-distance metrics:** Continue reporting `rank_mae_pred_vs_playoff` and `rank_mae_standings_vs_playoff` in sweeps; use best_by_rank_mae (min) alongside best_by_spearman and best_by_ndcg.  
3. **Config:** Restore or keep in `config/defaults.yaml` the full sweep grid (e.g. rolling_windows with 4 entries, model_a_epochs [8, 16, 24, 28], expanded model_b) when running full sweeps; use the reduced grid above only for quick comparison runs.
