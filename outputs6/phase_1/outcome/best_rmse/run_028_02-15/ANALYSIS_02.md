# Analysis 02 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4140
- spearman: 0.7495
- mrr_top2: 0.1429
- mrr_top4: 0.1429
- ndcg_at_4 (Conference Finals (top 4)): 0.0496
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3367
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.3375
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4140
- ndcg_at_30: 0.4140
- rank_mae_pred_vs_playoff_outcome_rank: 5.2667
- rank_rmse_pred_vs_playoff_outcome_rank: 6.1264
- roc_auc_upset: 0.8089
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.7651
- ndcg_at_4_standings: 0.3223
- ndcg_at_16_standings: 0.5674
- ndcg_at_30_standings: 0.5680
- rank_rmse_standings: 5.9330
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.7255, ndcg_at_4_final_four=0.0496, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4140, brier_championship_odds=0.0315, rank_mae_pred_vs_playoff_outcome_rank=5.2667, rank_rmse_pred_vs_playoff_outcome_rank=6.1264, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.267 | 6.126 | -2.133 | -1.677 |
| Model A | 6.933 | 8.741 | -3.800 | -4.291 |
| Model B | 5.733 | 7.510 | -2.600 | -3.060 |
| Model C | 8.400 | 10.149 | -5.267 | -5.699 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.1333 | [-3.5000, -0.6667] | 0.9985 |
| Model A | -3.8000 | [-5.7000, -1.9333] | 1.0000 |
| Model B | -2.6000 | [-4.8000, -0.4333] | 0.9905 |
| Model C | -5.2667 | [-7.7000, -2.8667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
