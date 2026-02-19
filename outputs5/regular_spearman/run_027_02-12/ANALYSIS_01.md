# Analysis 01 — Evaluation summary

**Run:** run_027
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.5974
- spearman: 0.4857
- mrr_top2: 0.5000
- mrr_top4: 0.5000
- ndcg_at_4 (Conference Finals (top 4)): 0.4644
- ndcg_at_12 (Clinch Playoff (top 12)): 0.4837
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.5263
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.5305
- ndcg_at_30: 0.5974
- rank_mae_pred_vs_playoff_outcome_rank: 6.8667
- rank_rmse_pred_vs_playoff_outcome_rank: 8.7788
- roc_auc_upset: 0.7589
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.4910
- ndcg_at_4_standings: 0.4432
- ndcg_at_16_standings: 0.4999
- ndcg_at_30_standings: 0.5956
- rank_rmse_standings: 8.7331
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.4857, ndcg_at_4_final_four=0.4644, ndcg_at_30_pred_vs_playoff_outcome_rank=0.5974, brier_championship_odds=0.0301, rank_mae_pred_vs_playoff_outcome_rank=6.8667, rank_rmse_pred_vs_playoff_outcome_rank=8.7788, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| ensemble | 6.867 | 8.779 | -3.733 | -4.329 |
| model_a | 6.867 | 8.779 | -3.733 | -4.329 |
| xgb | 5.933 | 7.694 | -2.800 | -3.244 |
| rf | 14.733 | 16.747 | -11.600 | -12.297 |

### Statistical significance (ensemble vs standings)

- **Method:** Paired bootstrap over teams (resample teams with replacement; mean improvement in MAE per team).
- **Mean MAE improvement:** -3.7333 (positive = ensemble better).
- **95% CI for improvement:** [-5.7000, -1.9667].
- **p-value (H0: no improvement):** 1.0000 → ensemble is **not significantly better** than standings at α=0.05.


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
