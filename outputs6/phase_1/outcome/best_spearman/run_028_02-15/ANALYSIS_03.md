# Analysis 03 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4303
- spearman: 0.6703
- mrr_top2: 0.1667
- mrr_top4: 0.1667
- ndcg_at_4 (Conference Finals (top 4)): 0.0621
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3150
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.3557
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4303
- ndcg_at_30: 0.4303
- rank_mae_pred_vs_playoff_outcome_rank: 5.8000
- rank_rmse_pred_vs_playoff_outcome_rank: 7.0285
- roc_auc_upset: 0.8444
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.6903
- ndcg_at_4_standings: 0.2709
- ndcg_at_16_standings: 0.5105
- ndcg_at_30_standings: 0.5302
- rank_rmse_standings: 6.8118
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.6507, ndcg_at_4_final_four=0.0621, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4303, brier_championship_odds=0.0312, rank_mae_pred_vs_playoff_outcome_rank=5.8000, rank_rmse_pred_vs_playoff_outcome_rank=7.0285, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.800 | 7.029 | -2.667 | -2.579 |
| Model A | 6.933 | 8.756 | -3.800 | -4.306 |
| Model B | 6.600 | 8.406 | -3.467 | -3.957 |
| Model C | 8.400 | 10.149 | -5.267 | -5.699 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.6667 | [-4.2333, -0.9667] | 0.9990 |
| Model A | -3.8000 | [-5.7000, -1.9667] | 1.0000 |
| Model B | -3.4667 | [-5.8000, -1.2000] | 0.9975 |
| Model C | -5.2667 | [-7.7000, -2.8667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
