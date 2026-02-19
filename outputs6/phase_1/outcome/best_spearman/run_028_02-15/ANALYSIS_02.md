# Analysis 02 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.3027
- spearman: -0.6770
- mrr_top2: 0.0833
- mrr_top4: 0.0833
- ndcg_at_4 (Conference Finals (top 4)): 0.0000
- ndcg_at_12 (Clinch Playoff (top 12)): 0.0878
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.1292
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.1292
- ndcg_at_30: 0.3027
- rank_mae_pred_vs_playoff_outcome_rank: 13.7333
- rank_rmse_pred_vs_playoff_outcome_rank: 15.8514
- roc_auc_upset: 0.9333
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: 0.6859
- ndcg_at_4_standings: 0.0001
- ndcg_at_16_standings: 0.0323
- ndcg_at_30_standings: 0.2806
- rank_rmse_standings: 15.8934
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=-0.6592, ndcg_at_4_final_four=0.0000, ndcg_at_30_pred_vs_playoff_outcome_rank=0.3027, brier_championship_odds=0.0337, rank_mae_pred_vs_playoff_outcome_rank=13.7333, rank_rmse_pred_vs_playoff_outcome_rank=15.8514, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 13.733 | 15.851 | -10.600 | -11.402 |
| Model A | 6.933 | 8.896 | -3.800 | -4.446 |
| Model B | 6.600 | 8.406 | -3.467 | -3.957 |
| Model C | 8.400 | 10.149 | -5.267 | -5.699 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -10.6000 | [-13.4000, -7.8667] | 1.0000 |
| Model A | -3.8000 | [-5.7333, -1.9000] | 1.0000 |
| Model B | -3.4667 | [-5.8000, -1.2000] | 0.9975 |
| Model C | -5.2667 | [-7.7000, -2.8667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
