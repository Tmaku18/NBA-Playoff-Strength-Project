# Analysis 01 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.2989
- spearman: -0.7041
- mrr_top2: 0.0769
- mrr_top4: 0.0769
- ndcg_at_4 (Conference Finals (top 4)): 0.0000
- ndcg_at_12 (Clinch Playoff (top 12)): 0.0000
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.0861
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.1251
- ndcg_at_30: 0.2989
- rank_mae_pred_vs_playoff_outcome_rank: 13.9333
- rank_rmse_pred_vs_playoff_outcome_rank: 15.9792
- roc_auc_upset: 0.9467
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: 0.7326
- ndcg_at_4_standings: 0.0000
- ndcg_at_16_standings: 0.0211
- ndcg_at_30_standings: 0.2789
- rank_rmse_standings: 16.1121
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=-0.6908, ndcg_at_4_final_four=0.0000, ndcg_at_30_pred_vs_playoff_outcome_rank=0.2989, brier_championship_odds=0.0337, rank_mae_pred_vs_playoff_outcome_rank=13.9333, rank_rmse_pred_vs_playoff_outcome_rank=15.9792, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 13.933 | 15.979 | -10.800 | -11.529 |
| Model A | 7.067 | 9.092 | -3.933 | -4.642 |
| Model B | 5.733 | 7.510 | -2.600 | -3.060 |
| Model C | 8.400 | 10.149 | -5.267 | -5.699 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -10.8000 | [-13.6342, -8.0992] | 1.0000 |
| Model A | -3.9333 | [-5.9333, -2.0325] | 1.0000 |
| Model B | -2.6000 | [-4.8000, -0.4333] | 0.9905 |
| Model C | -5.2667 | [-7.7000, -2.8667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
