# Analysis 01 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.2982
- spearman: -0.7028
- mrr_top2: 0.0833
- mrr_top4: 0.0833
- ndcg_at_4 (Conference Finals (top 4)): 0.0000
- ndcg_at_12 (Clinch Playoff (top 12)): 0.0878
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.0886
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.1263
- ndcg_at_30: 0.2982
- rank_mae_pred_vs_playoff_outcome_rank: 14.0667
- rank_rmse_pred_vs_playoff_outcome_rank: 15.9729
- roc_auc_upset: 0.9556
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: 0.7161
- ndcg_at_4_standings: 0.0000
- ndcg_at_16_standings: 0.0216
- ndcg_at_30_standings: 0.2774
- rank_rmse_standings: 16.0354
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=-0.6850, ndcg_at_4_final_four=0.0000, ndcg_at_30_pred_vs_playoff_outcome_rank=0.2982, brier_championship_odds=0.0338, rank_mae_pred_vs_playoff_outcome_rank=14.0667, rank_rmse_pred_vs_playoff_outcome_rank=15.9729, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 14.067 | 15.973 | -10.933 | -11.523 |
| Model A | 6.933 | 8.941 | -3.800 | -4.491 |
| Model B | 5.667 | 7.371 | -2.533 | -2.921 |
| Model C | 8.467 | 10.218 | -5.333 | -5.768 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -10.9333 | [-13.6342, -8.2992] | 1.0000 |
| Model A | -3.8000 | [-5.7667, -1.8333] | 1.0000 |
| Model B | -2.5333 | [-4.6333, -0.4667] | 0.9920 |
| Model C | -5.3333 | [-7.7333, -2.9333] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
