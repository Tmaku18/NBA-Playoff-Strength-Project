# Analysis 02 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4503
- spearman: 0.7384
- mrr_top2: 0.2000
- mrr_top4: 0.2000
- ndcg_at_4 (Conference Finals (top 4)): 0.0544
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3731
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.3732
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4503
- ndcg_at_30: 0.4503
- rank_mae_pred_vs_playoff_outcome_rank: 5.2000
- rank_rmse_pred_vs_playoff_outcome_rank: 6.2610
- roc_auc_upset: 0.8044
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.7424
- ndcg_at_4_standings: 0.3249
- ndcg_at_16_standings: 0.5832
- ndcg_at_30_standings: 0.6033
- rank_rmse_standings: 6.2129
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.7152, ndcg_at_4_final_four=0.0544, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4503, brier_championship_odds=0.0309, rank_mae_pred_vs_playoff_outcome_rank=5.2000, rank_rmse_pred_vs_playoff_outcome_rank=6.2610, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.200 | 6.261 | -2.067 | -1.811 |
| Model A | 6.867 | 8.764 | -3.733 | -4.314 |
| Model B | 5.667 | 7.371 | -2.533 | -2.921 |
| Model C | 8.467 | 10.218 | -5.333 | -5.768 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.0667 | [-3.5000, -0.5000] | 0.9960 |
| Model A | -3.7333 | [-5.6333, -1.8667] | 1.0000 |
| Model B | -2.5333 | [-4.6333, -0.4667] | 0.9920 |
| Model C | -5.3333 | [-7.7333, -2.9333] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
