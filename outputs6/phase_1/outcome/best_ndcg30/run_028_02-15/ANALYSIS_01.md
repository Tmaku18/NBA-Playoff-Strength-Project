# Analysis 01 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.2922
- spearman: -0.7277
- mrr_top2: 0.0714
- mrr_top4: 0.0714
- ndcg_at_4 (Conference Finals (top 4)): 0.0000
- ndcg_at_12 (Clinch Playoff (top 12)): 0.0001
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.0833
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.0840
- ndcg_at_30: 0.2922
- rank_mae_pred_vs_playoff_outcome_rank: 14.3333
- rank_rmse_pred_vs_playoff_outcome_rank: 16.0893
- roc_auc_upset: 0.9643
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: 0.7473
- ndcg_at_4_standings: 0.0000
- ndcg_at_16_standings: 0.0008
- ndcg_at_30_standings: 0.2752
- rank_rmse_standings: 16.1802
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=-0.7028, ndcg_at_4_final_four=0.0000, ndcg_at_30_pred_vs_playoff_outcome_rank=0.2922, brier_championship_odds=0.0338, rank_mae_pred_vs_playoff_outcome_rank=14.3333, rank_rmse_pred_vs_playoff_outcome_rank=16.0893, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 14.333 | 16.089 | -11.200 | -11.640 |
| Model A | 7.267 | 9.118 | -4.133 | -4.668 |
| Model B | 5.333 | 6.880 | -2.200 | -2.430 |
| Model C | 8.400 | 10.149 | -5.267 | -5.699 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -11.2000 | [-13.8000, -8.7658] | 1.0000 |
| Model A | -4.1333 | [-6.0675, -2.3000] | 1.0000 |
| Model B | -2.2000 | [-4.2675, -0.2333] | 0.9845 |
| Model C | -5.2667 | [-7.7000, -2.8667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
