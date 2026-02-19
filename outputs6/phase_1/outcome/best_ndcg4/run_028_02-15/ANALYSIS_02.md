# Analysis 02 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4269
- spearman: 0.6930
- mrr_top2: 0.1667
- mrr_top4: 0.1667
- ndcg_at_4 (Conference Finals (top 4)): 0.0544
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3083
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.3511
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4269
- ndcg_at_30: 0.4269
- rank_mae_pred_vs_playoff_outcome_rank: 5.8000
- rank_rmse_pred_vs_playoff_outcome_rank: 6.7823
- roc_auc_upset: 0.8444
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.7295
- ndcg_at_4_standings: 0.3249
- ndcg_at_16_standings: 0.5635
- ndcg_at_30_standings: 0.5836
- rank_rmse_standings: 6.3666
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.6868, ndcg_at_4_final_four=0.0544, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4269, brier_championship_odds=0.0312, rank_mae_pred_vs_playoff_outcome_rank=5.8000, rank_rmse_pred_vs_playoff_outcome_rank=6.7823, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.800 | 6.782 | -2.667 | -2.333 |
| Model A | 6.867 | 8.862 | -3.733 | -4.412 |
| Model B | 6.067 | 7.690 | -2.933 | -3.240 |
| Model C | 8.467 | 10.224 | -5.333 | -5.774 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.6667 | [-4.0333, -1.2000] | 1.0000 |
| Model A | -3.7333 | [-5.6333, -1.9658] | 1.0000 |
| Model B | -2.9333 | [-5.1675, -0.8000] | 0.9965 |
| Model C | -5.3333 | [-7.7333, -2.9333] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
