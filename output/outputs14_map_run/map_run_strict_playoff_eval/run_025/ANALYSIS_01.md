# Analysis 01 — Evaluation summary

**Run:** run_025
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.3535
- spearman: -0.1158
- kendall_tau: -0.0994
- pearson: -0.0406
- precision_at_4: 0.0000
- precision_at_8: 0.3750
- mrr_top2: 0.0833
- mrr_top4: 0.1250
- ndcg_at_4 (Conference Finals (top 4)): 0.0207
- ndcg_at_12 (Clinch Playoff (top 12)): 0.2257
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.3153
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.3535
- ndcg_at_30: 0.3535
- rank_mae_pred_vs_playoff_outcome_rank: 7.8947
- rank_rmse_pred_vs_playoff_outcome_rank: 9.7926
- roc_auc_upset: 0.9000
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 4.0000
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 6.0611
- spearman_standings: 0.0070
- kendall_tau_standings: 0.0409
- ndcg_at_4_standings: 0.1650
- ndcg_at_16_standings: 0.4229
- ndcg_at_30_standings: 0.4243
- rank_rmse_standings: 7.7731
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=-0.1772, kendall_tau_pred_vs_playoff_outcome_rank=-0.1228, ndcg_at_4_final_four=0.0206, ndcg_at_30_pred_vs_playoff_outcome_rank=0.3534, brier_championship_odds=0.0527, ece_championship_odds=0.0239, champion_rank=12, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 4.000 | 6.061 | — | — |
| Ensemble | 7.895 | 9.793 | -3.895 | -3.731 |
| Model A | 8.842 | 11.360 | -4.842 | -5.299 |
| Model B | 7.263 | 9.375 | -3.263 | -3.314 |
| Model C | 11.158 | 13.623 | -7.158 | -7.562 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.547 | 0.056 | 0.030 | 7.833 |
| West (W) | 0.517 | -0.607 | -0.524 | 8.000 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -3.8947 | [-5.9474, -1.7895] | 1.0000 |
| Model A | -4.8421 | [-7.0000, -2.7368] | 1.0000 |
| Model B | -3.2632 | [-5.3158, -1.2105] | 0.9995 |
| Model C | -7.1579 | [-9.7368, -4.3145] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
