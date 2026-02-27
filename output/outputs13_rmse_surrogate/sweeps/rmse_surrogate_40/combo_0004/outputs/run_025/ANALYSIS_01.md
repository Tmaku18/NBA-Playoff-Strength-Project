# Analysis 01 — Evaluation summary

**Run:** run_025
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.2774
- spearman: -0.1640
- kendall_tau: -0.0759
- pearson: -0.1640
- precision_at_4: 0.0000
- precision_at_8: 0.1250
- mrr_top2: 0.0417
- mrr_top4: 0.0667
- ndcg_at_4 (Conference Finals (top 4)): 0.0056
- ndcg_at_12 (Clinch Playoff (top 12)): 0.0118
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.0321
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.0322
- ndcg_at_30: 0.2774
- rank_mae_pred_vs_playoff_outcome_rank: 11.0000
- rank_rmse_pred_vs_playoff_outcome_rank: 13.2061
- roc_auc_upset: 0.8333
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: 0.2850
- kendall_tau_standings: 0.1816
- ndcg_at_4_standings: 0.0007
- ndcg_at_16_standings: 0.0694
- ndcg_at_30_standings: 0.2861
- rank_rmse_standings: 13.8756
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=-0.2957, kendall_tau_pred_vs_playoff_outcome_rank=-0.1770, ndcg_at_4_final_four=0.0056, ndcg_at_30_pred_vs_playoff_outcome_rank=0.2774, brier_championship_odds=0.0341, ece_championship_odds=0.0000, champion_rank=30, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 11.000 | 13.206 | -7.867 | -8.756 |
| Model A | 11.200 | 13.352 | -8.067 | -8.902 |
| Model B | 8.133 | 10.260 | -5.000 | -5.810 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.108 | -0.107 | -0.029 | 10.867 |
| West (W) | 0.207 | -0.171 | -0.124 | 11.133 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -7.8667 | [-10.9675, -4.9333] | 1.0000 |
| Model A | -8.0667 | [-11.1342, -5.2333] | 1.0000 |
| Model B | -5.0000 | [-7.1667, -3.0000] | 1.0000 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
