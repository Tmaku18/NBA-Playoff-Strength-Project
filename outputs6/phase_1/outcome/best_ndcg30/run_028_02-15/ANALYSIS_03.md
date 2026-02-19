# Analysis 03 — Evaluation summary

**Run:** run_028_02-15
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4315
- spearman: 0.7268
- kendall_tau: 0.5218
- pearson: 0.7268
- precision_at_4: 0.0000
- precision_at_8: 0.7500
- mrr_top2: 0.1667
- mrr_top4: 0.1667
- ndcg_at_4 (Conference Finals (top 4)): 0.0496
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3542
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.3549
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4315
- ndcg_at_30: 0.4315
- rank_mae_pred_vs_playoff_outcome_rank: 5.3333
- rank_rmse_pred_vs_playoff_outcome_rank: 6.3979
- roc_auc_upset: 0.7330
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.7722
- kendall_tau_standings: -0.5724
- ndcg_at_4_standings: 0.3223
- ndcg_at_16_standings: 0.5833
- ndcg_at_30_standings: 0.5839
- rank_rmse_standings: 5.8424
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.7295, kendall_tau_pred_vs_playoff_outcome_rank=0.5402, ndcg_at_4_final_four=0.0496, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4315, brier_championship_odds=0.0312, ece_championship_odds=0.0000, champion_rank=6, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.333 | 6.398 | -2.200 | -1.948 |
| Model A | 6.800 | 8.899 | -3.667 | -4.450 |
| Model B | 5.333 | 6.880 | -2.200 | -2.430 |
| Model C | 8.400 | 10.149 | -5.267 | -5.699 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.583 | 0.625 | 0.429 | 6.200 |
| West (W) | 0.617 | 0.843 | 0.638 | 4.467 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.2000 | [-3.6333, -0.7667] | 0.9995 |
| Model A | -3.6667 | [-5.6342, -1.8333] | 1.0000 |
| Model B | -2.2000 | [-4.2675, -0.2333] | 0.9845 |
| Model C | -5.2667 | [-7.7000, -2.8667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
