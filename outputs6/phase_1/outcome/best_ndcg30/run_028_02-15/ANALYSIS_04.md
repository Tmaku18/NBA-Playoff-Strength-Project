# Analysis 04 — Evaluation summary

**Run:** run_028_02-15
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4352
- spearman: 0.7477
- kendall_tau: 0.5448
- pearson: 0.7477
- precision_at_4: 0.0000
- precision_at_8: 0.7500
- mrr_top2: 0.1667
- mrr_top4: 0.1667
- ndcg_at_4 (Conference Finals (top 4)): 0.0496
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3556
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.4350
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4352
- ndcg_at_30: 0.4352
- rank_mae_pred_vs_playoff_outcome_rank: 5.2000
- rank_rmse_pred_vs_playoff_outcome_rank: 6.1482
- roc_auc_upset: 0.7195
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.7842
- kendall_tau_standings: -0.5862
- ndcg_at_4_standings: 0.3223
- ndcg_at_16_standings: 0.5856
- ndcg_at_30_standings: 0.5857
- rank_rmse_standings: 5.6862
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.7388, kendall_tau_pred_vs_playoff_outcome_rank=0.5448, ndcg_at_4_final_four=0.0496, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4352, brier_championship_odds=0.0312, ece_championship_odds=0.0000, champion_rank=6, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.200 | 6.148 | -2.067 | -1.698 |
| Model A | 6.800 | 8.764 | -3.667 | -4.314 |
| Model B | 5.333 | 6.880 | -2.200 | -2.430 |
| Model C | 8.400 | 10.149 | -5.267 | -5.699 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.631 | 0.696 | 0.467 | 6.067 |
| West (W) | 0.608 | 0.846 | 0.657 | 4.333 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.0667 | [-3.4333, -0.6333] | 0.9980 |
| Model A | -3.6667 | [-5.6000, -1.8333] | 1.0000 |
| Model B | -2.2000 | [-4.2675, -0.2333] | 0.9845 |
| Model C | -5.2667 | [-7.7000, -2.8667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
