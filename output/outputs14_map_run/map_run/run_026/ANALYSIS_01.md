# Analysis 01 — Evaluation summary

**Run:** run_026
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4152
- spearman: 0.2703
- kendall_tau: 0.1494
- pearson: 0.2703
- precision_at_4: 0.0000
- precision_at_8: 0.6250
- mrr_top2: 0.1667
- mrr_top4: 0.1667
- ndcg_at_4 (Conference Finals (top 4)): 0.0413
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3640
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.3640
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.3640
- ndcg_at_30: 0.4152
- rank_mae_pred_vs_playoff_outcome_rank: 8.2000
- rank_rmse_pred_vs_playoff_outcome_rank: 10.4563
- roc_auc_upset: 0.8517
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.4229
- kendall_tau_standings: -0.2644
- ndcg_at_4_standings: 0.0911
- ndcg_at_16_standings: 0.4733
- ndcg_at_30_standings: 0.4831
- rank_rmse_standings: 9.2987
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.3584, kendall_tau_pred_vs_playoff_outcome_rank=0.2414, ndcg_at_4_final_four=0.0413, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4152, brier_championship_odds=0.0312, ece_championship_odds=0.0000, champion_rank=6, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 8.200 | 10.456 | -5.067 | -6.007 |
| Model A | 12.600 | 14.560 | -9.467 | -10.111 |
| Model B | 8.067 | 10.159 | -4.933 | -5.709 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.456 | 0.154 | 0.124 | 8.600 |
| West (W) | 0.490 | 0.329 | 0.200 | 7.800 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -5.0667 | [-7.2000, -3.0333] | 1.0000 |
| Model A | -9.4667 | [-12.1008, -6.9000] | 1.0000 |
| Model B | -4.9333 | [-7.0667, -3.0325] | 1.0000 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
