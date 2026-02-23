# Analysis 01 — Evaluation summary

**Run:** run_025_02-18
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4304
- spearman: 0.7370
- kendall_tau: 0.5310
- pearson: 0.7370
- precision_at_4: 0.0000
- precision_at_8: 0.6250
- mrr_top2: 0.1667
- mrr_top4: 0.1667
- ndcg_at_4 (Conference Finals (top 4)): 0.0544
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3295
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.4302
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4304
- ndcg_at_30: 0.4304
- rank_mae_pred_vs_playoff_outcome_rank: 5.4667
- rank_rmse_pred_vs_playoff_outcome_rank: 6.2769
- roc_auc_upset: 0.8089
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.8100
- kendall_tau_standings: -0.6276
- ndcg_at_4_standings: 0.3249
- ndcg_at_16_standings: 0.5869
- ndcg_at_30_standings: 0.5869
- rank_rmse_standings: 5.3354
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.7424, kendall_tau_pred_vs_playoff_outcome_rank=0.5586, ndcg_at_4_final_four=0.0544, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4304, brier_championship_odds=0.0312, ece_championship_odds=0.0000, champion_rank=6, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.467 | 6.277 | -2.333 | -1.827 |
| Model A | 7.200 | 9.118 | -4.067 | -4.668 |
| Model B | 5.533 | 7.216 | -2.400 | -2.766 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.598 | 0.732 | 0.543 | 6.133 |
| West (W) | 0.625 | 0.768 | 0.581 | 4.800 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.3333 | [-3.5667, -1.1333] | 0.9995 |
| Model A | -4.0667 | [-5.9000, -2.5667] | 1.0000 |
| Model B | -2.4000 | [-4.6000, -0.3000] | 0.9865 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
