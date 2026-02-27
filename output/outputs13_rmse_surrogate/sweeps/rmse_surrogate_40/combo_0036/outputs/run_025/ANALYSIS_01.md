# Analysis 01 — Evaluation summary

**Run:** run_025
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.3108
- spearman: -0.3001
- kendall_tau: -0.2000
- pearson: -0.3001
- precision_at_4: 0.0000
- precision_at_8: 0.1250
- mrr_top2: 0.1250
- mrr_top4: 0.1250
- ndcg_at_4 (Conference Finals (top 4)): 0.0006
- ndcg_at_12 (Clinch Playoff (top 12)): 0.1039
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.1466
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.1466
- ndcg_at_30: 0.3108
- rank_mae_pred_vs_playoff_outcome_rank: 11.6000
- rank_rmse_pred_vs_playoff_outcome_rank: 13.9571
- roc_auc_upset: 0.8869
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: 0.3295
- kendall_tau_standings: 0.2322
- ndcg_at_4_standings: 0.0027
- ndcg_at_16_standings: 0.0411
- ndcg_at_30_standings: 0.2872
- rank_rmse_standings: 14.1138
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=-0.3277, kendall_tau_pred_vs_playoff_outcome_rank=-0.2276, ndcg_at_4_final_four=0.0006, ndcg_at_30_pred_vs_playoff_outcome_rank=0.3108, brier_championship_odds=0.0341, ece_championship_odds=0.0000, champion_rank=30, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 11.600 | 13.957 | -8.467 | -9.507 |
| Model A | 11.533 | 13.909 | -8.400 | -9.460 |
| Model B | 7.933 | 10.334 | -4.800 | -5.885 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.378 | -0.311 | -0.181 | 11.867 |
| West (W) | 0.035 | -0.443 | -0.276 | 11.333 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -8.4667 | [-11.3667, -5.4667] | 1.0000 |
| Model A | -8.4000 | [-11.5000, -5.3000] | 1.0000 |
| Model B | -4.8000 | [-7.1000, -2.7658] | 1.0000 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
