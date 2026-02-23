# Analysis 01 — Evaluation summary

**Run:** run_025_02-17
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4381
- spearman: 0.7771
- kendall_tau: 0.5494
- pearson: 0.7771
- precision_at_4: 0.0000
- precision_at_8: 0.7500
- mrr_top2: 0.1667
- mrr_top4: 0.1667
- ndcg_at_4 (Conference Finals (top 4)): 0.0422
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3974
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.4380
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4381
- ndcg_at_30: 0.4381
- rank_mae_pred_vs_playoff_outcome_rank: 4.8000
- rank_rmse_pred_vs_playoff_outcome_rank: 5.7793
- roc_auc_upset: 0.7589
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.8438
- kendall_tau_standings: -0.6460
- ndcg_at_4_standings: 0.4277
- ndcg_at_16_standings: 0.6791
- ndcg_at_30_standings: 0.6791
- rank_rmse_standings: 4.8374
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.8020, kendall_tau_pred_vs_playoff_outcome_rank=0.5862, ndcg_at_4_final_four=0.0422, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4381, brier_championship_odds=0.0315, ece_championship_odds=0.0000, champion_rank=7, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 4.800 | 5.779 | -1.667 | -1.330 |
| Model A | 14.000 | 15.778 | -10.867 | -11.328 |
| Model B | 5.933 | 8.157 | -2.800 | -3.707 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.651 | 0.729 | 0.486 | 5.333 |
| West (W) | 0.604 | 0.793 | 0.600 | 4.267 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -1.6667 | [-2.9000, -0.5000] | 0.9960 |
| Model A | -10.8667 | [-13.6333, -8.0333] | 1.0000 |
| Model B | -2.8000 | [-5.2008, -0.6667] | 0.9945 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
