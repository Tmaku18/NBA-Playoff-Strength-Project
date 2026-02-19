# Analysis 04 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.3940
- spearman: 0.7402
- kendall_tau: 0.5494
- pearson: 0.7402
- precision_at_4: 0.0000
- precision_at_8: 0.6250
- mrr_top2: 0.1111
- mrr_top4: 0.1250
- ndcg_at_4 (Conference Finals (top 4)): 0.0496
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3159
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.3160
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.3940
- ndcg_at_30: 0.3940
- rank_mae_pred_vs_playoff_outcome_rank: 5.4000
- rank_rmse_pred_vs_playoff_outcome_rank: 6.2397
- roc_auc_upset: 0.7511
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.7446
- kendall_tau_standings: -0.5540
- ndcg_at_4_standings: 0.3223
- ndcg_at_16_standings: 0.5478
- ndcg_at_30_standings: 0.5485
- rank_rmse_standings: 6.1860
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.7019, kendall_tau_pred_vs_playoff_outcome_rank=0.5218, ndcg_at_4_final_four=0.0496, ndcg_at_30_pred_vs_playoff_outcome_rank=0.3940, brier_championship_odds=0.0319, ece_championship_odds=0.0000, champion_rank=9, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.400 | 6.240 | -2.267 | -1.790 |
| Model A | 6.867 | 8.896 | -3.733 | -4.446 |
| Model B | 6.067 | 7.690 | -2.933 | -3.240 |
| Model C | 8.467 | 10.224 | -5.333 | -5.774 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.596 | 0.679 | 0.524 | 6.200 |
| West (W) | 0.569 | 0.807 | 0.619 | 4.600 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.2667 | [-3.7000, -0.5667] | 0.9980 |
| Model A | -3.7333 | [-5.7333, -1.8992] | 1.0000 |
| Model B | -2.9333 | [-5.1675, -0.8000] | 0.9965 |
| Model C | -5.3333 | [-7.7333, -2.9333] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
