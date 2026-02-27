# Analysis 01 — Evaluation summary

**Run:** run_025
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.3934
- spearman: 0.2004
- kendall_tau: 0.1356
- pearson: 0.2004
- precision_at_4: 0.0000
- precision_at_8: 0.5000
- mrr_top2: 0.1667
- mrr_top4: 0.1667
- ndcg_at_4 (Conference Finals (top 4)): 0.0241
- ndcg_at_12 (Clinch Playoff (top 12)): 0.2654
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.2654
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.3406
- ndcg_at_30: 0.3934
- rank_mae_pred_vs_playoff_outcome_rank: 8.6000
- rank_rmse_pred_vs_playoff_outcome_rank: 10.9453
- roc_auc_upset: 0.7956
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.3406
- kendall_tau_standings: -0.2138
- ndcg_at_4_standings: 0.0164
- ndcg_at_16_standings: 0.4219
- ndcg_at_30_standings: 0.4328
- rank_rmse_standings: 9.9398
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.2699, kendall_tau_pred_vs_playoff_outcome_rank=0.1908, ndcg_at_4_final_four=0.0241, ndcg_at_30_pred_vs_playoff_outcome_rank=0.3934, brier_championship_odds=0.0312, ece_championship_odds=0.0000, champion_rank=6, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 8.600 | 10.945 | -5.467 | -6.496 |
| Model A | 12.467 | 14.686 | -9.333 | -10.236 |
| Model B | 8.000 | 10.208 | -4.867 | -5.758 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.435 | 0.150 | 0.124 | 9.333 |
| West (W) | 0.471 | 0.314 | 0.200 | 7.867 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -5.4667 | [-7.6667, -3.1333] | 1.0000 |
| Model A | -9.3333 | [-12.1342, -6.4658] | 1.0000 |
| Model B | -4.8667 | [-7.0333, -2.9325] | 1.0000 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
