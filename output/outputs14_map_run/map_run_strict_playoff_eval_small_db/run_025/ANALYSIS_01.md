# Analysis 01 — Evaluation summary

**Run:** run_025
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.3528
- spearman: 0.2529
- kendall_tau: 0.1586
- pearson: 0.2529
- precision_at_4: 0.0000
- precision_at_8: 0.5000
- mrr_top2: 0.0833
- mrr_top4: 0.0833
- ndcg_at_4 (Conference Finals (top 4)): 0.0360
- ndcg_at_12 (Clinch Playoff (top 12)): 0.2149
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.2980
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.3172
- ndcg_at_30: 0.3528
- rank_mae_pred_vs_playoff_outcome_rank: 8.2000
- rank_rmse_pred_vs_playoff_outcome_rank: 10.5799
- roc_auc_upset: 0.8869
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.4065
- kendall_tau_standings: -0.2644
- ndcg_at_4_standings: 0.0898
- ndcg_at_16_standings: 0.4047
- ndcg_at_30_standings: 0.4150
- rank_rmse_standings: 9.4304
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.3615, kendall_tau_pred_vs_playoff_outcome_rank=0.2506, ndcg_at_4_final_four=0.0360, ndcg_at_30_pred_vs_playoff_outcome_rank=0.3528, brier_championship_odds=0.0324, ece_championship_odds=0.0000, champion_rank=12, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 8.200 | 10.580 | -5.067 | -6.130 |
| Model A | 11.467 | 13.909 | -8.333 | -9.460 |
| Model B | 8.067 | 10.237 | -4.933 | -5.787 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.543 | 0.179 | 0.124 | 9.067 |
| West (W) | 0.412 | 0.336 | 0.219 | 7.333 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -5.0667 | [-7.3000, -2.8325] | 1.0000 |
| Model A | -8.3333 | [-11.1342, -5.4000] | 1.0000 |
| Model B | -4.9333 | [-7.0342, -2.9992] | 1.0000 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
