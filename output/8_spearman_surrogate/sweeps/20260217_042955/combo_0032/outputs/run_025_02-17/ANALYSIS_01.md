# Analysis 01 — Evaluation summary

**Run:** run_025_02-17
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.5219
- spearman: 0.7370
- kendall_tau: 0.5218
- pearson: 0.7370
- precision_at_4: 0.2500
- precision_at_8: 0.6250
- mrr_top2: 0.3333
- mrr_top4: 0.3333
- ndcg_at_4 (Conference Finals (top 4)): 0.3490
- ndcg_at_12 (Clinch Playoff (top 12)): 0.5211
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.5218
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.5218
- ndcg_at_30: 0.5219
- rank_mae_pred_vs_playoff_outcome_rank: 5.0667
- rank_rmse_pred_vs_playoff_outcome_rank: 6.2769
- roc_auc_upset: 0.6578
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.8024
- kendall_tau_standings: -0.6000
- ndcg_at_4_standings: 0.4106
- ndcg_at_16_standings: 0.5578
- ndcg_at_30_standings: 0.5578
- rank_rmse_standings: 5.4406
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.7513, kendall_tau_pred_vs_playoff_outcome_rank=0.5310, ndcg_at_4_final_four=0.3490, ndcg_at_30_pred_vs_playoff_outcome_rank=0.5219, brier_championship_odds=0.0304, ece_championship_odds=0.0000, champion_rank=3, champion_in_top_4=1.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.067 | 6.277 | -1.933 | -1.827 |
| Model A | 13.600 | 15.457 | -10.467 | -11.008 |
| Model B | 5.800 | 7.581 | -2.667 | -3.131 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.625 | 0.807 | 0.600 | 4.533 |
| West (W) | 0.613 | 0.664 | 0.448 | 5.600 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -1.9333 | [-3.2000, -0.6333] | 0.9985 |
| Model A | -10.4667 | [-13.3667, -7.5667] | 1.0000 |
| Model B | -2.6667 | [-4.9333, -0.5325] | 0.9935 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
