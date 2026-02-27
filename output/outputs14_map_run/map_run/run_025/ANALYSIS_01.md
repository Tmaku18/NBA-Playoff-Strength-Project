# Analysis 01 — Evaluation summary

**Run:** run_025
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.5957
- spearman: 0.4358
- kendall_tau: 0.3057
- pearson: 0.4358
- precision_at_4: 0.2500
- precision_at_8: 0.3750
- mrr_top2: 0.5000
- mrr_top4: 0.5000
- ndcg_at_4 (Conference Finals (top 4)): 0.4644
- ndcg_at_12 (Clinch Playoff (top 12)): 0.4822
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.5248
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.5266
- ndcg_at_30: 0.5957
- rank_mae_pred_vs_playoff_outcome_rank: 7.4000
- rank_rmse_pred_vs_playoff_outcome_rank: 9.1942
- roc_auc_upset: 0.7946
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.4647
- kendall_tau_standings: -0.3287
- ndcg_at_4_standings: 0.4433
- ndcg_at_16_standings: 0.4971
- ndcg_at_30_standings: 0.5881
- rank_rmse_standings: 8.9554
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.4607, kendall_tau_pred_vs_playoff_outcome_rank=0.3333, ndcg_at_4_final_four=0.4644, ndcg_at_30_pred_vs_playoff_outcome_rank=0.5957, brier_championship_odds=0.0301, ece_championship_odds=0.0000, champion_rank=2, champion_in_top_4=1.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 7.400 | 9.194 | -4.267 | -4.744 |
| Model A | 7.400 | 9.194 | -4.267 | -4.744 |
| Model B | 8.067 | 10.159 | -4.933 | -5.709 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.254 | 0.221 | 0.162 | 9.267 |
| West (W) | 0.750 | 0.646 | 0.467 | 5.533 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -4.2667 | [-6.1333, -2.6667] | 1.0000 |
| Model A | -4.2667 | [-6.0667, -2.5000] | 1.0000 |
| Model B | -4.9333 | [-7.0667, -3.0325] | 1.0000 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
