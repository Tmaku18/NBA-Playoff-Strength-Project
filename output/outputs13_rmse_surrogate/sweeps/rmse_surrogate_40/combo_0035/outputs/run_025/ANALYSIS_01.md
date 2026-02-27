# Analysis 01 — Evaluation summary

**Run:** run_025
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.3224
- spearman: -0.3068
- kendall_tau: -0.1862
- pearson: -0.3068
- precision_at_4: 0.0000
- precision_at_8: 0.1250
- mrr_top2: 0.1429
- mrr_top4: 0.1429
- ndcg_at_4 (Conference Finals (top 4)): 0.0000
- ndcg_at_12 (Clinch Playoff (top 12)): 0.1522
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.1731
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.1831
- ndcg_at_30: 0.3224
- rank_mae_pred_vs_playoff_outcome_rank: 12.4667
- rank_rmse_pred_vs_playoff_outcome_rank: 13.9929
- roc_auc_upset: 0.8711
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: 0.4990
- kendall_tau_standings: 0.3471
- ndcg_at_4_standings: 0.0000
- ndcg_at_16_standings: 0.0122
- ndcg_at_30_standings: 0.2778
- rank_rmse_standings: 14.9867
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=-0.4189, kendall_tau_pred_vs_playoff_outcome_rank=-0.2690, ndcg_at_4_final_four=0.0000, ndcg_at_30_pred_vs_playoff_outcome_rank=0.3224, brier_championship_odds=0.0341, ece_championship_odds=0.0000, champion_rank=30, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 12.467 | 13.993 | -9.333 | -9.543 |
| Model A | 12.533 | 14.055 | -9.400 | -9.605 |
| Model B | 7.933 | 10.334 | -4.800 | -5.885 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.377 | -0.171 | -0.086 | 12.333 |
| West (W) | 0.180 | -0.457 | -0.295 | 12.600 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -9.3333 | [-12.1000, -6.6333] | 1.0000 |
| Model A | -9.4000 | [-12.1667, -6.6325] | 1.0000 |
| Model B | -4.8000 | [-7.1000, -2.7658] | 1.0000 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
