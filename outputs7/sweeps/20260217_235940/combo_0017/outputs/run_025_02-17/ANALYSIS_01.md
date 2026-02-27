# Analysis 01 — Evaluation summary

**Run:** run_025_02-17
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4657
- spearman: 0.7646
- kendall_tau: 0.5678
- pearson: 0.7646
- precision_at_4: 0.0000
- precision_at_8: 0.6250
- mrr_top2: 0.2000
- mrr_top4: 0.2000
- ndcg_at_4 (Conference Finals (top 4)): 0.0621
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3818
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.4649
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4657
- ndcg_at_30: 0.4657
- rank_mae_pred_vs_playoff_outcome_rank: 5.2667
- rank_rmse_pred_vs_playoff_outcome_rank: 5.9386
- roc_auc_upset: 0.7867
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.7677
- kendall_tau_standings: -0.5724
- ndcg_at_4_standings: 0.2709
- ndcg_at_16_standings: 0.5323
- ndcg_at_30_standings: 0.5518
- rank_rmse_standings: 5.8992
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.7362, kendall_tau_pred_vs_playoff_outcome_rank=0.5494, ndcg_at_4_final_four=0.0621, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4657, brier_championship_odds=0.0309, ece_championship_odds=0.0000, champion_rank=5, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.267 | 5.939 | -2.133 | -1.489 |
| Model A | 7.000 | 8.843 | -3.867 | -4.393 |
| Model B | 6.067 | 7.434 | -2.933 | -2.984 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.631 | 0.775 | 0.600 | 5.333 |
| West (W) | 0.661 | 0.757 | 0.600 | 5.200 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.1333 | [-3.4000, -0.6658] | 0.9975 |
| Model A | -3.8667 | [-5.4667, -2.3667] | 1.0000 |
| Model B | -2.9333 | [-4.8333, -0.9333] | 0.9980 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
