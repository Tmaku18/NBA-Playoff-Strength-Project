# Analysis 01 — Evaluation summary

**Run:** run_025_02-17
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4400
- spearman: 0.7664
- kendall_tau: 0.5540
- pearson: 0.7664
- precision_at_4: 0.0000
- precision_at_8: 0.8750
- mrr_top2: 0.1667
- mrr_top4: 0.1667
- ndcg_at_4 (Conference Finals (top 4)): 0.0403
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3983
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.4399
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4400
- ndcg_at_30: 0.4400
- rank_mae_pred_vs_playoff_outcome_rank: 4.8000
- rank_rmse_pred_vs_playoff_outcome_rank: 5.9161
- roc_auc_upset: 0.8080
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.8679
- kendall_tau_standings: -0.6874
- ndcg_at_4_standings: 0.3044
- ndcg_at_16_standings: 0.5527
- ndcg_at_30_standings: 0.5528
- rank_rmse_standings: 4.4497
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.8536, kendall_tau_pred_vs_playoff_outcome_rank=0.6460, ndcg_at_4_final_four=0.0403, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4400, brier_championship_odds=0.0315, ece_championship_odds=0.0000, champion_rank=7, champion_in_top_4=0.0000

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 4.800 | 5.916 | -1.667 | -1.466 |
| Model A | 13.733 | 15.319 | -10.600 | -10.869 |
| Model B | 5.667 | 7.465 | -2.533 | -3.016 |
| Model C | 15.500 | 17.753 | -12.367 | -13.303 |

## East vs West (conference)

Within-conference NDCG, Spearman, and Kendall τ (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Kendall τ | Ensemble MAE vs outcome |
|------------|------|----------|------------|--------------------------|
| East (E) | 0.650 | 0.704 | 0.505 | 5.400 |
| West (W) | 0.617 | 0.811 | 0.619 | 4.200 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -1.6667 | [-2.8000, -0.6000] | 0.9985 |
| Model A | -10.6000 | [-13.5333, -7.5000] | 1.0000 |
| Model B | -2.5333 | [-4.7667, -0.4000] | 0.9910 |
| Model C | -12.3667 | [-15.3333, -9.2667] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
