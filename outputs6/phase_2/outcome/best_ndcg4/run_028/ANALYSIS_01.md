# Analysis 01 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.3940
- spearman: 0.7402
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
- ndcg_at_4_standings: 0.3223
- ndcg_at_16_standings: 0.5478
- ndcg_at_30_standings: 0.5485
- rank_rmse_standings: 6.1860
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.7019, ndcg_at_4_final_four=0.0496, ndcg_at_30_pred_vs_playoff_outcome_rank=0.3940, brier_championship_odds=0.0319, rank_mae_pred_vs_playoff_outcome_rank=5.4000, rank_rmse_pred_vs_playoff_outcome_rank=6.2397, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

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

Within-conference NDCG and Spearman (relevance = EOS-derived rank 1=best in conf). Full per-model MAE/RMSE in `eval_report.json` → `test_metrics_by_conference`.

| Conference | NDCG | Spearman | Ensemble MAE vs outcome |
|------------|------|----------|--------------------------|
| East (E) | 0.596 | 0.679 | 6.200 |
| West (W) | 0.569 | 0.807 | 4.600 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.2667 | [-3.7000, -0.5667] | 0.9980 |
| Model A | -3.7333 | [-5.7333, -1.8992] | 1.0000 |
| Model B | -2.9333 | [-5.1675, -0.8000] | 0.9965 |
| Model C | -5.3333 | [-7.7333, -2.9333] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
