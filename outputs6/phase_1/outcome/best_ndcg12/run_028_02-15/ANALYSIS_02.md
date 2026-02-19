# Analysis 02 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.4002
- spearman: 0.7130
- mrr_top2: 0.1250
- mrr_top4: 0.1250
- ndcg_at_4 (Conference Finals (top 4)): 0.0496
- ndcg_at_12 (Clinch Playoff (top 12)): 0.3014
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.3221
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.4002
- ndcg_at_30: 0.4002
- rank_mae_pred_vs_playoff_outcome_rank: 5.6667
- rank_rmse_pred_vs_playoff_outcome_rank: 6.5574
- roc_auc_upset: 0.8356
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: -0.7486
- ndcg_at_4_standings: 0.3223
- ndcg_at_16_standings: 0.5575
- ndcg_at_30_standings: 0.5582
- rank_rmse_standings: 6.1373
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=0.6854, ndcg_at_4_final_four=0.0496, ndcg_at_30_pred_vs_playoff_outcome_rank=0.4002, brier_championship_odds=0.0317, rank_mae_pred_vs_playoff_outcome_rank=5.6667, rank_rmse_pred_vs_playoff_outcome_rank=6.5574, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 5.667 | 6.557 | -2.533 | -2.108 |
| Model A | 7.200 | 8.993 | -4.067 | -4.543 |
| Model B | 6.067 | 7.690 | -2.933 | -3.240 |
| Model C | 8.467 | 10.224 | -5.333 | -5.774 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -2.5333 | [-3.9333, -0.9667] | 0.9995 |
| Model A | -4.0667 | [-5.9667, -2.2333] | 1.0000 |
| Model B | -2.9333 | [-5.1675, -0.8000] | 0.9965 |
| Model C | -5.3333 | [-7.7333, -2.9333] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
