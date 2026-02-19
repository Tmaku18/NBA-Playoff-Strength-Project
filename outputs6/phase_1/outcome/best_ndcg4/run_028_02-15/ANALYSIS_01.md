# Analysis 01 — Evaluation summary

**Run:** run_028
**EOS source:** eos_final_rank

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.2976
- spearman: -0.7264
- mrr_top2: 0.0833
- mrr_top4: 0.0833
- ndcg_at_4 (Conference Finals (top 4)): 0.0000
- ndcg_at_12 (Clinch Playoff (top 12)): 0.0878
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.0886
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.1256
- ndcg_at_30: 0.2976
- rank_mae_pred_vs_playoff_outcome_rank: 14.0667
- rank_rmse_pred_vs_playoff_outcome_rank: 16.0831
- roc_auc_upset: 0.9556
- rank_mae_wl_record_standings_vs_playoff_outcome_rank: 3.1333
- rank_rmse_wl_record_standings_vs_playoff_outcome_rank: 4.4497
- spearman_standings: 0.7597
- ndcg_at_4_standings: 0.0000
- ndcg_at_16_standings: 0.0211
- ndcg_at_30_standings: 0.2767
- rank_rmse_standings: 16.2378
- playoff_metrics: spearman_pred_vs_playoff_outcome_rank=-0.7175, ndcg_at_4_final_four=0.0000, ndcg_at_30_pred_vs_playoff_outcome_rank=0.2976, brier_championship_odds=0.0338, rank_mae_pred_vs_playoff_outcome_rank=14.0667, rank_rmse_pred_vs_playoff_outcome_rank=16.0831, rank_mae_wl_record_standings_vs_playoff_outcome_rank=3.1333, rank_rmse_wl_record_standings_vs_playoff_outcome_rank=4.4497

## Model vs regular-season standings (same outcome ranks)

All metrics compare predicted/standings rank to the **same** final outcome rank (EOS_global_rank).

| Source | MAE vs outcome | RMSE vs outcome | Δ MAE vs standings | Δ RMSE vs standings |
|--------|----------------|-----------------|--------------------|---------------------|
| W/L standings (baseline) | 3.133 | 4.450 | — | — |
| Ensemble | 14.067 | 16.083 | -10.933 | -11.633 |
| Model A | 7.000 | 8.752 | -3.867 | -4.302 |
| Model B | 6.067 | 7.690 | -2.933 | -3.240 |
| Model C | 8.467 | 10.224 | -5.333 | -5.774 |

### Statistical significance (vs standings)

Paired bootstrap over teams (resample with replacement; mean MAE improvement per team). H0: no improvement; positive = model better.

| Model | Mean MAE improvement | 95% CI | p-value |
|-------|----------------------|--------|--------|
| Ensemble | -10.9333 | [-13.7667, -8.2000] | 1.0000 |
| Model A | -3.8667 | [-5.7008, -2.0992] | 1.0000 |
| Model B | -2.9333 | [-5.1675, -0.8000] | 0.9965 |
| Model C | -5.3333 | [-7.7333, -2.9333] | 1.0000 |


See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE and significance).
