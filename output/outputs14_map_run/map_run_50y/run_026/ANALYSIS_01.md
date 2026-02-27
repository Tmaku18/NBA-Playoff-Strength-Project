# Analysis 01 — Evaluation summary

**Run:** run_026
**EOS source:** standings

## Test metrics (ensemble)

*NDCG cutoffs: ndcg_at_4=Conference Finals (top 4); ndcg_at_12=Clinch Playoff (top 12); ndcg_at_16=One Play-In Tournament (top 16); ndcg_at_20=Qualify for Playoffs (top 20); ndcg_at_30=full order.*

- ndcg: 0.9542
- spearman: 0.9791
- kendall_tau: 0.8943
- pearson: 0.9791
- precision_at_4: 0.7500
- precision_at_8: 1.0000
- mrr_top2: 1.0000
- mrr_top4: 1.0000
- ndcg_at_4 (Conference Finals (top 4)): 0.9293
- ndcg_at_12 (Clinch Playoff (top 12)): 0.9542
- ndcg_at_16 (One Play-In Tournament (top 16)): 0.9542
- ndcg_at_20 (Qualify for Playoffs (top 20)): 0.9542
- ndcg_at_30: 0.9542
- rank_mae_pred_vs_playoff_outcome_rank: 1.3333
- rank_rmse_pred_vs_playoff_outcome_rank: 1.7701
- roc_auc_upset: 0.5215

See `eval_report.json` and `eval_report_<season>.json` for full report (incl. per-model MAE/RMSE, significance, `confusion_matrices`, and `confusion_matrices_ranking_top16`). Plot: `confusion_matrix_ranking_top16.png` (top 16 in order).
