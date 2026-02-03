# Analysis 02 — Evaluation summary

**Run:** run_022
**EOS source:** eos_final_rank

## Test metrics (ensemble)

- ndcg: 0.4820
- spearman: 0.4305
- mrr_top2: 0.5000
- mrr_top4: 0.5000
- roc_auc_upset: 0.7277
- rank_mae_pred_vs_playoff: 7.5333
- rank_rmse_pred_vs_playoff: 9.2376
- rank_mae_standings_vs_playoff: 3.1333
- rank_rmse_standings_vs_playoff: 4.4497
- playoff_metrics: spearman_pred_vs_playoff_rank=0.4607, ndcg_at_4_final_four=0.4645, brier_championship_odds=0.0322, rank_mae_pred_vs_playoff=7.5333, rank_rmse_pred_vs_playoff=9.2376, rank_mae_standings_vs_playoff=3.1333, rank_rmse_standings_vs_playoff=4.4497

See `eval_report.json` and `eval_report_<season>.json` for full report.
