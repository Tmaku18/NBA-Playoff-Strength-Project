# Outputs retention notes (main)

Summary of the retention run and how output folders are tracked.

## Retention policy

- **Rule:** [.cursor/rules/outputs-retention-rmse.mdc](../.cursor/rules/outputs-retention-rmse.mdc)  
- **Policy:** For all `outputs*` **except outputs8**, each sweep batch keeps **top 3 combos by RMSE** + **1 worst combo by RMSE**; all other combos in that batch are deleted. Ranking uses **RMSE only** (not MAE).  
- **outputs8** is the canonical best (Spearman-surrogate) and is **never** modified by retention.

## Last retention run (results)

Run: `python -m scripts.retain_top3_worst1_rmse` (dry-run then execute).

| Output root | Sweep batch | Kept | Deleted |
|-------------|-------------|------|---------|
| outputs10 | standing_rank_spearman_40 | 4 | 0 |
| outputs11_listmle_standing_rank | listmle_standing_rank_40 | 4 | 0 |
| outputs13_rmse_surrogate | rmse_surrogate_40 | 4 | 0 |
| outputs2 | 20260203_021923 | 4 | 0 |
| outputs2 | optuna_3trial | 3 | 0 |
| outputs3 | baseline_ndcg_final_rank | 4 | 0 |
| outputs3 | baseline_ndcg_playoff_outcome | 4 | 0 |
| outputs3 | baseline_spearman_final_rank | 4 | 0 |
| outputs3 | baseline_spearman_playoff_outcome | 4 | 0 |
| outputs3 | phase1_spearman_final_rank | 4 | 0 |
| outputs3 | phase1_spearman_playoff_outcome | 4 | 0 |
| outputs3 | test_run024 | 1 | 0 |
| outputs4 | 20260212_054114 | 4 | 0 |
| outputs4 | wsl_playoff_spearman | 4 | 0 |
| outputs7 | 20260217_124115 | 4 | 0 |
| outputs7 | 20260217_235940 | 4 | 0 |
| outputs9 | outputs9_listmle_spearman | 4 | 0 |
| **Total** | 18 batches | **64** | **0** |

No batch had more than four combos, so nothing was deleted; all existing combos were within the “top 3 + worst 1” allowance.

## Gitignore: disposable outputs

All output roots **except outputs8** are treated as disposable for version control:

- **Files inside** those output folders are **gitignored** (sweep combos, run dirs, models, reports, etc. are not committed).
- **Folders** are kept in the repo via `.gitkeep` so future runs can write into the same paths; if a future run is better, you can re-add specific files or un-ignore patterns as needed.

**Canonical output (not disposable):** **outputs8** — Spearman-surrogate best run; MODEL.md, ANALYSIS*.md, and selected sweep/best-run paths remain tracked.

## Re-running retention

```bash
python -m scripts.retain_top3_worst1_rmse --dry-run
python -m scripts.retain_top3_worst1_rmse
```

To target one root: `python -m scripts.retain_top3_worst1_rmse --outputs outputs4`
