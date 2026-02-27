# Outputs retention notes (main)

Summary of the retention run and how output folders are tracked.

**Layout:** All output roots live under **`output/`** with model-based names: `0_outputs`, `2_listmle`, `4_listmle`, `7_listmle`, `8_spearman_surrogate` (canonical best), `8_spearman_surrogate_sweep`, `9_listmle`, `10_spearman_surrogate_standing_rank`, `11_listmle`, `11_listmle_standing_rank`, `12_baseline_standing_rank`, `13_rmse_surrogate`, `14_map_run`, etc.

## Retention policy

- **Rule:** [.cursor/rules/outputs-retention-rmse.mdc](../.cursor/rules/outputs-retention-rmse.mdc)  
- **Policy:** For all output roots **except 8_spearman_surrogate**, each sweep batch keeps **top 3 combos by RMSE** + **1 worst combo by RMSE**; all other combos in that batch are deleted. Ranking uses **RMSE only** (not MAE).  
- **8_spearman_surrogate** is the canonical best (Spearman-surrogate) and is **never** modified by retention.

## Last retention run (results)

Run: `python -m scripts.retain_top3_worst1_rmse` (dry-run then execute).

| Output root | Sweep batch | Kept | Deleted |
|-------------|-------------|------|---------|
| 10_spearman_surrogate_standing_rank | standing_rank_spearman_40 | 4 | 0 |
| 11_listmle_standing_rank | listmle_standing_rank_40 | 4 | 0 |
| 13_rmse_surrogate | rmse_surrogate_40 | 4 | 0 |
| 2_listmle | 20260203_021923 | 4 | 0 |
| 2_listmle | optuna_3trial | 3 | 0 |
| 3_listmle | baseline_ndcg_final_rank | 4 | 0 |
| 3_listmle | baseline_ndcg_playoff_outcome | 4 | 0 |
| 3_listmle | baseline_spearman_final_rank | 4 | 0 |
| 3_listmle | baseline_spearman_playoff_outcome | 4 | 0 |
| 3_listmle | phase1_spearman_final_rank | 4 | 0 |
| 3_listmle | phase1_spearman_playoff_outcome | 4 | 0 |
| 3_listmle | test_run024 | 1 | 0 |
| 4_listmle | 20260212_054114 | 4 | 0 |
| 4_listmle | wsl_playoff_spearman | 4 | 0 |
| 7_listmle | 20260217_124115 | 4 | 0 |
| 7_listmle | 20260217_235940 | 4 | 0 |
| 9_listmle | outputs9_listmle_spearman | 4 | 0 |
| **Total** | 18 batches | **64** | **0** |

No batch had more than four combos, so nothing was deleted; all existing combos were within the “top 3 + worst 1” allowance.

## Gitignore: disposable outputs

All output roots under **`output/`** except **output/8_spearman_surrogate** are treated as disposable for version control:

- **Files inside** those output folders are **gitignored** (sweep combos, run dirs, models, reports, etc. are not committed).
- **Folders** are kept in the repo via `.gitkeep` so future runs can write into the same paths; if a future run is better, you can re-add specific files or un-ignore patterns as needed.

**Canonical output (not disposable):** **8_spearman_surrogate** — Spearman-surrogate best run; MODEL.md, ANALYSIS*.md, and selected sweep/best-run paths remain tracked.

## Re-running retention

```bash
python -m scripts.retain_top3_worst1_rmse --dry-run
python -m scripts.retain_top3_worst1_rmse
```

To target one root: `python -m scripts.retain_top3_worst1_rmse --outputs output/4_listmle`
