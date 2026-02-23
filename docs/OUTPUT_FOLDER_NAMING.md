# Output folder naming (differentiators)

Output folders use the pattern **`outputs#_<differentiator>`** so you can tell at a glance which implementation produced the results.

| Folder | Differentiator | Meaning |
|--------|----------------|--------|
| outputs4 | *(none)* | Production/sweep root (default in config). |
| outputs5 | *(none)* | Outcome vs standings comparison (subpaths: playoff_spearman, etc.). |
| **outputs6_baseline** | baseline | Phase 1 outcome runs; **project baseline** for comparisons. |
| **outputs7_listmle_sweep** | listmle_sweep | ListMLE Optuna sweep. |
| **outputs8_spearman_surrogate** | spearman_surrogate | **Best** — Spearman-surrogate loss, 40 trials. |
| **outputs9_listmle_spearman** | listmle_spearman | ListMLE sweep, Spearman objective. |
| **outputs10_spearman_surrogate_standing_rank** | spearman_surrogate_standing_rank | Spearman surrogate + standing rank as input. |
| **outputs11_listmle_standing_rank** | listmle_standing_rank | ListMLE sweep + standing rank. |
| **outputs12_baseline_standing_rank** | baseline_standing_rank | Baseline-style config + standing rank (single run). |
| **outputs13_rmse_surrogate** | rmse_surrogate | Rank RMSE surrogate sweep. |
| **outputs14_map_run** | map_run | Future MAP branch runs. |
| **outputs15_rmse_surrogate_standing_rank** | rmse_surrogate_standing_rank | RMSE surrogate sweep + standing rank as input. |
| **outputs16_map_standing_rank** | map_standing_rank | MAP branch run + standing rank as input. |

Configs set `paths.outputs` to these folder names (e.g. `config/outputs8_sweep_spearman.yaml` → `outputs8_spearman_surrogate`). If you have existing data in old folder names (e.g. `outputs8`), rename the folder to the new name (e.g. `outputs8_spearman_surrogate`) to match.
