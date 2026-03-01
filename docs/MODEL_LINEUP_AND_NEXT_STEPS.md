# Model lineup and next steps

**Baseline vs best vs experiments.** This doc summarizes the canonical roles of each output set and what still needs to be tested.

**Per-folder docs:** Each output folder has a **`MODEL.md`** that describes the model that produced those results. All output roots live under **`output/`** with **model-based names** (e.g. **8_spearman_surrogate**, **6_baseline**). See [OUTPUT_FOLDER_NAMING.md](OUTPUT_FOLDER_NAMING.md).

---

## Roles

| Role | Output / config | Description |
|------|------------------|-------------|
| **Baseline** | **6_baseline** (phase_1 outcome) | Phase 1 outcome runs: ListMLE, playoff_outcome, rolling [15,30]; configs from 4_listmle. Reference for all comparisons. Path: `output/6_baseline`. |
| **Best** | **8_spearman_surrogate** (e.g. combo_0033 / combo_0038) | **Spearman-surrogate** loss, playoff_outcome, rolling [10,30]. Best Spearman / playoff_spearman / rank_mae / rank_rmse. Path: `output/8_spearman_surrogate`. See [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md). |
| **MAP run** | **14_map_run** (`config/outputs14_map_run.yaml`) | MAP run tested (mixed): stronger NDCG/top-end behavior, weaker Spearman/rank error vs 8_spearman_surrogate. Path: `output/14_map_run`. See [OUTPUTS14_MAP_ANALYSIS.md](OUTPUTS14_MAP_ANALYSIS.md), [OUTPUTS14_MAP_RUN.md](OUTPUTS14_MAP_RUN.md). |
| **RMSE surrogate** | **13_rmse_surrogate** | **Not recommended.** Tested (sweep + 80-epoch run); underperformed Spearman surrogate. Path: `output/13_rmse_surrogate`. See [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md), [OUTPUTS13_RMSE_SURROGATE_SWEEP.md](OUTPUTS13_RMSE_SURROGATE_SWEEP.md). |
| **RMSE surrogate + standing rank** | **15_rmse_surrogate_standing_rank** (`config/outputs15_sweep_rmse_surrogate_standing_rank.yaml`) | RMSE surrogate sweep with **standing rank as input**. Path: `output/15_rmse_surrogate_standing_rank`. Compare to 13_rmse_surrogate (no standing). See [OUTPUTS15_RMSE_SURROGATE_STANDING_RANK.md](OUTPUTS15_RMSE_SURROGATE_STANDING_RANK.md). |
| **Baseline-style + standing rank** | **12_baseline_standing_rank** (`config/outputs4_baseline_standing_rank.yaml`) | Same as baseline config (ListMLE, final_rank, [15,30]) but with **standing rank as input** (stat_dim 22). Single run. Path: `output/12_baseline_standing_rank`. Compare to **baseline (6_baseline)**. |
| **ListMLE + standing (sweep)** | **11_listmle_standing_rank** | ListMLE loss, playoff_outcome, standing rank; Optuna sweep. Path: `output/11_listmle_standing_rank`. Compare to **baseline (6_baseline)** and 10_spearman_surrogate_standing_rank. |
| **MAP + standing rank** | **16_map_standing_rank** (`config/outputs16_map_standing_rank.yaml`) | MAP branch run with **standing rank as input**. Path: `output/16_map_standing_rank`. Compare to 14_map_run (MAP, no standing). See [OUTPUTS16_MAP_STANDING_RANK.md](OUTPUTS16_MAP_STANDING_RANK.md). |

---

## Still to test

1. **MAP + standing rank** — run with `config/outputs16_map_standing_rank.yaml` → **output/16_map_standing_rank**; compare against 14_map_run (MAP without standing), baseline, and 8_spearman_surrogate best.
2. **RMSE surrogate** — *Done.* Not recommended; see [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md).
3. **Baseline-style + standing rank** — single pipeline run with `config/outputs4_baseline_standing_rank.yaml` → `output/12_baseline_standing_rank/baseline_standing_rank`; compare metrics to **baseline (6_baseline)**.

---

## Run: baseline-style + standing rank (12_baseline_standing_rank)

From project root:

```powershell
python -m scripts.run_pipeline_from_model_a --config config/outputs4_baseline_standing_rank.yaml --outputs output/12_baseline_standing_rank/baseline_standing_rank
```

WSL:

```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/outputs4_baseline_standing_rank.yaml --outputs output/12_baseline_standing_rank/baseline_standing_rank
```

Results under `output/12_baseline_standing_rank/baseline_standing_rank/` (e.g. `run_XXX/eval_report.json`). Compare Spearman, playoff_spearman, rank_mae, rank_rmse, NDCG to **baseline (6_baseline)** on the same eval (test seasons, eos_final_rank if available).

---

## Output folders and MODEL.md

All live under **`output/`**. See [OUTPUT_FOLDER_NAMING.md](OUTPUT_FOLDER_NAMING.md) for the full mapping.

| Folder | MODEL.md describes |
|--------|---------------------|
| output/0_outputs/ | Legacy/early runs. |
| output/2_listmle/ | Baseline full-pipeline runs (020/021/022). |
| output/3_listmle/ | Baseline and Phase 1 sweeps. |
| output/4_listmle/ | Production/sweep root; default in config. Phase 3 combo 18 etc. |
| output/5_listmle/ | Outcome vs standings comparison. |
| output/6_baseline/ | **Baseline** (phase_1 outcome runs; reference for comparisons). |
| output/7_listmle/ | ListMLE Optuna sweep (best combo 17). |
| output/8_spearman_surrogate/ | **Best** (Spearman surrogate; combo_0033, etc.). |
| output/9_listmle/ | ListMLE sweep (Spearman objective). |
| output/10_spearman_surrogate_standing_rank/ | Spearman surrogate + standing rank. |
| output/11_listmle_standing_rank/ | ListMLE sweep + standing rank; compare to 6_baseline. |
| output/12_baseline_standing_rank/ | Baseline-style + standing rank (single run); compare to 6_baseline. |
| output/13_rmse_surrogate/ | **RMSE surrogate** sweep (rank_rmse_surrogate). |
| output/14_map_run/ | **MAP run** (tested; mixed). See [OUTPUTS14_MAP_ANALYSIS.md](OUTPUTS14_MAP_ANALYSIS.md). |
| output/15_rmse_surrogate_standing_rank/ | RMSE surrogate + standing rank. |
| output/16_map_standing_rank/ | MAP + standing rank; compare to 14_map_run. |
| output/team_stats_listmle/ | Team-stats/standing + ListMLE (branch feature/team-stats-listmle). |
| output/team_stats_spearman_surrogate/ | Team-stats + Spearman surrogate (branch feature/team-stats-spearman-surrogate). |
