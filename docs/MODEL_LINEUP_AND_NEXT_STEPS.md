# Model lineup and next steps

**Baseline vs best vs experiments.** This doc summarizes the canonical roles of each output set and what still needs to be tested.

**Per-folder docs:** Each output folder has a **`MODEL.md`** that describes (1) the model that produced those results and (2) the **difference from the previous output** folder. Folder names use differentiators (e.g. **outputs8_spearman_surrogate**). See [OUTPUT_FOLDER_NAMING.md](OUTPUT_FOLDER_NAMING.md). Use them to quickly see what each run/sweep is and how it evolved.

---

## Roles

| Role | Output / config | Description |
|------|------------------|-------------|
| **Baseline** | **outputs6_baseline** (phase_1 outcome) | Phase 1 outcome runs: ListMLE, playoff_outcome, rolling [15,30]; configs from outputs4. Reference for all comparisons. |
| **Best** | **outputs8_spearman_surrogate** (e.g. combo_0033 / combo_0038) | **Spearman-surrogate** loss, playoff_outcome, rolling [10,30]. Best Spearman / playoff_spearman / rank_mae / rank_rmse. See [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md). |
| **MAP run** | **outputs14_map_run** (`config/outputs14_map_run.yaml`) | MAP run tested (mixed): stronger NDCG/top-end behavior, weaker Spearman/rank error vs outputs8. Compare to baseline and best. See [OUTPUTS14_MAP_ANALYSIS.md](OUTPUTS14_MAP_ANALYSIS.md), [OUTPUTS14_MAP_RUN.md](OUTPUTS14_MAP_RUN.md). |
| **RMSE surrogate** | **outputs13_rmse_surrogate** | **Not recommended.** Tested (sweep + 80-epoch run); underperformed Spearman surrogate. See [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md), [OUTPUTS13_RMSE_SURROGATE_SWEEP.md](OUTPUTS13_RMSE_SURROGATE_SWEEP.md). |
| **RMSE surrogate + standing rank** | **outputs15_rmse_surrogate_standing_rank** (`config/outputs15_sweep_rmse_surrogate_standing_rank.yaml`) | RMSE surrogate sweep with **standing rank as input**. Compare to outputs13_rmse_surrogate (no standing). See [OUTPUTS15_RMSE_SURROGATE_STANDING_RANK.md](OUTPUTS15_RMSE_SURROGATE_STANDING_RANK.md). |
| **Baseline-style + standing rank** | **outputs12_baseline_standing_rank** (`config/outputs4_baseline_standing_rank.yaml`) | Same as baseline config (ListMLE, final_rank, [15,30]) but with **standing rank as input** (stat_dim 22). Single run. Compare to **baseline (outputs6_baseline)**. |
| **ListMLE + standing (sweep)** | **outputs11_listmle_standing_rank** | ListMLE loss, playoff_outcome, standing rank; Optuna sweep. Compare to **baseline (outputs6_baseline)** and outputs10_spearman_surrogate_standing_rank. |
| **MAP + standing rank** | **outputs16_map_standing_rank** (`config/outputs16_map_standing_rank.yaml`) | MAP branch run with **standing rank as input**. Compare to outputs14_map_run (MAP, no standing). See [OUTPUTS16_MAP_STANDING_RANK.md](OUTPUTS16_MAP_STANDING_RANK.md). |

---

## Still to test

1. **MAP + standing rank** — run with `config/outputs16_map_standing_rank.yaml` → **outputs16_map_standing_rank**; compare against outputs14_map_run (MAP without standing), baseline, and outputs8 best.
2. **RMSE surrogate** — *Done.* Not recommended; see [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md).
3. **Baseline-style + standing rank** — single pipeline run with `config/outputs4_baseline_standing_rank.yaml` → `outputs12_baseline_standing_rank/baseline_standing_rank`; compare metrics to **baseline (outputs6_baseline)**.

---

## Run: baseline-style + standing rank (outputs12)

From project root:

```powershell
python -m scripts.run_pipeline_from_model_a --config config/outputs4_baseline_standing_rank.yaml --outputs outputs12_baseline_standing_rank/baseline_standing_rank
```

WSL:

```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/outputs4_baseline_standing_rank.yaml --outputs outputs12_baseline_standing_rank/baseline_standing_rank
```

Results under `outputs12_baseline_standing_rank/baseline_standing_rank/` (e.g. `outputs/run_XXX/eval_report.json`). Compare Spearman, playoff_spearman, rank_mae, rank_rmse, NDCG to **baseline (outputs6_baseline)** on the same eval (test seasons, eos_final_rank if available).

---

## Output folders and MODEL.md

| Folder    | MODEL.md describes |
|-----------|--------------------|
| outputs/  | Legacy/early runs; no prior output in lineage. |
| outputs2/ | Baseline full-pipeline runs (020/021/022); diff from outputs. |
| outputs3/ | Baseline and Phase 1 sweeps; diff from outputs2. |
| outputs4/ | Production/sweep root; config source for baseline. Phase 3 combo 18 etc.; diff from outputs3. |
| outputs5/ | Outcome vs standings comparison; diff from outputs4. |
| outputs6_baseline/ | **Baseline** (phase_1 outcome runs; reference for comparisons); diff from outputs5. |
| outputs7_listmle_sweep/ | ListMLE Optuna sweep; diff from outputs6_baseline. |
| outputs8_spearman_surrogate/ | **Best** (Spearman surrogate); diff from outputs7_listmle_sweep. |
| outputs9_listmle_spearman/ | ListMLE sweep (spearman objective); diff from outputs8_spearman_surrogate. |
| outputs10_spearman_surrogate_standing_rank/ | Spearman surrogate + standing rank; diff from outputs9_listmle_spearman. |
| outputs11_listmle_standing_rank/ | ListMLE sweep + standing rank; compare to baseline (outputs6_baseline). |
| outputs12_baseline_standing_rank/ | Baseline-style + standing rank (single run); compare to outputs6_baseline. |
| outputs13_rmse_surrogate/ | **RMSE surrogate** sweep (rank_rmse_surrogate); diff from outputs10_spearman_surrogate_standing_rank. |
| outputs14_map_run/ | **MAP run** (tested; mixed). See [OUTPUTS14_MAP_ANALYSIS.md](OUTPUTS14_MAP_ANALYSIS.md). |
| outputs15_rmse_surrogate_standing_rank/ | **RMSE surrogate + standing rank**; diff from outputs13_rmse_surrogate. |
| outputs16_map_standing_rank/ | **MAP + standing rank** (future); compare to outputs14_map_run. |
