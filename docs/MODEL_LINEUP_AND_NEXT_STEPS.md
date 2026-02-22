# Model lineup and next steps

**Baseline vs best vs experiments.** This doc summarizes the canonical roles of each output set and what still needs to be tested.

**Per-folder docs:** Each output folder has a **`MODEL.md`** that describes (1) the model that produced those results and (2) the **difference from the previous output** folder. Folder names use differentiators (e.g. **outputs8_spearman_surrogate**). See [OUTPUT_FOLDER_NAMING.md](OUTPUT_FOLDER_NAMING.md). Use them to quickly see what each run/sweep is and how it evolved.

---

## Roles

| Role | Output / config | Description |
|------|------------------|-------------|
| **Baseline** | **outputs6_baseline** (phase_1 outcome) | Phase 1 outcome runs: ListMLE, playoff_outcome, rolling [15,30]; configs from outputs4. Reference for all comparisons. |
| **Best** | **outputs8_spearman_surrogate** (e.g. combo_0033 / combo_0038) | **Spearman-surrogate** loss, playoff_outcome, rolling [10,30]. Best Spearman / playoff_spearman / rank_mae / rank_rmse. See [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md). |
| **MAP run** | **outputs14_map_run** (`config/outputs14_map_run.yaml`) | Future MAP branch model; per-game evaluation. Compare to baseline (outputs6_baseline) and best (outputs8_spearman_surrogate). See [OUTPUTS14_MAP_RUN.md](OUTPUTS14_MAP_RUN.md). |
| **RMSE surrogate** | **outputs13_rmse_surrogate** (`config/outputs13_sweep_rmse_surrogate.yaml`) | Same as outputs8_spearman_surrogate but **rank_rmse_surrogate** loss; Optuna minimizes rank_rmse. Compare to outputs8_spearman_surrogate. See [OUTPUTS13_RMSE_SURROGATE_SWEEP.md](OUTPUTS13_RMSE_SURROGATE_SWEEP.md). |
| **Baseline-style + standing rank** | **outputs12_baseline_standing_rank** (`config/outputs4_baseline_standing_rank.yaml`) | Same as baseline config (ListMLE, final_rank, [15,30]) but with **standing rank as input** (stat_dim 22). Single run. Compare to **baseline (outputs6_baseline)**. |
| **ListMLE + standing (sweep)** | **outputs11_listmle_standing_rank** | ListMLE loss, playoff_outcome, standing rank; Optuna sweep. Compare to **baseline (outputs6_baseline)** and outputs10_spearman_surrogate_standing_rank. |

---

## Still to test

1. **MAP branch model** — run with `config/outputs14_map_run.yaml` → **outputs14_map_run**; test and run **per-game** evaluation; compare to baseline (outputs6_baseline) and best (outputs8_spearman_surrogate).
2. **RMSE surrogate** — run the **outputs13_rmse_surrogate** sweep: `config/outputs13_sweep_rmse_surrogate.yaml`, `--objective rank_rmse`, then compare to Spearman-surrogate best (outputs8_spearman_surrogate).
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
| outputs14_map_run/ | **MAP run** (future); per-game evaluation; compare to outputs6_baseline, outputs8_spearman_surrogate. |
