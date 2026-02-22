# Model lineup and next steps

**Baseline vs best vs experiments.** This doc summarizes the canonical roles of each output set and what still needs to be tested.

**Per-folder docs:** Each output folder (**outputs/** through **outputs14/**) has a **`MODEL.md`** that describes (1) the model that produced those results and (2) the **difference from the previous output** folder. Use them to quickly see what each run/sweep is and how it evolved.

---

## Roles

| Role | Output / config | Description |
|------|------------------|-------------|
| **Baseline** | **outputs6** (phase_1 outcome) | Phase 1 outcome runs: ListMLE, playoff_outcome, rolling [15,30]; configs from outputs4. Reference for all comparisons. |
| **Best** | **outputs8** (e.g. combo_0033 / combo_0038) | **Spearman-surrogate** loss, playoff_outcome, rolling [10,30]. Best Spearman / playoff_spearman / rank_mae / rank_rmse. See [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md). |
| **MAP run** | **outputs14** (`config/outputs14_map_run.yaml`) | Future MAP branch model; per-game evaluation. Compare to baseline (outputs6) and best (outputs8). See [OUTPUTS14_MAP_RUN.md](OUTPUTS14_MAP_RUN.md). |
| **RMSE surrogate** | **outputs13** (`config/outputs13_sweep_rmse_surrogate.yaml`) | Same as outputs8 but **rank_rmse_surrogate** loss; Optuna minimizes rank_rmse. Compare to outputs8. See [OUTPUTS13_RMSE_SURROGATE_SWEEP.md](OUTPUTS13_RMSE_SURROGATE_SWEEP.md). |
| **Baseline-style + standing rank** | **outputs12** (`config/outputs4_baseline_standing_rank.yaml`) | Same as baseline config (ListMLE, final_rank, [15,30]) but with **standing rank as input** (stat_dim 22). Single run. Compare to **baseline (outputs6)**. |
| **outputs11** | ListMLE sweep + standing rank | ListMLE loss, playoff_outcome, standing rank; Optuna sweep. Compare to **baseline (outputs6)** and outputs10 (Spearman surrogate + standing). |

---

## Still to test

1. **MAP branch model** — run with `config/outputs14_map_run.yaml` → **outputs14**; test and run **per-game** evaluation; compare to baseline (outputs6) and best (outputs8).
2. **RMSE surrogate** — run the **outputs13** sweep: `config/outputs13_sweep_rmse_surrogate.yaml`, `--objective rank_rmse`, then compare to Spearman-surrogate best (outputs8).
3. **Baseline-style + standing rank** — single pipeline run with `config/outputs4_baseline_standing_rank.yaml` → `outputs12/baseline_standing_rank`; compare metrics to **baseline (outputs6)**.

---

## Run: baseline-style + standing rank (outputs12)

From project root:

```powershell
python -m scripts.run_pipeline_from_model_a --config config/outputs4_baseline_standing_rank.yaml --outputs outputs12/baseline_standing_rank
```

WSL:

```bash
export PYTHONPATH="$PWD"
python -m scripts.run_pipeline_from_model_a --config config/outputs4_baseline_standing_rank.yaml --outputs outputs12/baseline_standing_rank
```

Results under `outputs12/baseline_standing_rank/` (e.g. `outputs/run_XXX/eval_report.json`). Compare Spearman, playoff_spearman, rank_mae, rank_rmse, NDCG to **baseline (outputs6)** on the same eval (test seasons, eos_final_rank if available).

---

## Output folders and MODEL.md

| Folder    | MODEL.md describes |
|-----------|--------------------|
| outputs/  | Legacy/early runs; no prior output in lineage. |
| outputs2/ | Baseline full-pipeline runs (020/021/022); diff from outputs. |
| outputs3/ | Baseline and Phase 1 sweeps; diff from outputs2. |
| outputs4/ | Production/sweep root; config source for baseline. Phase 3 combo 18 etc.; diff from outputs3. |
| outputs5/ | Outcome vs standings comparison; diff from outputs4. |
| outputs6/ | **Baseline** (phase_1 outcome runs; reference for comparisons); diff from outputs5. |
| outputs7/ | ListMLE Optuna sweep; diff from outputs6. |
| outputs8/ | **Best** (Spearman surrogate); diff from outputs7. |
| outputs9/ | ListMLE sweep (spearman objective); diff from outputs8. |
| outputs10/| Spearman surrogate + standing rank; diff from outputs9. |
| outputs11/| ListMLE sweep + standing rank; compare to baseline (outputs6). |
| outputs12/| (When created) Baseline-style + standing rank (single run); compare to outputs6. |
| outputs13/| **RMSE surrogate** sweep (rank_rmse_surrogate); diff from outputs10. |
| outputs14/| **MAP run** (future); per-game evaluation; compare to outputs6, outputs8. |
