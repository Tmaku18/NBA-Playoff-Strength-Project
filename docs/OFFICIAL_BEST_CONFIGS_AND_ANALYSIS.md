# Official Best Configs and Cross-Run Analysis (Feb 2026)

**Last updated:** Feb 2026

---

## Summary

The **outputs8_spearman_surrogate** sweep (Spearman-surrogate loss, 40 Optuna trials, batch 20260217_042955) is the current **official best** for playoff-outcome evaluation. This doc lists the official best config path per metric, compares outputs4, outputs6 phase_1 outcome (run_028), outputs7, and outputs8, notes the difference between outputs6 and outputs7, and states the planned **outputs9** sweep.

---

## 1. Official best config per metric (outputs8_spearman_surrogate)

All paths under `outputs8_spearman_surrogate/sweeps/20260217_042955/`. Eval: test seasons 2023-24, 2024-25; `eos_final_rank` (playoff outcome rank).

| Metric | Official best config | Key value |
|--------|----------------------|-----------|
| **Spearman** | `combo_0033/config.yaml` | 0.777 |
| **playoff_spearman** | `combo_0038/config.yaml` | 0.854 |
| **rank_mae** | `combo_0033/config.yaml` | 4.80 |
| **rank_rmse** | `combo_0033/config.yaml` | 5.78 |
| **NDCG@4 / NDCG@16 / NDCG@20 / NDCG@30** | `combo_0032/config.yaml` | NDCG@30 0.522 |
| spearman_standings | combo_0001 | — |
| ndcg4_standings | combo_0009 | — |
| ndcg16_standings | combo_0033 | — |
| ndcg30_standings | combo_0020 | — |
| rank_rmse_standings | combo_0038 | — |

**Example:** Run pipeline with best Spearman config:
```powershell
python -m scripts.run_pipeline_from_model_a --config "outputs8_spearman_surrogate/sweeps/20260217_042955/combo_0033/config.yaml" --outputs "outputs8_spearman_surrogate/official_best_spearman"
```

---

## 2. Cross-run comparison

| Metric | outputs4 (prev) | outputs6_baseline phase_1 (run_028) | outputs7_listmle_sweep best | outputs8_spearman_surrogate best | Winner |
|--------|-----------------|-------------------------------------|-----------------------------|----------------------------------|--------|
| **Spearman** | 0.557 | 0.749 (best_rmse) | 0.765 (combo 17) | **0.777** (combo 33) | outputs8_spearman_surrogate |
| **playoff_spearman** | 0.568 | 0.725 (best_rmse) | 0.742 (combo 30) | **0.854** (combo 38) | outputs8_spearman_surrogate |
| **rank_mae** | 6.33 | 5.27 (best_rmse) | 5.13 (combo 1) | **4.80** (combo 33) | outputs8_spearman_surrogate |
| **rank_rmse** | 8.15 | 6.13 (best_rmse) | 5.94 (combo 17) | **5.78** (combo 33) | outputs8_spearman_surrogate |
| **NDCG@30** | — | 0.450 (best_ndcg16) | 0.487 (combo 16) | **0.522** (combo 32) | outputs8_spearman_surrogate |

- **outputs4:** Phase 3 fine NDCG@16 combo 18; `listmle_target: final_rank`, rolling [15,30]. See [SWEEP_ANALYSIS_02-08.md](SWEEP_ANALYSIS_02-08.md).
- **outputs6_baseline phase_1 outcome (run_028):** Pipeline runs into `outputs6_baseline/phase_1/outcome/best_*` using configs from outputs4 (e.g. WSL combo_0018 for best_rmse, Phase 6 combo_0016 for best_ndcg16). ListMLE, rolling [15,30].
- **outputs7_listmle_sweep:** Sweep 20260217_235940, 40 trials; ListMLE, rolling [10,30]. See [OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md](OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md).
- **outputs8_spearman_surrogate:** Sweep 20260217_042955, 40 trials; **Spearman-surrogate** loss, rolling [10,30]. See [OUTPUTS8_SWEEP_ANALYSIS_02-17.md](OUTPUTS8_SWEEP_ANALYSIS_02-17.md).

---

## 3. outputs6 vs outputs7 (differences)

Both use **ListMLE** and **listmle_target: playoff_outcome**. Differences:

| Aspect | outputs6 | outputs7 |
|--------|----------|----------|
| **What it is** | Pipeline run **destination**: runs use **existing** configs from outputs4 with `--outputs "outputs6_baseline/phase_1/outcome/best_*"`. | **Sweep** output: Optuna **generated** 40 configs under `outputs7_listmle_sweep/sweeps/<batch_id>/combo_*/`. |
| **Config source** | Fixed configs from outputs4 (e.g. WSL combo_0018, Phase 6 combo_0016). | New configs from the sweep. |
| **training.rolling_windows** | **[15, 30]** (from outputs4). | **[10, 30]** (sweep/defaults). |
| **stacking.use_confidence** | **true** (in WSL configs used for best_rmse, best_ndcg30). | **false** (outputs7 baseline). |
| **Run ID** | 028 (when path contains `outputs6`). | 029 (when path contains `outputs7`). |

So: same loss (listmle); different **config source** (outputs4 vs sweep), **rolling windows** (15/30 vs 10/30), and **use_confidence** (true vs false).

---

## 4. Planned: outputs9 sweep

A sweep **outputs9** is planned with the **same mechanics** as outputs7 and outputs8 (Optuna, `paths.outputs: outputs9`, same phase/baseline style), but with **ListMLE** loss and config aligned with outputs6 (i.e. **no** spearman_surrogate; `training.loss_type: listmle`). Purpose: compare a ListMLE-based sweep vs the Spearman-surrogate sweep (outputs8) on the same evaluation. When run, use a config overlay similar to outputs7 (e.g. `paths.outputs: "outputs9"`, `training.loss_type: "listmle"`, `listmle_target: playoff_outcome`).

## 5. outputs10 sweep (standing rank as input)

**outputs10** uses the **same methodology as outputs8** (Spearman-surrogate, playoff_outcome, Optuna) but with the **new implementation**: **standing rank as an input feature** for Model A, B, and C (see [STANDING_RANK_FEATURE.md](STANDING_RANK_FEATURE.md)). Hypothesis: since standings-trained models matched or beat outcome-trained in prior comparisons, giving the model current standings as input should increase accuracy. Config: `config/outputs10_sweep_standing_rank.yaml`. Writes to `outputs10/sweeps/<batch_id>/`. Optuna results: `optuna_study.json`, `optuna_importances.json`, `sweep_results.csv`, `sweep_results_summary.json`. See [OUTPUTS10_SWEEP_STANDING_RANK.md](OUTPUTS10_SWEEP_STANDING_RANK.md).

---

## 6. Model lineup and next steps

- **Baseline:** **outputs6_baseline** (phase_1 outcome runs). **Best:** outputs8_spearman_surrogate (Spearman surrogate).
- **Still to test:** **MAP run** → **outputs14_map_run** (`config/outputs14_map_run.yaml`); per-game eval; compare to outputs6_baseline, outputs8_spearman_surrogate. See [OUTPUTS14_MAP_RUN.md](OUTPUTS14_MAP_RUN.md).
- **outputs13 / RMSE surrogate:** Tested (sweep + 80-epoch singular run). **Not recommended** — underperformed Spearman surrogate on all primary metrics; see [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md) and [OUTPUTS13_RMSE_SURROGATE_SWEEP.md](OUTPUTS13_RMSE_SURROGATE_SWEEP.md).
- **Baseline-style + standing rank:** Single run with same training as baseline config but standing rank as input. Config: `config/outputs4_baseline_standing_rank.yaml`; run to `outputs12_baseline_standing_rank/baseline_standing_rank`. Compare to **baseline (outputs6_baseline)**. See [MODEL_LINEUP_AND_NEXT_STEPS.md](MODEL_LINEUP_AND_NEXT_STEPS.md).
- **Per-folder model docs:** Each output folder has a **`MODEL.md`** at its root. Folder names use differentiators (e.g. **outputs6_baseline**, **outputs8_spearman_surrogate**). See [OUTPUT_FOLDER_NAMING.md](OUTPUT_FOLDER_NAMING.md).

## 7. outputs13 / RMSE surrogate (not recommended)

The **rank_rmse_surrogate** sweep (outputs13) and an 80-epoch singular run were run and compared to outputs8. **Findings:** RMSE surrogate did not achieve positive correlation; best rank_rmse was ~13.2 (vs outputs8 5.78); Spearman and playoff_spearman were negative or near zero. **Use outputs8 (Spearman surrogate) for production.** Full write-up: [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md).

---

## 8. Related docs

- [MODEL_LINEUP_AND_NEXT_STEPS.md](MODEL_LINEUP_AND_NEXT_STEPS.md) — Baseline vs best, MAP/per-game, outputs4+standing
- [RMSE_SURROGATE_FINDINGS.md](RMSE_SURROGATE_FINDINGS.md) — outputs13 RMSE surrogate consolidated findings (not recommended)
- [OUTPUTS8_SWEEP_ANALYSIS_02-17.md](OUTPUTS8_SWEEP_ANALYSIS_02-17.md) — outputs8 sweep details and combo metrics  
- [OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md](OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md) — outputs4 / outputs7 / outputs8 three-way comparison  
- [BEST_METRICS_02-15.md](BEST_METRICS_02-15.md) — Best configs and outputs6 layout  
- [SWEEP_ANALYSIS_02-08.md](SWEEP_ANALYSIS_02-08.md) — outputs4 Phase 2/3 sweep history  
- [BRANCH_CLEANUP_SUGGESTIONS.md](BRANCH_CLEANUP_SUGGESTIONS.md) — which branches can be deleted (no dedicated output or unproductive)  
