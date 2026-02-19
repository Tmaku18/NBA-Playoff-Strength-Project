# Official Best Configs and Cross-Run Analysis (Feb 2026)

**Last updated:** Feb 2026

---

## Summary

The **outputs8** sweep (Spearman-surrogate loss, 40 Optuna trials, batch 20260217_042955) is the current **official best** for playoff-outcome evaluation. This doc lists the official best config path per metric, compares outputs4, outputs6 phase_1 outcome (run_028), outputs7, and outputs8, notes the difference between outputs6 and outputs7, and states the planned **outputs9** sweep.

---

## 1. Official best config per metric (outputs8)

All paths under `outputs8/sweeps/20260217_042955/`. Eval: test seasons 2023-24, 2024-25; `eos_final_rank` (playoff outcome rank).

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
python -m scripts.run_pipeline_from_model_a --config "outputs8/sweeps/20260217_042955/combo_0033/config.yaml" --outputs "outputs8/official_best_spearman"
```

---

## 2. Cross-run comparison

| Metric | outputs4 (prev) | outputs6 phase_1 (run_028) | outputs7 best | outputs8 best | Winner |
|--------|-----------------|----------------------------|---------------|---------------|--------|
| **Spearman** | 0.557 | 0.749 (best_rmse) | 0.765 (combo 17) | **0.777** (combo 33) | outputs8 |
| **playoff_spearman** | 0.568 | 0.725 (best_rmse) | 0.742 (combo 30) | **0.854** (combo 38) | outputs8 |
| **rank_mae** | 6.33 | 5.27 (best_rmse) | 5.13 (combo 1) | **4.80** (combo 33) | outputs8 |
| **rank_rmse** | 8.15 | 6.13 (best_rmse) | 5.94 (combo 17) | **5.78** (combo 33) | outputs8 |
| **NDCG@30** | — | 0.450 (best_ndcg16) | 0.487 (combo 16) | **0.522** (combo 32) | outputs8 |

- **outputs4:** Phase 3 fine NDCG@16 combo 18; `listmle_target: final_rank`, rolling [15,30]. See [SWEEP_ANALYSIS_02-08.md](SWEEP_ANALYSIS_02-08.md).
- **outputs6 phase_1 outcome (run_028):** Pipeline runs into `outputs6/phase_1/outcome/best_*` using configs from outputs4 (e.g. WSL combo_0018 for best_rmse, Phase 6 combo_0016 for best_ndcg16). ListMLE, rolling [15,30].
- **outputs7:** Sweep 20260217_235940, 40 trials; ListMLE, rolling [10,30]. See [OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md](OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md).
- **outputs8:** Sweep 20260217_042955, 40 trials; **Spearman-surrogate** loss, rolling [10,30]. See [OUTPUTS8_SWEEP_ANALYSIS_02-17.md](OUTPUTS8_SWEEP_ANALYSIS_02-17.md).

---

## 3. outputs6 vs outputs7 (differences)

Both use **ListMLE** and **listmle_target: playoff_outcome**. Differences:

| Aspect | outputs6 | outputs7 |
|--------|----------|----------|
| **What it is** | Pipeline run **destination**: runs use **existing** configs from outputs4 with `--outputs "outputs6/phase_1/outcome/best_*"`. | **Sweep** output: Optuna **generated** 40 configs under `outputs7/sweeps/<batch_id>/combo_*/`. |
| **Config source** | Fixed configs from outputs4 (e.g. WSL combo_0018, Phase 6 combo_0016). | New configs from the sweep. |
| **training.rolling_windows** | **[15, 30]** (from outputs4). | **[10, 30]** (sweep/defaults). |
| **stacking.use_confidence** | **true** (in WSL configs used for best_rmse, best_ndcg30). | **false** (outputs7 baseline). |
| **Run ID** | 028 (when path contains `outputs6`). | 029 (when path contains `outputs7`). |

So: same loss (listmle); different **config source** (outputs4 vs sweep), **rolling windows** (15/30 vs 10/30), and **use_confidence** (true vs false).

---

## 4. Planned: outputs9 sweep

A sweep **outputs9** is planned with the **same mechanics** as outputs7 and outputs8 (Optuna, `paths.outputs: outputs9`, same phase/baseline style), but with **ListMLE** loss and config aligned with outputs6 (i.e. **no** spearman_surrogate; `training.loss_type: listmle`). Purpose: compare a ListMLE-based sweep vs the Spearman-surrogate sweep (outputs8) on the same evaluation. When run, use a config overlay similar to outputs7 (e.g. `paths.outputs: "outputs9"`, `training.loss_type: "listmle"`, `listmle_target: playoff_outcome`).

---

## 5. Related docs

- [OUTPUTS8_SWEEP_ANALYSIS_02-17.md](OUTPUTS8_SWEEP_ANALYSIS_02-17.md) — outputs8 sweep details and combo metrics  
- [OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md](OUTPUTS7_SWEEP_ANALYSIS_AND_COMPARISON.md) — outputs4 / outputs7 / outputs8 three-way comparison  
- [BEST_METRICS_02-15.md](BEST_METRICS_02-15.md) — Best configs and outputs6 layout  
- [SWEEP_ANALYSIS_02-08.md](SWEEP_ANALYSIS_02-08.md) — outputs4 Phase 2/3 sweep history  
