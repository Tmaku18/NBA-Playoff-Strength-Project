# Hyperparameter Sweep — Analysis & Findings Report

**Date:** 2026-02-03  
**Scope:** Grid and Optuna sweeps (scripts/sweep_hparams.py), existing batch results, run_022 baseline.

---

## 1. Sweep design (config/defaults.yaml)

| Dimension | Values | Count |
|-----------|--------|-------|
| **model_a_epochs** | [8, 16, 24, 28] | 4 |
| **rolling_windows** | [5,10], [10,20], [10,30], [15,30] | 4 |
| **max_depth (XGB)** | [3, 4, 5] | 3 |
| **learning_rate (XGB)** | [0.05, 0.08, 0.10] | 3 |
| **n_estimators_xgb** | [200, 250, 300, 350] | 4 |
| **n_estimators_rf** | [150, 200, 250] | 3 |
| **subsample** | [0.8] | 1 |
| **colsample_bytree** | [0.7] | 1 |
| **min_samples_leaf (RF)** | [5] | 1 |

- **Full grid size:** 4×4×3×3×4×3×1×1×1 = **1,728 combos**.
- **Pipeline per combo:** Script 3 (Model A) → 4 (Model B) → 4b (stacking) → 6 (inference) → 5 (evaluate) → 4c (clone classifier if `include_clone_classifier: true`).
- **Outputs per combo:** `<outputs>/sweeps/<batch_id>/combo_XXXX/outputs/` (split_info, OOF, models, run_XXX, eval_report.json).
- **Metrics collected:** From `eval_report.json`: test_metrics_ensemble (ndcg, spearman, mrr_top2/4, roc_auc_upset, playoff_metrics), test_metrics_model_a, test_metrics_xgb, test_metrics_rf.

**Optuna mode:** `--method optuna --n-trials N` (default 20). Maximizes `test_metrics_ensemble_spearman`. Same pipeline per trial; fewer trials than full grid.

---

## 2. Runs performed

### 2.1 Existing batches (outputs2/sweeps)

| Batch ID | Method | n_combos | Result |
|----------|--------|----------|--------|
| **20260201_165611** | grid | 1 | Combo 0 **failed** at **Model A** (error: "Model A"). No metrics. |
| **20260201_165650** | grid | 1 | Combo 0 **failed** at **Model A**. No metrics. |
| **20260203_021923** | grid | 4 (max-combos) | **Timed out** during combo 0 (Model A OOF + final training completed; pipeline did not reach script 4). No sweep_results.csv. |
| **optuna_3trial** | optuna | 3 trials | **Timed out** during trial 1 (Model A completed; pipeline did not finish 4→5). No optuna_study.json / sweep_results. |

### 2.2 Failure point

- All observed failures/timeouts occur **during or right after Model A (script 3)**.
- Model A often hits early stopping (“Model A is not learning: train loss did not improve”) and still writes OOF and final model; the pipeline then continues to script 4. So the **reported error "Model A"** in the two 20260201 batches likely means script 3 **exited non-zero** (e.g. assertion or exception), not just early stopping.
- In the timed-out runs, script 3 **did** complete (oof_model_a.parquet, best_deep_set.pt, split_info.json written); the timeout happened later (script 4 or beyond).

---

## 3. Baseline: run_022 (single full pipeline)

The only **completed** full pipeline with evaluation in this repo is **run_022** (outputs2/run_022), used as the baseline for “what good looks like” until sweeps complete.

**Config (effective):** rolling_windows [10, 30], model_a epochs 28, XGB max_depth 4, lr 0.08, n_estimators 250, RF 200, min_samples_leaf 5 (from defaults; run_022 was produced by scripts 3–5 with defaults/outputs2).

**Test metrics (ensemble, last test season 2024-25):**

| Metric | Value |
|--------|--------|
| NDCG | 0.482 |
| Spearman | 0.430 |
| MRR top-2 | 0.50 |
| MRR top-4 | 0.50 |
| ROC-AUC upset | 0.73 |
| spearman_pred_vs_playoff_rank | 0.46 |
| ndcg_at_4_final_four | 0.46 |
| brier_championship_odds | 0.032 |

**Per-conference (EOS-derived within conference):** East NDCG 0.25, Spearman 0.25; West NDCG 0.75, Spearman 0.50.

So any **successful** sweep combo should be compared to these numbers (and to run_022’s eval_report_*.json / RESULTS_AND_OUTPUTS_EXPLAINED.md).

---

## 4. Findings summary

1. **No successful sweep combos in repo**  
   Every batch either failed at Model A (2 batches) or timed out before writing sweep_results (2 batches). So there are **no sweep_results.csv rows with metrics** to analyze yet.

2. **Full grid is large**  
   1,728 combos × ~10–30+ minutes per combo ⇒ **very long** total runtime. Use `--max-combos N` (e.g. 8–20) or Optuna with `--n-trials 20–50` for a shorter run.

3. **Clone classifier**  
   `include_clone_classifier: true` adds script 4c at the end. If 4c fails, the combo is reported as error. Disable in config (`sweep.include_clone_classifier: false`) if clone data or deps are missing, to avoid failing combos after a long pipeline.

4. **Model A early stopping**  
   “Model A is not learning” is common with some hyperparameters (e.g. rolling=[5,10], epochs=8). The model still saves and the pipeline continues; the combo only fails if script 3 exits with non-zero. So some combos may complete with “weak” Model A; sweep metrics will show that.

5. **Where results go**  
   - Grid: `sweep_results.csv`, `sweep_results_summary.json` (best by Spearman, NDCG, rank_mae) in `<outputs>/sweeps/<batch_id>/`.  
   - Optuna: `optuna_study.json` (best_value, best_params) plus same CSV/summary if implemented.  
   - Per-combo: `combo_XXXX/outputs/eval_report.json` (and run_XXX under outputs).

6. **Optuna warning**  
   Categorical choices for `rolling_windows` are tuples; Optuna warns for persistence. Cosmetic; does not affect optimization.

---

## 5. Recommendations

1. **Run a shorter sweep to get at least one full result**  
   - `python -m scripts.sweep_hparams --max-combos 8`  
   - Or `python -m scripts.sweep_hparams --method optuna --n-trials 10 --batch-id optuna_10`  
   Run in background or on a server (no 10-minute timeout); allow ~2–4 hours for 8 combos or 10 trials.

2. **Turn off clone for sweep if not needed**  
   In `config/defaults.yaml`, set `sweep.include_clone_classifier: false` to avoid 4c failures and shorten each combo.

3. **Use run_022 as the comparison baseline**  
   When sweep results exist, compare best_by_spearman / best_by_ndcg to run_022’s ensemble metrics and per-conference metrics.

4. **Optional: reduce grid for a quick full grid**  
   e.g. `model_a_epochs: [16, 28]`, `rolling_windows: [[10, 30]]`, `max_depth: [4]`, `learning_rate: [0.08]`, single values for the rest ⇒ 2×1×1×1 = 2 combos for a fast end-to-end test.

---

## 6. How to run and re-analyze

```bash
# Dry-run (see combo count)
python -m scripts.sweep_hparams --dry-run

# Grid with limit
python -m scripts.sweep_hparams --max-combos 8

# Optuna
python -m scripts.sweep_hparams --method optuna --n-trials 10 --batch-id my_run

# After a completed batch, open:
#   <outputs>/sweeps/<batch_id>/sweep_results.csv
#   <outputs>/sweeps/<batch_id>/sweep_results_summary.json
```

Re-run this analysis (or extend it) once `sweep_results.csv` has at least one row without `error`: compare best combo to run_022, and report best_by_spearman, best_by_ndcg, and best_by_rank_mae from `sweep_results_summary.json`.
