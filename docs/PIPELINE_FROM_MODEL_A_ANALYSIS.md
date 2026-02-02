# Pipeline from Model A Onwards — Run and Analysis

## What runs

**Script:** `scripts/run_from_model_a.py`  
**Steps (in order):** 3_train_model_a → 4_train_model_b → 4b_train_stacking → 6_run_inference → 5_evaluate → 5b_explain.

**Model A debug:** `config/defaults.yaml` has `model_a.attention_debug: true`, so each epoch you see:

- **Training loss:** `epoch N loss=X.XXXX` (and `val_loss N` if early stopping is used).
- **Attention diagnostics:** `Attention debug (train): teams=… attn_sum_mean=X.XX attn_max_mean=… attn_grad_norm=X.XX`.

Interpretation:

- **Loss decreasing** over epochs → Model A is learning; ListMLE is no longer flat.
- **attn_sum_mean ≈ 1.0** (normalized weights) and **attn_grad_norm > 0** → attention is active and gradients flow; no collapsed zero attention.

## Where results live

| Output | Location |
|--------|----------|
| Model A checkpoint | `outputs3/best_deep_set.pt` |
| OOF Model A | `outputs3/oof_model_a.parquet` |
| Model B (XGB, RF) | `outputs3/xgb_model.joblib`, `outputs3/rf_model.joblib` |
| Stacker | `outputs3/ridgecv_meta.joblib` |
| Run folder | `outputs3/run_XXX/` (XXX from `.current_run`) |
| Predictions | `outputs3/run_XXX/predictions.json` |
| Eval report | `outputs3/run_XXX/eval_report.json` (after step 5) |
| Explain / figures | `outputs3/run_XXX/` (after step 5b) |

## How to interpret results

**1. Training (step 3)**  
- Loss should **decrease** (e.g. first epoch ~13–28, last epoch lower).  
- Attention debug: **attn_sum_mean** and **attn_grad_norm** non-zero each epoch.

**2. Eval report (`eval_report.json`)**  
- **test_metrics_ensemble:** Spearman, NDCG, MRR, ROC-AUC (upset), playoff_metrics.  
- **test_metrics_model_a / xgb / rf:** Same metrics per base model.  
- **by_season:** Same metrics broken down by test season.

**3. Predictions (`predictions.json`)**  
- Per team: `prediction` (predicted_strength, ensemble_score, championship_odds, …), `analysis` (actual ranks, classification), `roster_dependence`:  
  - **primary_contributors:** List of {player, attention_weight} from Model A.  
  - **contributors_are_fallback:** `true` when attention was all zero/non-finite → contributors list is empty and this flag is set (no fabricated contributors).

## Skipping batch rebuild (cache)

If you already built batches and config/lists are unchanged, set in `config/defaults.yaml`:

- `training.skip_batch_rebuild: true`
- `training.batch_cache_dir: "outputs3/cache"` (or another path under the project)

Script 3 will then load OOF and final batches from `outputs3/cache/batches_<key>.pt` when the cache key matches (config + list signature). First run still builds and writes the cache; later runs with the same config and lists skip building.

## WSL note

If the project is on a Windows mount (e.g. OneDrive), step 6 may fail with `Operation not permitted` when writing `outputs3/run_XXX/predictions.json`. Then:

- Run steps 6, 5, 5b from **Windows** (PowerShell/cmd) with the same config, or  
- Move the project (or at least `outputs3/`) to a Linux path (e.g. under `/home/...`) and re-run from WSL.

See `docs/NEXT_STEPS_WSL.md` for details.
