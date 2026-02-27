# Analysis 01 — outputs13 RMSE surrogate sweep (rmse_surrogate_40)

**Sweep:** 40 Optuna trials, objective **rank_rmse** (minimize), `listmle_target: playoff_outcome`, rolling [10,30], **rank_rmse_surrogate** loss. No standing rank as input (baseline feature set).

---

## Best trial (Optuna objective = rank_rmse)

| Field | Value |
|-------|--------|
| **Best combo** | **combo_004** |
| **rank_rmse** (ensemble vs playoff outcome) | 13.21 |
| **rank_mae** | 11.0 |
| **Spearman** (ensemble) | -0.164 |
| **playoff_spearman** | -0.296 |
| **NDCG@30** | 0.277 |
| **NDCG@4** | 0.006 |

**Best combo params:** model_a_epochs 22, max_depth 4, lr 0.08, n_estimators_xgb 258, n_estimators_rf 200, rolling [10,30], subsample 0.8, colsample_bytree 0.7, min_samples_leaf 5.

---

## Comparison to official best (outputs8_spearman_surrogate)

| Metric | outputs13 (RMSE surrogate) best | outputs8 (Spearman surrogate) best | Winner |
|--------|----------------------------------|-------------------------------------|--------|
| **Spearman** | -0.164 (combo_004) | **0.777** (combo_0033) | outputs8 |
| **playoff_spearman** | -0.296 (combo_004) | **0.854** (combo_0038) | outputs8 |
| **rank_mae** | 11.0 | **4.80** (combo_0033) | outputs8 |
| **rank_rmse** | 13.21 | **5.78** (combo_0033) | outputs8 |
| **NDCG@30** | 0.277 | **0.522** (combo_0032) | outputs8 |

**Summary:** The **RMSE surrogate** sweep (outputs13) underperforms the **Spearman surrogate** sweep (outputs8) on every primary metric. Ensemble Spearman and playoff_spearman are **negative** for the best rank_rmse combo, indicating Model A (trained with rank_rmse_surrogate) is not aligning with playoff-outcome rank on this test set. **outputs8 remains the official best**; the RMSE-surrogate objective did not improve rank error or correlation over the Spearman surrogate.

---

## vs W/L standings baseline (same test seasons)

- **Standings:** rank_mae 3.13, rank_rmse 4.45 vs playoff outcome.
- **Ensemble (combo_004):** rank_mae 11.0, rank_rmse 13.21 — model is **worse** than W/L standings on rank distance and has negative correlation (standings have positive spearman_standings in some combos).

---

## Conference (East vs West)

- **best_by_ndcg** (E): combo_008; **best_by_spearman** (W): combo_037 (per by_conference_summary).
- Ensemble metrics are weak or negative in both conferences; no combo reaches the correlation or rank-accuracy levels of outputs8.

---

## Files

- **Full results:** `sweep_results.csv`, `sweep_results_summary.json`
- **Best config (by rank_rmse):** `combo_0004/config.yaml`
- **Optuna:** `optuna_study.json`, `optuna_importances.json` (if generated)

---

## Why so bad? Possible underfitting (epochs)

The sweep’s best combo used **22 Model A epochs** (Optuna-sampled). Model A training loss is **not saved to disk** — it is only printed to stdout (`epoch N loss=X`), so we cannot inspect after the fact whether loss was still decreasing at epoch 22.

If you observed loss still going down when the run finished, **22 epochs may have been too low** for the RMSE surrogate. To test:

- **Single run with higher epochs:** Use config `config/outputs13_combo004_high_epochs.yaml` (same combo_004 hyperparameters, **80 epochs** max, **early_stopping_patience: 10**). Run from project root:
  ```bash
  export PYTHONPATH="$PWD"
  python -m scripts.run_pipeline_from_model_a --config config/outputs13_combo004_high_epochs.yaml
  ```
  Watch script 3 output for `epoch N loss=X`; training can stop earlier if train loss doesn’t improve for 3 epochs or val loss for 10. Outputs go to `outputs13_rmse_surrogate/combo_004_high_epochs/`. Compare eval metrics there to the 22-epoch combo_004 run.

See [docs/OUTPUTS13_RMSE_SURROGATE_SWEEP.md](../../../docs/OUTPUTS13_RMSE_SURROGATE_SWEEP.md) and [docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](../../../docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md).
