# Project state and best models — analysis (Feb 27, 2026)

Snapshot of the project's current state and a ranking of all trained models by evidence, based on docs, sweep results (`sweep_results.csv`), and eval reports across all `output/` folders.

---

## Project state

The project is in good shape structurally:

- All outputs consolidated under **`output/`** with model-based names ([OUTPUT_FOLDER_NAMING.md](OUTPUT_FOLDER_NAMING.md)).
- Docs current: [MODELS.md](MODELS.md) (all models labeled **A–H**), [MEMOIZATION.md](MEMOIZATION.md), [MODEL_LINEUP_AND_NEXT_STEPS.md](MODEL_LINEUP_AND_NEXT_STEPS.md), [OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md).
- Recent work: feature-rank vs playoff-outcome plots (scatter + KDE/PDF), memoization across training/inference (batch cache script 3; shared feature cache in scripts 4/5b/4c/plots/inference via `src.features.feature_cache`), and inference that runs on **any subset of models**.
- The 40-trial wide sweeps for the team-stats models (**D–G**) and logistic regression (**H**) have all completed.

---

## Best models (ranked by evidence)

### 1. Model A + Spearman-surrogate loss + ensemble — `8_spearman_surrogate` (official best)

From the 40-trial Optuna sweep (batch `20260217_042955`), evaluated on 2023-24 and 2024-25 test seasons vs playoff outcome:

| Metric | Best combo | Value |
|--------|-----------|-------|
| Spearman | combo_0033 | **0.777** |
| Playoff Spearman | combo_0038 | **0.854** |
| Rank MAE / RMSE | combo_0033 | **4.80 / 5.78** |
| NDCG@30 | combo_0032 | 0.522 |

Beats every alternative on every metric: ListMLE baseline `6_baseline` (Spearman 0.749), ListMLE sweep `7_listmle` (0.765), older production `4_listmle` (0.557).

### 2. ListMLE (`7_listmle` / `6_baseline`)

Solid second: Spearman 0.749–0.765. Reference baseline for all comparisons.

---

## What didn't work

| Experiment | Result |
|-----------|--------|
| **Standing rank as input** (`10_spearman_surrogate_standing_rank`, `11_listmle_standing_rank`) | Hurt badly: best Spearman **0.309** (vs 0.777 without) and **0.459** (vs 0.765). Hypothesis not supported. |
| **RMSE surrogate** (`13_rmse_surrogate`) | Failed: best Spearman **−0.164**. Confirmed "not recommended". |
| **MAP** (`14_map_run`) | Mixed: better top-end NDCG/champion placement, worse Spearman and rank error. |
| **MAP + standing rank** (`16_map_standing_rank`) | **Never run** — folder contains only `MODEL.md` and `.gitkeep`. Open item on the test matrix. |

---

## Team-stats models (D–H) — weak as primary models

All five 40-combo wide sweeps finished; ensembles land far below the main pipeline (Spearman ~0.26–0.29 vs 0.777):

| Model | Best ensemble Spearman | Playoff Spearman | Rank MAE |
|-------|------------------------|------------------|----------|
| F — GPR | 0.286 | 0.293 | 6.53 |
| D — Linear Regression | 0.283 | 0.290 | 6.53 |
| E — Bayesian Ridge | 0.277 | 0.300 | 6.63 |
| H — Logistic Regression | 0.274 | 0.293 | 6.53 |
| G — GMM | 0.258 | **0.325** | **6.42** |

Caveats from eval reports:

- Every sweep had **champion_rank = 18** (eventual champion ranked 18th — very poor top-end).
- All **lose to the plain W/L-standings baseline** (rank MAE 4.0 vs their ~6.5); bootstrap significance shows negative MAE improvement with p ≈ 1.0.
- `team_stats_spearman_surrogate` branch run is similarly weak (ensemble Spearman 0.137).

These models remain useful as inference extras/diagnostics (inference now runs on any model subset), but none is a candidate to replace the main pipeline.

---

## Recommendations

1. **Production:** use `8_spearman_surrogate` combo_0033 (best overall balance) or combo_0038 if playoff Spearman is the priority.
2. **Drop standing-rank input** from future sweeps — it degraded both loss types consistently.
3. **Finish the test matrix:** `16_map_standing_rank` is the only planned experiment not yet run.
4. **Team-stats track:** keep Models D–G as comparison/extras in inference. If improving them, the biggest gap is top-end ranking (champion placement), not average correlation — a top-weighted objective is the thing to try.

---

## Improvement opportunities for the best model (8_spearman_surrogate)

*(Filled in below after switching to `main` and analyzing best-combo configs, Optuna importances, and per-conference evals — see § Improvements.)*
