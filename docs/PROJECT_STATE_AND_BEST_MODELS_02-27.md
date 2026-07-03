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

Based on the combo_0033 config (`output/8_spearman_surrogate/sweeps/20260217_042955/combo_0033/config.yaml`), its eval report (`.../outputs/run_025_02-17/eval_report.json`), and the sweep analysis ([OUTPUTS8_SWEEP_ANALYSIS_02-17.md](OUTPUTS8_SWEEP_ANALYSIS_02-17.md)).

### Key observations from the best run

| Observation | Evidence |
|-------------|----------|
| **Model A's raw score is inverted** | Model A test Spearman is **−0.661** on its own; the RidgeCV meta learned coef ≈ **[−37.2 (A), +43.2 (B)]**, i.e. it flips A's sign. The ensemble (0.777) works *because* the meta negates A. |
| **Standings still beat the ensemble on rank MAE** | Standings MAE **3.13** vs ensemble **4.80** (E: 2.93 vs 5.33; W: 3.33 vs 4.27). The model wins on correlation, loses on absolute rank error. |
| **Top-end is weak in the Spearman-best combo** | ndcg@4 **0.042**, precision@4 **0.0**, champion_rank 7, champion_in_top_4 0. Combo_0032 trades a little Spearman (0.737) for far better top-end (ndcg@4 0.349, champion_rank **3**, champion_in_top_4 **1.0**). |
| **East is weaker than West** | E Spearman 0.729 / MAE 5.33 vs W 0.793 / 4.27. Also `ridgecv_meta_E.joblib` has coefficients identical to the global meta — the per-conference meta isn't adding anything. |
| **Season variance** | 2023-24 Spearman 0.675 vs 2024-25 0.777 — one full season gap in generalization. |
| **Features off in the best config** | `sos_srs.enabled: false`, `team_rolling.enabled: false`, `injury.enabled: false`, `stacking.use_confidence: false`. Elo, Massey, motivation, RAPTOR are on. |
| **Training caps** | `early_stopping_patience: 0` (no early stopping), epochs 15, `max_lists_oof: 100`, `max_final_batches: 100` (heavy subsampling of training lists). |

### Ranked improvements

1. **Fix / exploit the Model A sign inversion (highest value, low effort).**
   Model A trained with the Spearman surrogate produces scores *negatively* correlated with strength; the stacker rescues it. Verify the surrogate loss sign convention (higher score should mean better rank). If A's score were correctly oriented and individually strong (target: Spearman > 0.6 alone instead of −0.66), the meta could blend two strong models instead of subtracting a noisy one — likely raising both Spearman and MAE.

2. **Blend standings into the stacker (not into Model A).**
   Standings-as-input hurt Model A (outputs 10/11), but standings as a **third stacking column** is different: the meta could anchor on standings (MAE 3.13) and use A+B for the correlation signal. Concretely: add `wl_record` rank to the OOF table in script 4b and let RidgeCV weight it. This directly attacks the MAE gap vs baseline.

3. **Top-end objective for production use.**
   For championship questions, use combo_0032 (or re-sweep with a top-weighted Spearman surrogate that up-weights ranks 1–4). Candidate: weighted Spearman surrogate with weights ∝ 1/rank, or a two-head loss (Spearman + NDCG@4 surrogate). The current best-Spearman combo is effectively unusable for champion prediction (precision@4 = 0).

4. **Feature ablation sweep on the flags that are off.**
   `sos_srs`, `team_rolling`, and `injury` are disabled in the best config, but were never swept as toggles in outputs8 (the sweep varied epochs/XGB HPs only). A small 8–12 trial sweep toggling these three (with combo_0033 HPs fixed) is cheap thanks to the batch/feature caches and could lift Model B especially in the East, where MAE is worst.

5. **Confidence-weighted stacking (4-column) on the best config.**
   `stacking.use_confidence: false` in combo_0033; the code supports attention-entropy confidence for A and tree-variance for B. Given the meta currently has to fully invert A, per-team confidence could down-weight A where its attention is diffuse. One pipeline re-run with `use_confidence: true` answers this.

6. **Lift training caps + early stopping.**
   `max_lists_oof: 100` / `max_final_batches: 100` subsample the training lists, and early stopping is off (fixed 15 epochs). Raising caps to 200–300 and enabling `early_stopping_patience: 3–5` with `val_frac 0.25` is a low-risk change that uses more data without overfitting; caches keep the cost manageable.

7. **East-specific investigation.**
   E is consistently worse across all runs (also seen in team-stats sweeps). Check East roster churn / play-in volatility in the feature plots (`docs/feature_rank_vs_playoff_outcome_pdf/`); consider conference-specific `odds_temperature` or a true per-conference meta (the current `ridgecv_meta_E` is a copy of the global one — fit it on East-only OOF rows).

### Suggested order of execution

Steps 1 and 2 first (both cheap, both target the two biggest weaknesses: A's inversion and MAE vs standings), then 5 and 6 as single pipeline re-runs, then the ablation sweep (4), then the top-end loss work (3) as the larger research item.

---

## Implementation status (all 7 improvements implemented — Feb 27, 2026)

1. **Sign inversion — root cause found and fixed.** The bug was in `_soft_ranks` in `src/models/ranking_surrogate_losses.py`: `scores.unsqueeze(2) - scores.unsqueeze(1)` computes `s_i - s_j`, not `s_j - s_i` as the comment claimed, so soft ranks were inverted (highest score got rank L instead of rank 1). Training therefore drove the *best* team to the *lowest* score — confirmed empirically by optimizing free scores against a fixed relevance vector (best team converged to the lowest score before the fix, highest after). This affected both `spearman_surrogate` and `rank_rmse_surrogate` (which explains `13_rmse_surrogate`'s Spearman of −0.164). Fixed to `scores.unsqueeze(1) - scores.unsqueeze(2)`.
   A second inconsistency was fixed in script 4b: with `target_rank: playoffs` it wrote playoff rank (1 = best) into a `y` column whose non-replaced rows held relevance (30 = best), so the meta target mixed two opposite conventions. Now y is always **higher = better** (`31 - playoff_rank`), matching inference, and rows without a playoff rank are NaN → mean-imputed instead of scale-mixed.
2. **Standings as stacking column.** Script 4b keeps Model B's win-rate-to-date as a `standings_wr` column and (with `stacking.use_standings: true`, on in the improved config) passes it as an extra meta column. Column order convention lives in `src/models/stacking.py`; inference (`src/inference/predict.py`) detects meta width (2/3/4/5 cols) and feeds win rate to date accordingly.
3. **Top-weighted Spearman surrogate.** New `weighted_spearman_surrogate` loss (weights ∝ 1/actual_rank^`training.loss_top_weight_power`) in `src/models/ranking_surrogate_losses.py`, wired through `training.loss_type`. Config: `config/8_spearman_improved_topweighted.yaml`.
4. **Flag-ablation sweep.** New `--phase flags` in `scripts/sweep_hparams.py`: 8-combo grid over `sos_srs`/`team_rolling`/`injury` toggles with combo_0033 HPs fixed. Run: `python -m scripts.sweep_hparams --phase flags --config config/8_spearman_improved.yaml --n-jobs 3`.
5. **Confidence-weighted stacking.** `stacking.use_confidence: true` in the improved config; script 4b now respects this flag (before it silently used confidence whenever columns existed).
6. **Caps + early stopping.** Improved config: `max_lists_oof`/`max_final_batches` 100 → 250, `epochs` 15 → 20 with `early_stopping_patience: 4`, `val_frac 0.25`.
7. **Per-conference metas.** Script 4b / `src/training/train_stacking.py` now fit `ridgecv_meta_E.joblib` / `ridgecv_meta_W.joblib` on conference-only OOF rows (min 50 rows), replacing the previous copy of the global meta.

**Configs:** `config/8_spearman_improved.yaml` (improvements 1, 2, 5, 6, 7 in one run; outputs → `output/8_spearman_surrogate/improved_02-27/`) and `config/8_spearman_improved_topweighted.yaml` (adds improvement 3; outputs → `output/8_spearman_surrogate/improved_topweighted_02-27/`).

**Note on old artifacts:** models/metas trained before the sign fix (including combo_0033 artifacts) keep their inverted-A convention internally; retrain with the improved configs before comparing raw Model A scores across runs.

**Bonus bug found while implementing improvement 7 (East-only training).** Script 3's list subsampling (`sorted_by_date[::step]`) walked lists sorted by (date, conference); since lists alternate E/W per date, any even step picked a **single conference**. With 282 train lists and `max_lists_oof: 100` (step = 2), combo_0033's Model A OOF is 1500 rows **all East**, and the RidgeCV meta was fit on those East-only rows (Model B OOF has 2115 E + 2115 W rows; the inner join leaves only E). The final Model A was also trained on a single-conference list subset. This plausibly contributes to both the East/West asymmetry and Model A's weak solo performance. Fixed with date-stratified subsampling (`_subsample_lists_stratified` keeps every conference for each kept date); script 3 also now writes a correct `conference` column into `oof_model_a.parquet`, and script 4b picks whichever conference column actually contains both E and W.

---

## Results of the improved run (Jul 2026 re-run, `improved_02-27`)

Full pipeline with `config/8_spearman_improved.yaml` completed; detailed comparison vs combo_0033 in
`output/8_spearman_surrogate/improved_02-27/ANALYSIS_01.md`. Summary:

- **Model A is fixed and now the strongest single model**: test Spearman −0.661 → **+0.760**, NDCG@4 0.00 → 0.53, and the actual champion is ranked **2nd** (old: 7th, outside top 4). Meta coefficients on Model A flipped to strongly positive (E: 19.0, W: 17.8), and the standings column earns real weight (≈12–14).
- **Top-of-table ensemble metrics improved a lot**: final-four NDCG@4 0.04 → 0.46, champion-in-top-4 now true, championship-odds Brier slightly better.
- **Two more bugs found and fixed while chasing a Model B regression** (test Spearman had dropped 0.556 → 0.070):
  1. `standing_rank_norm` (added to team-context features on 02-19, *after* combo_0033 was trained) is the rank of Model B's own training target (win rate to date); the retrained XGB gave it 0.61 importance and degenerated into a standings echo. Fixed with `model_b.exclude_features: ["standing_rank_norm"]` in both improved configs — the standings signal reaches the meta explicitly via improvement 2 instead.
  2. **Inference feature starvation** (`src/inference/predict.py`): the shared inference feature table kept only each team's first-seen as-of date across season specs, so for the second test season the join matched nothing and Model B's scores were silently **all zeros** (its "ranking" was team-id order). Now every (team_id, as_of_date) pair from every spec is built.
- **After the fixes (run_026)**: Model B restored (0.522 vs old 0.556), Model A +0.566 Spearman with the best top-of-table quality of any model (NDCG 0.816, NDCG@4 0.683). Ensemble Spearman 0.524 still trails the old 0.777 — but the old number partly reflected the meta exploiting the *inverted* Model A, and the current 5-col meta underperforms its own components (A's OOF-to-inference scale mismatch is the prime suspect). Model A also shows high run-to-run variance (0.760 vs 0.566 across identical retrains).
- **Top-weighted variant (improvement 3) evaluated and rejected**: `weighted_spearman_surrogate` with power 1.0 was worse on every metric (ensemble Spearman 0.357 vs 0.524, Model A NDCG@4 0.02 vs 0.68, champion rank 15 vs 6). With only 250 training lists, concentrating loss mass on the top ~4 teams per list starves the training signal. Keep the plain `spearman_surrogate`. Details: `output/8_spearman_surrogate/improved_topweighted_02-27/ANALYSIS_01.md`.
- **Flag ablation (improvement 4) complete — keep all three flags off.** The 8-combo sweep (`outputs/sweeps/20260703_134837`) confirmed the current config (combo_0000, all off) as best: `team_rolling` hurts Model B outright (0.655 → 0.582, → 0.506 with injury), `injury` slightly hurts (→ 0.644), `sos_srs` is neutral for B and never helps the ensemble. Details: `output/8_spearman_surrogate/improved_02-27/ANALYSIS_02.md`. (Three bugs found and fixed on the way: `Path` shadowing in `team_context.py`, `_project_minutes` missing the `minutes_heuristic` kwarg, and the inference feature-cache key omitting feature flags.)
- **Next**: stabilize Model A (seed averaging / more lists), revisit meta scale handling (rank-transform meta inputs), and fix a late-season snapshot bug in `build_team_context_as_of_dates` (corrupted eFG/pace on a 2024-04-14 inference snapshot).
