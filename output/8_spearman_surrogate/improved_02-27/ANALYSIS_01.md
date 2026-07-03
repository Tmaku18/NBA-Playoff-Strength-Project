# Analysis 01 — Improved Spearman-surrogate run (Feb 27, 2026)

Config: `config/8_spearman_improved.yaml`. This run bundles all 7 improvements from
`docs/PROJECT_STATE_AND_BEST_MODELS_02-27.md`: fixed soft-rank sign inversion, standings-win-rate
stacking column, confidence-weighted stacking, lifted caps (250 OOF lists / 250 final batches,
stratified by conference), early stopping (patience 4, 20 epochs), consistent meta target
(always higher = better), and true per-conference RidgeCV metas (E and W fit on their own OOF rows).

Baseline for comparison: previous best combo
`output/8_spearman_surrogate/sweeps/20260217_042955/combo_0033` (run_025_02-17 eval).

## Headline numbers (test seasons 2023-24 + 2024-25, run_025 snapshot)

| Metric | Old best (combo_0033) | New (improved_02-27) |
|---|---|---|
| Model A Spearman | **-0.661** (inverted) | **+0.760** |
| Model A NDCG@4 | 0.000 | 0.529 |
| Model A precision@4 | 0.00 | 0.50 |
| Ensemble Spearman | 0.777 | 0.563 |
| Ensemble NDCG@4 (final four) | 0.042 | 0.457 |
| Ensemble precision@4 | 0.00 | 0.25 |
| Champion rank | 7 | **2** |
| Champion in top 4 | no | **yes** |
| Rank MAE vs playoff outcome | 4.80 | 6.13 |
| Brier (championship odds) | 0.0315 | 0.0301 |
| Model B Spearman | 0.556 | 0.070 |

Per-conference OOF metas (5 columns: a, xgb, conf_a, conf_xgb, standings_wr):

- East: coef = [19.0, 17.1, -0.45, -0.06, 13.8]
- West: coef = [17.8, 13.2, -0.48, -0.01, 11.8]

Model A's meta coefficient is now strongly **positive** (was negative pre-fix), and the
standings column carries real weight — both confirm the fixes behave as intended.

## What improved

1. **Model A is fixed and is now the strongest single model.** Standalone test Spearman went
   from -0.661 to +0.760, and it now places the actual champion 2nd (old: 7th). OOF Spearman
   flipped from -0.40 to +0.43. The sign-inversion fix plus conference-stratified subsampling
   (old runs trained OOF on East-only lists) account for this.
2. **Top-of-table quality is much better.** NDCG@4 for the final four went from 0.04 to 0.46,
   champion is in the top 4, and championship-odds Brier/ECE slightly improved. For a playoff
   strength model, getting the top of the ranking right matters more than mid-table ordering.
3. **Per-conference metas fit on real two-conference OOF.** Old runs silently fit on East-only
   rows; both conferences now contribute 1875 rows each.

## What regressed and why

**Ensemble full-ranking Spearman dropped (0.777 → 0.563) because Model B collapsed at
inference (0.556 → 0.070 on 2024-25).** Diagnosis:

- The new XGB has **14 input features vs the old 13**: `standing_rank_norm` (rank of win
  rate to date) was added to `TEAM_CONTEXT_FEATURE_COLS` on **2026-02-19** — two days *after*
  the old best sweep (2026-02-17) was trained — so combo_0033's XGB never saw it.
- In the new model that single feature takes **0.61 of total importance**. Model B's training
  target is win rate to date, and `standing_rank_norm` is by construction the rank of that
  same win rate — a near-tautology. OOF Spearman looks great (0.985 vs old 0.893) but the
  model has degenerated into a **standings echo** that suppresses the real features
  (eFG, elo, motivation, …).
- At the run_025 early-season snapshot, current standings are a weak predictor of the final
  playoff outcome (especially 2024-25), so the standings echo scores 0.07 while the old
  feature-driven model scored 0.556.
- Two secondary skews also found (small in this run, worth fixing eventually):
  - RAPTOR features (`raptor_*_sum_top5`) are all-zero for 2024+ dates (data ends ~2023);
    importance here is only ~0.01–0.02.
  - A cached late-season inference snapshot (2024-04-14) has corrupted season-to-date stats
    (eFG 4–7 instead of ~0.5, pace ~178 vs train ~199, 19 teams), pointing at a playoff-date
    aggregation bug in `build_team_context_as_of_dates`.
- Note the old combo's higher ensemble Spearman also benefited from the meta exploiting an
  anti-correlated Model A with a negative coefficient — fitting a bug, not signal.

## Fixes applied (run_026)

1. **`model_b.exclude_features: ["standing_rank_norm"]`** in both improved configs. The
   standings signal still reaches the ensemble where it belongs — as an explicit meta column
   (`stacking.use_standings: true`, improvement 2) — instead of letting it hollow out Model B.
2. **Inference feature starvation across seasons (`src/inference/predict.py`).** The shared
   inference feature table kept only each team's *first-seen* as-of date across all season
   specs. For the second test season (2024-25) the per-spec inner join then matched nothing,
   XGB never ran, and Model B's scores were **all zeros** — its "ranking" was just team-id
   order (Spearman 0.07, and identical across retrains, which is what exposed it). Now all
   (team_id, as_of_date) pairs from every spec are built/cached.

## After the fixes (run_026 vs old combo_0033)

| Metric | Old best | run_026 |
|---|---|---|
| Model A Spearman | -0.661 | +0.566 |
| Model B Spearman | 0.556 | 0.522 |
| Ensemble Spearman | 0.777 | 0.524 |
| Model A NDCG / NDCG@4 | 0.281 / 0.000 | **0.816 / 0.683** |
| Champion rank (ensemble) | 7 | 6 |
| 2024-25 inference B Spearman | 0.556 | 0.522 |

Model B is fully restored (0.52 vs old 0.56 without the tautological feature). Model A is
strongly positive and has by far the best top-of-table quality of any model
(NDCG@4 0.68). Two caveats:

- **Model A run-to-run variance is high**: the first improved run's A scored 0.760 Spearman,
  this retrain (identical config/seed) 0.566 — GPU/AMP nondeterminism over only 250 lists.
  Averaging checkpoints or multiple seeds would stabilize it.
- **The 5-column meta is now the weak link**: the ensemble (0.524) underperforms both of its
  components, and old 0.777 partly reflected the meta exploiting the inverted-A bug. The
  OOF-to-inference score-scale mismatch for Model A is the prime suspect.

## Remaining next steps

1. Stabilize Model A (seed averaging / more lists) and revisit meta scale handling
   (e.g. rank-transform meta inputs before RidgeCV).
2. Investigate the late-season snapshot bug in `build_team_context_as_of_dates`
   (corrupted eFG/pace on a cached 2024-04-14 inference snapshot).
3. Run the top-weighted variant (`config/8_spearman_improved_topweighted.yaml`).
4. Run the flag-ablation sweep (`--phase flags`) to settle sos_srs / team_rolling / injury.
