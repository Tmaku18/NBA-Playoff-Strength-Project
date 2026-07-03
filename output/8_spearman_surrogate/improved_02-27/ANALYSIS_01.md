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

- Model B's own OOF quality is *better* than before (0.985 vs 0.893 Spearman vs win rate),
  so training is fine. The failure is train/inference feature skew:
  - **RAPTOR features are all-zero in test seasons.** The FiveThirtyEight RAPTOR data ends
    around 2023; `raptor_offense_sum_top5` / `raptor_defense_sum_top5` have real signal in
    train years and are exactly 0.0 for 2024+ dates. The new XGB (14 features, trained on the
    rebuilt `nba_build_run.duckdb` where the raptor table exists) leans on them; the old model
    (13 features, older DB) could not.
  - A cached inference feature snapshot (2024-04-14) shows corrupted season-to-date stats
    (eFG 4–7 instead of ~0.5, pace ~178 vs train ~199, 19 teams instead of 30), pointing to a
    playoff-date aggregation bug in `build_team_context_as_of_dates` for late-season snapshots.
- The old combo's higher ensemble Spearman also benefited from the meta exploiting an
  anti-correlated Model A with a negative coefficient — that stacking behavior was fitting a
  bug, not signal.

## Recommended next steps

1. **Disable RAPTOR for test-era models** (`raptor.enabled: false` or add
   `raptor_*` to `model_b.exclude_features`) — the columns are guaranteed dead for 2024+ and
   only add train/test skew. Re-run and re-check Model B and ensemble Spearman.
2. Investigate the late-season inference feature snapshot (eFG/pace out of range on
   2024-04-14) in `build_team_context_as_of_dates` — likely playoff games leaking into the
   season-to-date aggregation or a duplicate-row sum.
3. Run the top-weighted variant (`config/8_spearman_improved_topweighted.yaml`) once B is
   fixed — top-of-table metrics are already the strongest axis of improvement.
4. Run the flag-ablation sweep (`--phase flags`) to settle sos_srs / team_rolling / injury.
