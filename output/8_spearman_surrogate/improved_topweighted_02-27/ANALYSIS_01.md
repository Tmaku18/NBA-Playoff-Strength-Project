# Analysis 01 — Top-weighted Spearman surrogate (improvement 3)

Config: `config/8_spearman_improved_topweighted.yaml` — identical to
`config/8_spearman_improved.yaml` (all fixes included: sign fix, standings + confidence
stacking, `standing_rank_norm` excluded from Model B, per-conference metas, lifted caps)
except `training.loss_type: weighted_spearman_surrogate` with `loss_top_weight_power: 1.0`
(per-team weights ∝ 1/actual_rank, up-weighting agreement on the top of the list).

## Result: the top-weighted loss is worse on every axis (test 2023-24 + 2024-25)

| Metric | improved (plain surrogate, run_026) | top-weighted |
|---|---|---|
| Ensemble Spearman | 0.524 | 0.357 |
| Model A Spearman | 0.566 | 0.447 |
| Model A NDCG / NDCG@4 | 0.816 / 0.683 | 0.363 / 0.021 |
| Champion rank (ensemble) | 6 | 15 |
| Rank MAE vs playoff outcome | 6.73 | 8.00 |
| Brier (championship odds) | 0.0312 | 0.0329 |
| Model B Spearman | 0.522 | 0.522 (same model) |

The meta also trusts the top-weighted Model A far less (East A-coefficient 5.97 vs 18.29
for the plain loss).

## Interpretation

- The hypothesis was that up-weighting top ranks would sharpen final-four quality. The
  opposite happened — including at the top (NDCG@4 0.02 vs 0.68). With only 250 training
  lists, concentrating gradient mass on ~4 teams per list effectively shrinks the training
  signal, and the model generalizes worse everywhere.
- Given Model A's known run-to-run variance (0.76 vs 0.57 across identical plain-loss
  retrains), part of the gap may be noise, but the deficit is consistent across every metric
  and the low meta coefficient independently confirms weaker OOF quality.

## Verdict

Keep `spearman_surrogate` (plain) as the loss for the improved model. Do not pursue
`loss_top_weight_power: 1.0`; if revisiting, try milder weighting (e.g. 0.25–0.5) only after
Model A is stabilized (seed averaging / more lists).
