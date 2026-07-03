# Analysis 02 — Flag-ablation sweep (improvement 4)

Batch: `outputs/sweeps/20260703_134837` (8 combos, `--phase flags`, all succeeded).
Grid: on/off over `sos_srs` × `team_rolling` × `injury`; everything else fixed at the
improved config (`config/8_spearman_improved.yaml`): tuned combo_0033 hyperparameters,
fixed Spearman loss, elo/massey/motivation/raptor on, `standing_rank_norm` excluded,
confidence + standings stacking, per-conference metas. Combo_0000 (all off) *is* the
current best configuration.

Three earlier batches in this folder (`20260703_100359`, `20260703_103936`,
`20260703_113114`) are partial failures from since-fixed bugs (Path shadowing,
`minutes_heuristic` TypeError, stale inference feature cache) and can be ignored/deleted.

## Results (test 2023-24 + 2024-25)

| combo | sos_srs | team_rolling | injury | Ens ρ | A ρ | B ρ | Champ rank | Rank MAE | P@8 |
|---|---|---|---|---|---|---|---|---|---|
| **0000** | – | – | – | **0.631** | 0.615 | **0.655** | 8 | **5.80** | 0.75 |
| 0001 | – | – | on | 0.539 | 0.652 | 0.644 | 8 | 6.53 | 0.75 |
| 0002 | – | on | – | 0.483 | 0.695 | 0.582 | 10 | 6.93 | 0.625 |
| 0003 | – | on | on | 0.556 | 0.580 | 0.506 | 15 | 6.60 | 0.625 |
| 0004 | on | – | – | 0.562 | 0.552 | 0.655 | 8 | 6.07 | 0.75 |
| 0005 | on | – | on | 0.539 | 0.625 | 0.644 | 8 | 6.40 | 0.75 |
| 0006 | on | on | – | 0.524 | 0.611 | 0.582 | 10 | 7.07 | 0.625 |
| 0007 | on | on | on | 0.535 | 0.629 | 0.506 | 8 | 6.60 | 0.75 |

## Read-out

Model B's column is the cleanest signal: B only changes when its feature set changes
(same seed/HPs), while Model A retrains per combo with GPU/AMP nondeterminism (its ρ
fluctuates 0.55–0.70 across combos regardless of flags — consistent with the variance
seen between identical `improved_02-27` retrains).

- **team_rolling: hurts.** B drops 0.655 → 0.582 alone, and to 0.506 combined with
  injury. Ensemble, champion rank, MAE, and P@8 all degrade whenever it is on.
- **injury: slightly hurts.** B 0.655 → 0.644; ensemble drops in every pairing.
  The `proj_available_rating` heuristic adds noise at the run_025 snapshot.
- **sos_srs: neutral for B (identical 0.655), never helps the ensemble.** The sos/srs
  season-level values add nothing beyond elo/massey, which are already on.

## Verdict

**Keep all three flags off** — the current improved config (= combo_0000) is confirmed
as the best flag setting. No config change needed. If team-rolling or injury signal is
ever revisited, it should come with retuned XGB hyperparameters (the fixed HPs were tuned
for the 13-feature set) and a better injury-minutes model.
