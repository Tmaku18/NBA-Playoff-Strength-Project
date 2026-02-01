---
name: Walk-Forward + EOS Final Rank (Option B)
overview: "Implement per-season walk-forward training, EOS final rank (Option B), EORS rank in outputs, EORS vs EOS_global_rank graph, and per-season inference/evaluation for all test years."
todos: []
isProject: false
---

# Combined: Per-Season Walk-Forward Training + EOS Final Rank (Option B)

This plan merges three features:

1. **Per-season walk-forward training** for Model A (expand training set season by season, validate on next unseen season)
2. **EOS final rank for validation** via **Option B**: when playoff data exists for the target season, set `EOS_global_rank` in predictions to the EOS final rank instead of standings order. Script 5 unchanged; it reads `EOS_global_rank` and evaluates against it.
3. **EORS rank (End of Regular Season = playoff standings)** in outputs: add `analysis.EORS_rank` to each team — the final regular-season standings rank (1–30 by win %), which determines playoff seeding.

---

## Part A: Per-Season Walk-Forward Training

### A.1 Season utilities in split module

**File:** [src/utils/split.py](src/utils/split.py)

- Export `date_to_season(as_of_date, seasons_cfg) -> str | None` (promote `_date_to_season` or add public wrapper)
- Add `get_train_seasons_ordered(config) -> list[str]`: return `train_seasons` from config, sorted by season start date. Ensures chronological order for walk-forward
- Add `group_lists_by_season(lists, seasons_cfg) -> dict[str, list]`: map season key -> list of lists. Uses `date_to_season` per list

### A.2 Config option

**File:** [config/defaults.yaml](config/defaults.yaml)

Add under `training:`:

```yaml
walk_forward: false   # if true, use per-season walk-forward instead of pooled OOF
```

### A.3 Script 3 walk-forward mode

**File:** [scripts/3_train_model_a.py](scripts/3_train_model_a.py)

When `walk_forward: true`:

1. Compute split and write `split_info.json` (unchanged)
2. Get `train_seasons_ordered` (e.g. `["2015-16", ..., "2022-23"]`)
3. Group `train_lists` by season via `group_lists_by_season`
4. For each step k in 1..N:
   - Train on seasons 1..k, validate on season k+1 (if k < N)
   - Build batches, train via `train_model_a_on_batches`, collect OOF rows `(team_id, as_of_date, oof_a, y)` for validation season
   - Print step summary
5. Last step (k == N): train on all, no next season, save `best_deep_set.pt`
6. Write `oof_model_a.parquet` from accumulated OOF rows

When `walk_forward: false`: keep current pooled OOF + final model behavior.

---

## Part B: EOS Final Rank (Option B)

### B.1 New EOS final rank in playoffs.py

**File:** [src/evaluation/playoffs.py](src/evaluation/playoffs.py)

Add `compute_eos_final_rank(...)` returning `dict[int, int]` (team_id -> rank 1–30):

- **Rank 1:** Champion (most playoff wins; tie-break reg-season win %)
- **Ranks 29–30:** First 2 teams eliminated from playoffs (2 playoff teams with fewest playoff wins; tie-break: worse reg-season record gets 30, then 29)
- **Ranks 2–28:** Remaining teams: other 14 playoff teams (by playoff wins desc, tie-break reg %), then 14 lottery teams (by reg %). Order: champion (1), playoff 2–15, lottery 16–28, first 2 eliminated (29–30)

Reuse `get_playoff_wins`, `get_reg_season_win_pct`, `_filtered_playoff_tgl`. Require at least 16 playoff teams.

### B.2 Option B: Override EOS_global_rank in inference

**File:** [src/inference/predict.py](src/inference/predict.py)

Current flow: `actual_global_rank` is set from standings-to-date (win_rate_map sorted by win rate desc) at line ~315.

**Change:** When playoff data exists for the target season (same logic as existing playoff_rank_map: load playoff data, check 16+ teams):

1. Call `compute_eos_final_rank(pg, ptgl, games, tgl, target_season, ...)` instead of (or in addition to) `compute_playoff_performance_rank`
2. If the result is non-empty (16+ playoff teams), set `actual_global_rank = eos_final_rank_map` instead of the standings-based rank
3. Otherwise: fall back to standings-based `actual_global_rank` (current behavior)

### B.3 Add `eos_rank_source` to predictions for traceability

**File:** [src/inference/predict.py](src/inference/predict.py)

Add a top-level field in the predictions output (or in `notes`): `"eos_rank_source": "eos_final_rank"` or `"standings"`.

### B.4 EORS Rank (End of Regular Season = Playoff Standings) in outputs

**EORS rank** = End of Regular Season rank = **Playoff standings** — the final regular-season standings (1–30 by win % after all reg-season games).

**File:** [src/evaluation/playoffs.py](src/evaluation/playoffs.py)

Add `compute_eors_rank(games, tgl, season, *, season_start, season_end, all_team_ids) -> dict[int, int]`.

**File:** [src/inference/predict.py](src/inference/predict.py)

Add `analysis.EORS_rank` to each team in `predict_teams` output via optional `eors_rank: dict[int, int] | None = None`.

---

## Part C: How Metrics Change and How to Interpret New Outputs

### C.1–C.5 Metrics interpretation

See walk_forward_and_eos_final_rank_combined.plan.md for full details on metric changes, breaking comparison, and per-conference metrics.

### C.6 EORS vs EOS_global_rank graph

Add scatter plot: **X-axis** EORS_rank (playoff standings), **Y-axis** EOS_global_rank (playoff outcome). Output: `outputs/run_NNN/eors_vs_eos_global_rank_{season}.png`.

### C.7 Per-season inference and evaluation (all test years)

**Derive test seasons:** Add `test_seasons` to split_info.json (from test_dates + date_to_season).

**Per-season inference:** For each test season, use last test date in that season, run inference, write `predictions_{season}.json` (e.g. `predictions_2023-24.json`, `predictions_2024-25.json`). Per-season figures: `pred_vs_actual_{season}.png`, `eors_vs_eos_global_rank_{season}.png`, etc.

**Per-season evaluation:** Loop over `predictions_{season}.json`, write `eval_report_{season}.json` per season. Optionally aggregate `eval_report.json`.

**Dynamic:** When config `training.test_seasons` changes, pipeline processes all included test seasons.

---

## Part D–F: Data flow, files to modify, rollback

See walk_forward_and_eos_final_rank_combined.plan.md for full plan details.
