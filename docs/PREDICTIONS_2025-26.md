# 2025-26 NBA True Strength Predictions (production best — `improved_07-06`)

**Date generated:** 2026-07-06  
**As-of date:** **2026-04-02** (latest test checkpoint for 2025-26)  
**Model:** Spearman-surrogate Deep Set (Model A) + XGBoost (Model B) + RidgeCV stacking  
**Run:** `output/8_spearman_surrogate/improved_07-06/outputs/run_026_07-06_0937`  
**Predictions file:** `.../run_026_07-06_0937/predictions_2025-26.json`  
**Eval:** `eval_report_2025-26.json` (playoff outcomes in DB; `eos_final_rank`)

Full pipeline retrain with `config/8_spearman_improved.yaml` after the 30-team franchise-metadata fix. All **30 teams** appear in every prediction file.

## Eval summary (2025-26 final)

| Metric | Ensemble | Model B |
|--------|----------|---------|
| Spearman | 0.837 | 0.822 |
| NDCG@4 | 0.385 | 0.398 |
| Champion rank (actual #1 = NYK) | **#8** | **#10** |

Actual playoff champion: **New York Knicks** (outcome #1). Model favored OKC (#1 pred, outcome #3) and San Antonio (#2 pred, outcome #2).

## Predictions (as of 2026-04-02)

### Top 10 overall (ensemble)

| Rank | Team | Conf | Ensemble (0..100) | Championship odds | Actual outcome |
|-----:|------|:----:|------------------:|------------------:|---------------:|
| 1 | Oklahoma City Thunder | W | 100.0 | 32.4% | 3 |
| 2 | San Antonio Spurs | W | 96.6 | 21.9% | 2 |
| 3 | Los Angeles Lakers | W | 93.1 | 14.8% | 7 |
| 4 | Boston Celtics | E | 89.7 | 10.0% | 9 |
| 5 | Detroit Pistons | E | 86.2 | 6.8% | 5 |
| 6 | Denver Nuggets | W | 82.8 | 4.6% | 12 |
| 7 | Cleveland Cavaliers | E | 79.3 | 3.1% | 4 |
| **8** | **New York Knicks** | E | **75.9** | **2.1%** | **1** |
| 9 | Minnesota Timberwolves | W | 72.4 | 1.4% | 6 |
| 10 | Houston Rockets | W | 69.0 | 1.0% | 13 |

### Model B top 5 (for comparison)

| B-rank | Team | Outcome |
|-------:|------|--------:|
| 1 | Oklahoma City Thunder | 3 |
| 2 | San Antonio Spurs | 2 |
| 3 | Boston Celtics | 9 |
| 4 | Detroit Pistons | 5 |
| 5 | Cleveland Cavaliers | 4 |

Model B ranks the champion NYK at **#10**; ensemble at **#8**.

## How this run was produced

1. `2025-26` in `training.test_seasons` (`config/defaults.yaml`, `config/8_spearman_improved.yaml`).
2. Raw 2025-26 RS + playoff logs downloaded; `data/processed/nba_build.duckdb` built (1,230 RS games, 85 playoff games).
3. Franchise metadata fix (`src/data/team_meta.py`) — all 30 teams in conference lists.
4. Full pipeline: `python -m scripts.run_pipeline_from_model_a --config config/8_spearman_improved.yaml` (WSL).

## Companion artifacts

In `run_026_07-06_0937/`:

- `predictions_2025-26.json` — full diagnostics (Model A/B ranks, attention, championship odds)
- `eval_report_2025-26.json` — season eval metrics
- `pred_vs_playoff_outcome_rank_2025-26.png`, `odds_top10_2025-26.png`, `title_contender_scatter_2025-26.png`

## Analysis

- Run-level: [output/8_spearman_surrogate/improved_07-06/ANALYSIS_03.md](../output/8_spearman_surrogate/improved_07-06/ANALYSIS_03.md)
- Auto-generated: `run_026_07-06_0937/ANALYSIS_01.md`

## Caveats

- 2025-26 is a **held-out test season**; high Spearman (0.837) but weak champion/top-4 placement for NYK.
- Prior inference-only doc (Feb 2026, combo_0033, 19 teams) is **obsolete** — do not use `sweeps/.../run_2025-26_inference/` for current reporting.
