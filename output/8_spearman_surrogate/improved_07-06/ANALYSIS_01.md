# Analysis 01 — Full pipeline with correct DB (`improved_07-06`)

**Run:** `output/8_spearman_surrogate/improved_07-06/outputs/run_025_07-06_0141`  
**Config:** `config/8_spearman_improved.yaml` (`nba_build.duckdb`, sklearn 1.8.0 retrain)  
**Baseline:** `improved_07-03` / `run_025_07-03_2349`

---

## Executive summary

The pipeline **completed** and **2025-26 now uses real playoff outcome ranks** (`eos_final_rank` on all checkpoints). That fixes the wrong-DB / standings-proxy issue from `improved_07-05`.

However, this run has a **critical coverage bug: only 19 of 30 teams appear in every prediction file.** Oklahoma City, Golden State, Philadelphia, and 8 others are missing. That corrupts EOS outcome ranks (2024-25 “champion” shows as Indiana, not OKC) and makes pooled metrics **not comparable** to production.

**Production best remains `improved_07-03`** until the 19-team list-building issue is fixed and a full re-run completes with 30 teams.

---

## Headline pooled metrics (6 checkpoints — different dates than 07-03)

| Run | Ens Spearman | NDCG@4 | Ens MAE | Stand MAE | EOS source |
|-----|--------------|--------|---------|-----------|------------|
| **07-03 prod** | **0.750** | 0.258 | 4.10 | 3.97 | eos_final_rank |
| 07-06 new | 0.770 | 0.522 | 4.07 | 4.18 | eos_final_rank |

The 07-06 NDCG@4 jump (0.258 → 0.522) and Spearman bump are **not apples-to-apples**: different checkpoint dates, only 19-team universe, and corrupted outcome ranks for missing teams.

---

## Per-checkpoint metrics (ensemble Spearman / NDCG@4 / champion rank)

| Checkpoint | 07-03 prod (n=30) | 07-06 new (n=19) |
|------------|-------------------|------------------|
| 2023-24 final | 0.761 / 0.845 / **#1** ✓ | 0.753 / 0.923 / **#1** ✓ |
| 2023-24 mid-season | 0.712 @ 2024-03-01 | 0.656 @ 2024-03-01 |
| 2024-25 final | 0.844 / 0.444 / **#2** (OKC) | 0.775 / 0.197 / **#7** (OKC missing) |
| 2024-25 mid-season | 0.778 @ 2025-02-28 | 0.802 @ 2025-02-26 |
| 2025-26 final | — | 0.798 / 0.470 / **#4** (NYK actual #1) |
| 2025-26 mid-season | — | 0.839 / 0.677 / **#4** |

On overlapping finals, 07-06 is **equal or worse** on Spearman; inflated NDCG@4 on 2023-24 is an artifact of the 19-team subset (easier top-4 set).

---

## 2025-26 (first valid playoff-outcome eval)

- **EOS source:** `eos_final_rank` ✓  
- **Actual champion (outcome #1):** New York Knicks  
- **Ensemble pred:** #4 (miss) | **Model B:** #3  
- **Top prediction:** Detroit #1 (actual outcome #3)

| Pred | Team | Outcome rank |
|------|------|--------------|
| 1 | Detroit Pistons | 3 |
| 2 | Boston Celtics | 6 |
| 3 | Denver Nuggets | 9 |
| 4 | **New York Knicks** | **1** |
| 5 | Los Angeles Lakers | 5 |
| 6 | Cleveland Cavaliers | 2 |

`post_playoff_rank` populated for 19/30 teams (only teams in the inference list).

---

## Champion picks (final snapshots)

| Season | Actual #1 | Ensemble | Model B |
|--------|-----------|----------|---------|
| 2023-24 | Boston Celtics | **#1** ✓ | **#1** ✓ |
| 2024-25 | Indiana (wrong — OKC missing from list) | #7 | #6 |
| 2025-26 | New York Knicks | #4 | #3 |

2024-25 champion is wrong because **OKC is not in the 19-team prediction set**, so `compute_eos_final_rank` was scoped to an incomplete `all_team_ids` list.

**Missing teams (11):** Brooklyn, Philadelphia, Charlotte, **Oklahoma City**, Golden State, LA Clippers, Memphis, Sacramento, San Antonio, New Orleans, Utah.

---

## Split / list coverage

| | 07-03 | 07-06 |
|---|-------|-------|
| Train lists | 279 | 60 |
| Test lists | 91 | 24 |
| Teams per prediction | 30 | **19** |

The smaller list counts suggest batch/list building did not cover the full league — likely a bug or cache/DB interaction to investigate in `build_lists` / `build_batches_from_lists`.

---

## Model vs standings (07-06 pool)

Ensemble MAE **4.07** vs standings **4.18** — ensemble slightly better on MAE, but with only 19 teams and wrong outcome ranks this is not reliable.

---

## Recommendations

1. **Investigate why test lists contain only 19 teams** (missing West/East teams including OKC).
2. **Do not promote 07-06 metrics** until predictions include all 30 teams and 2024-25 champion = OKC.
3. **Keep production artifacts at `improved_07-03`** for reporting.
4. After fix: re-run `run_pipeline_from_model_a` into `improved_07-06/outputs` (or new folder).

---

## Artifacts

- `outputs/run_025_07-06_0141/eval_report.json`
- `outputs/run_025_07-06_0141/predictions_2025-26.json` — eos_final_rank ✓
- `outputs/run_025_07-06_0141/ANALYSIS_01.md` — auto-generated eval summary
