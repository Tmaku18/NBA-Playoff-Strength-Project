# Analysis 02 — Post-fix pipeline status (`improved_07-06`)

**Date:** 2026-07-06  
**Latest complete eval:** `run_025_07-06_0141` (invalid — 19 teams)  
**Post-fix run started:** `run_026_07-06_0439` (inference never finished — no run folder)  
**Production baseline:** `improved_07-03` / `run_025_07-03_2349`

---

## Executive summary

The **19-team bug is fixed in code and DB** (`src/data/team_meta.py`, `teams.conference` backfilled — 15 E / 15 W, modern abbreviations). Updated `split_info.json` now shows **62 train / 24 test lists** (≈2 conference lists × 31 train dates), consistent with full 30-team coverage.

However, **no valid post-fix eval exists yet.** The only completed inference is `run_025_07-06_0141`, which was built before the fix. A re-run allocated `run_026_07-06_0439` (`.current_run`) but did not produce predictions or a `run_026_*` folder.

**Continue using `improved_07-03` for reporting** until the WSL pipeline completes end-to-end.

---

## Run status

| Artifact | Timestamp | Status |
|----------|-----------|--------|
| `teams` DB (30 conf, 0 NULL) | backfilled | ✓ fixed |
| `split_info.json` (62/24 lists) | 07-06 04:40 | ✓ 30-team lists |
| `.current_run` → `run_026_07-06_0439` | 07-06 04:39 | started, incomplete |
| `run_025_07-06_0141` eval | 07-06 03:58 | complete but **19-team bug** |
| `run_026_*` folder | — | **missing** |

---

## Latest complete metrics (`run_025` — not comparable)

| Run | Teams | Ens Spearman | NDCG@4 | Ens MAE | Valid? |
|-----|-------|--------------|--------|---------|--------|
| **07-03 prod** | 30 | **0.750** | 0.258 | 4.10 | ✓ |
| 07-06 run_025 | **19** | 0.770 | 0.522 | 4.07 | ✗ |

Inflated 07-06 NDCG@4 and wrong 2024-25 champion (Indiana vs OKC) are artifacts of the missing-team bug.

---

## Per-checkpoint (`run_025` vs prod)

| Checkpoint | 07-03 (n=30) | 07-06 run_025 (n=19) |
|------------|--------------|----------------------|
| 2023-24 final | 0.761 / champ **#1** | 0.753 / champ **#1** |
| 2024-25 final | 0.844 / champ **#2** (OKC) | 0.775 / champ **#7** (OKC missing) |
| 2025-26 final | — | 0.798 / champ **#4** (NYK actual #1) |

2025-26 is the first eval on real `eos_final_rank` for that season, but still on a 19-team subset.

---

## Next step

Re-run the full pipeline in WSL and verify:

```bash
python - <<'PY'
import json, glob, os
run=sorted(glob.glob("output/8_spearman_surrogate/improved_07-06/outputs/run_*"), key=os.path.getmtime)[-1]
for f in sorted(glob.glob(run+"/predictions_*.json")):
    d=json.load(open(f)); print(os.path.basename(f), "teams=", len(d["teams"]))
PY
```

Expect `teams= 30` on every file and 2024-25 actual champion = Oklahoma City Thunder.
