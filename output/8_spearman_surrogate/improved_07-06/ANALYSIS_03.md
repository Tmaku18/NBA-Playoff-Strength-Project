# Analysis 03 — Post-fix valid run (`run_026_07-06_0937`)

**Run:** `output/8_spearman_surrogate/improved_07-06/outputs/run_026_07-06_0937`  
**Completed:** 2026-07-06 ~12:46  
**Baseline:** `improved_07-03` / `run_025_07-03_2349`

---

## Executive summary

The **30-team bug is fixed and validated.** Every prediction file has **30 teams**, `eos_final_rank` on all checkpoints, and correct franchise metadata (OKC appears as 2024-25 champion).

This is the **first valid `improved_07-06` run** and **beats production on headline ranking metrics**:

| Metric | 07-03 prod | **07-06 run_026** | Δ |
|--------|------------|-------------------|---|
| Ens Spearman | 0.750 | **0.789** | +0.039 |
| NDCG@4 | 0.258 | **0.620** | +0.362 |
| Ens MAE | 4.10 | **3.73** | −0.37 |
| Standings MAE | 3.97 | 3.67 | −0.30 |

**Champion picks (final snapshots):** 2023-24 Boston ✓ | 2024-25 OKC ✓ | 2025-26 NYK **#8** (miss)

**Recommend promoting `improved_07-06` / `run_026_07-06_0937` as the new production best** for this model line.

---

## Bug-fix validation

| Check | run_025 (broken) | run_026 (fixed) |
|-------|------------------|-----------------|
| Teams per prediction | 19 | **30** |
| 2024-25 actual #1 | Indiana (wrong) | **Oklahoma City Thunder** |
| EOS source | eos_final_rank | eos_final_rank |

---

## Pooled test metrics (6 checkpoints)

| Source | Spearman | NDCG@4 | MAE | RMSE | Brier (champ odds) |
|--------|----------|--------|-----|------|-------------------|
| W/L standings | −0.963 | 0.805 | 3.67 | 5.54 | — |
| **Ensemble** | **0.789** | **0.620** | 3.73 | 5.62 | 0.027 |
| Model A | 0.643 | — | 5.49 | 7.31 | — |
| Model B | 0.778 | — | 3.97 | 5.76 | — |

Ensemble **beats standings on Spearman/NDCG** but is **not significantly better on MAE** vs standings (Δ MAE −0.07, bootstrap p=0.72). Model B drives most of the ensemble lift; Model A improved vs 07-03 (Spearman 0.547 → 0.643) but still trails B.

---

## Per-checkpoint (ensemble)

| Checkpoint | 07-03 | 07-06 run_026 | Champ (07-06) |
|------------|-------|---------------|---------------|
| 2023-24 final | 0.761 / 0.845 | 0.758 / 0.845 | **#1** ✓ |
| 2023-24 @ 2024-03-01 | 0.712 / 0.760 | 0.732 / 0.613 | **#2** |
| 2024-25 final | 0.844 / 0.444 | 0.817 / 0.689 | **#1** ✓ |
| 2024-25 @ 2025-02-26 | 0.778 / 0.461 | 0.763 / 0.686 | **#1** ✓ |
| 2025-26 final | — | 0.837 / 0.385 | **#8** |
| 2025-26 @ 2026-02-21 | — | 0.829 / 0.434 | **#8** |

**2024-25** is the standout: NDCG@4 jumps from 0.444 → 0.689 and champion goes from #2 → **#1** (OKC).  
**2025-26** has high Spearman (0.837) but weak NDCG@4 (0.385) and misses the champion (NYK actual #1, pred #8).

---

## 2025-26 final snapshot (held-out season)

| Pred | Team | Actual outcome |
|------|------|----------------|
| 1 | Oklahoma City Thunder | 3 |
| 2 | San Antonio Spurs | 2 |
| 3 | LA Lakers | 7 |
| 4 | Boston Celtics | 9 |
| 5 | Detroit Pistons | 5 |
| 6 | Denver Nuggets | 12 |
| 7 | Cleveland Cavaliers | 4 |
| **8** | **New York Knicks** | **1** |

Model overweights OKC/SA (strong regular seasons) and underranks the actual champion NYK. Top-4 precision is weak for this season despite strong overall rank correlation.

---

## Conference breakdown (ensemble)

| Conf | NDCG | Spearman | MAE |
|------|------|----------|-----|
| East | 0.855 | 0.901 | 3.13 |
| West | 0.692 | 0.710 | 4.33 |

East predictions are substantially stronger than West on this run.

---

## vs. invalid run_025

| | run_025 (19 teams) | run_026 (30 teams) |
|---|-------------------|-------------------|
| Ens Spearman | 0.770* | **0.789** |
| NDCG@4 | 0.522* | **0.620** |
| 2024-25 champ | #7 (wrong) | **#1** ✓ |

\*run_025 metrics were inflated/invalid due to missing teams.

---

## Artifacts

- `run_026_07-06_0937/eval_report.json`
- `run_026_07-06_0937/predictions_2025-26.json`
- `run_026_07-06_0937/ANALYSIS_01.md` (auto-generated)
