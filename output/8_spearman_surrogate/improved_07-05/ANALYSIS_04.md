# Analysis 04 — Expanded test set with 2025-26 (`run_026_07-06_0025`)

**Run:** `output/8_spearman_surrogate/improved_07-05/outputs/run_026_07-06_0025`  
**Config:** `config/8_spearman_improved.yaml` (rolling/injury **off**; same production settings as `improved_07-03`)  
**Test seasons:** `2023-24`, `2024-25`, **`2025-26`** (9 pooled checkpoints: 6 historical + 3 for 2025-26)  
**Baseline:** `improved_07-03` / `run_025_07-03_2349` (6 checkpoints, playoff outcome rank)

---

## Executive summary

Adding **2025-26** does **not** change performance on the original test seasons — all six historical checkpoints match `improved_07-03` to three decimals (pooled Spearman **0.750**).

The **9-checkpoint pooled headline (Ens Spearman 0.817)** looks much better but is **not comparable** to production **0.750**: 2025-26 evaluation uses **`eos_rank_source: standings`** (playoffs incomplete), so the target is essentially current W/L standing rank, not Playoff Outcome Rank. That inflates correlation and makes the ensemble look worse than standings on MAE in the combined pool.

**Production best for playoff-outcome prediction remains `improved_07-03` on 2023-24 + 2024-25.** Treat 2025-26 as a **live standings sanity check**, not a playoff-strength benchmark, until EOS playoff ranks exist.

---

## Headline metrics (do not mix pools)

| Pool | Checkpoints | Ens Spearman | Ens NDCG@4 | Ens MAE | Standings MAE | Notes |
|------|-------------|--------------|------------|---------|---------------|-------|
| **Production (07-03)** | 6 | **0.750** | 0.258 | 4.10 | 3.97 | Playoff outcome rank |
| **Same model seasons (07-06)** | 6 | **0.750** | 0.258 | 4.10 | 3.97 | Identical to 07-03 |
| 07-06 with 2025-26 | 9 | 0.817 | 0.808 | 3.44 | 2.64 | **Misleading** — includes standings-proxy season |

---

## Per-checkpoint ensemble Spearman (07-06 vs 07-03)

| Checkpoint | 07-03 | 07-06 | Δ |
|------------|-------|-------|---|
| 2023-24 (final) | 0.761 | 0.761 | 0 |
| 2023-24 @ 2024-01-19 | 0.706 | 0.706 | 0 |
| 2023-24 @ 2024-03-01 | 0.712 | 0.712 | 0 |
| 2024-25 (final) | 0.844 | 0.844 | 0 |
| 2024-25 @ 2025-01-18 | 0.697 | 0.697 | 0 |
| 2024-25 @ 2025-02-28 | 0.778 | 0.778 | 0 |
| **2025-26 (final)** | — | **0.956** | standings target |
| 2025-26 @ 2025-12-07 | — | 0.946 | standings target |
| 2025-26 @ 2025-12-27 | — | 0.955 | standings target |

---

## Champion / #1 team (final snapshots)

| Season | Actual #1 (target) | Ensemble pred | Model B pred |
|--------|-------------------|---------------|--------------|
| 2023-24 | Boston Celtics | **#1** ✓ | **#1** ✓ |
| 2024-25 | Oklahoma City Thunder | #2 | **#1** ✓ |
| 2025-26 | OKC (standings proxy) | **#1** ✓ | — (no champion eval yet) |

2024-25 ensemble ranks Cleveland #1 (miss); Model B still picks OKC.

---

## 2025-26 ensemble top 10 (final snapshot, standings target)

| Pred rank | Team | Target rank (standings) |
|-----------|------|-------------------------|
| 1 | Oklahoma City Thunder | 1 |
| 2 | Boston Celtics | 5 |
| 3 | Denver Nuggets | 4 |
| 4 | San Antonio Spurs | 3 |
| 5 | Detroit Pistons | 2 |
| 6 | Minnesota Timberwolves | 9 |
| 7 | Los Angeles Lakers | 8 |
| 8 | Houston Rockets | 6 |
| 9 | New York Knicks | 10 |
| 10 | Phoenix Suns | 7 |

OKC at #1 aligns with the standings proxy. Middle-of-board ordering (ranks 2–10) still diverges — high pooled Spearman partly reflects getting the top team right on an easy target, not deep playoff ordering.

---

## Model vs standings (9-checkpoint pool — interpret with caution)

On the combined 9-checkpoint pool, **ensemble MAE (3.44) is worse than standings MAE (2.64)**; bootstrap p-value = 1.0 (no evidence ensemble beats standings). This is expected when one-third of checkpoints use standings as both feature and target.

---

## Recommendations

1. **Report playoff metrics on 2023-24 + 2024-25 only** until 2025-26 playoffs finish and `eos_final_rank` is available.
2. **Re-run inference** on 2025-26 after playoffs; compare to playoff outcome rank, not standings.
3. **Keep production artifacts** at `improved_07-03/outputs` — this run confirms no regression on historical test seasons.

---

## Artifacts

- `outputs/run_026_07-06_0025/eval_report.json` — full pooled (9 ckpt) report
- `outputs/run_026_07-06_0025/eval_report_2025-26.json` — standings-proxy season
- `outputs/run_026_07-06_0025/predictions_2025-26.json` — live-season predictions
