# Analysis 02 — Pipeline deep-dive retrain (`improved_07-03`)

**Run:** `output/8_spearman_surrogate/improved_07-03/outputs/run_025_07-03_2349`  
**Config:** `config/8_spearman_improved.yaml` (post deep-dive fixes; `team_rolling` and `injury` still **off** in this run)  
**Eval:** Pooled across **6 checkpoints** (2 test seasons × 3 as-of dates); fair standings baseline (`standings_to_date_rank`); EOS source `eos_final_rank`.

---

## Executive summary

First full retrain after the pipeline deep-dive (season-scoped features, causal Massey/SRS, causal OOF, seed averaging, rank-transform meta, fair evaluation). Under honest metrics the ensemble reaches **Spearman 0.750** pooled — up from **~0.52** on the pre-fix `improved_02-27/run_026` baseline. Model B is now the primary signal (**0.76** Spearman); Model A contributes via stacking but is weaker alone (**0.55**) on clean features. The ensemble **does not beat** the fair standings baseline on pooled MAE (4.10 vs 3.97; p=0.85); Model B alone is closest (+0.03 MAE).

Flag ablation (sweep `20260703_151849`) identifies **`team_rolling=true` + `injury=true`** as the best feature combo (Spearman 0.824 on older single-snapshot eval). Those flags are **enabled in config** for the next run (`improved_07-05`).

---

## Headline comparison

| Run | Eval scope | Ensemble Spearman | Model A | Model B | Standings MAE (fair?) |
|-----|------------|-------------------|---------|---------|------------------------|
| **improved_07-03** | 6 checkpoints pooled | **0.750** | 0.547 | **0.760** | **3.97** (standings-to-date) |
| improved_02-27 / run_026 | Last season snapshot | 0.524 | 0.566 | 0.522 | 3.13 (EOS standings — unfair) |
| improved_02-27 / run_025_07-03_0451 | Last season snapshot | 0.563 | 0.760* | 0.070* | 3.13 (unfair) |
| combo_0033 (Feb sweep) | Single snapshot | ~0.80 | — | — | 3.13 (unfair + corrupted features) |

\* run_025 Model B at 0.07 was the `standing_rank_norm` tautology bug; Model A at 0.76 was inflated by all-franchise-history feature corruption.

---

## Per-checkpoint ensemble metrics

| Checkpoint | Spearman | NDCG@4 | Ens MAE | Standings MAE | Model − standings (MAE) |
|------------|----------|--------|---------|---------------|-------------------------|
| 2023-24 (final) | 0.761 | 0.845 | 3.87 | 3.80 | −0.07 |
| 2023-24 @ 2024-01-19 | 0.706 | 0.676 | 4.93 | 4.33 | −0.60 |
| 2023-24 @ 2024-03-01 | 0.712 | 0.760 | 4.07 | 4.20 | +0.13 |
| **2024-25 (final)** | **0.844** | 0.444 | 3.47 | 3.27 | **+0.20** |
| 2024-25 @ 2025-01-18 | 0.697 | 0.376 | 4.47 | 4.33 | −0.13 |
| 2024-25 @ 2025-02-28 | 0.778 | 0.461 | 3.80 | 3.87 | +0.07 |

**Patterns:**

- End-of-season snapshots rank best (0.76–0.84 Spearman); mid-season is harder (~0.70).
- 2024-25 outperforms 2023-24 (possibly cleaner RAPTOR carry-forward / data completeness).
- High global Spearman can coexist with weak **NDCG@4** (2024-25 final: 0.84 Spearman but 0.44 NDCG@4) — good global ordering, weak Final Four cutoffs.
- Under fair baseline, the model does not consistently beat standings on MAE.

---

## Component roles

| Component | Pooled Spearman | Notes |
|-----------|-----------------|-------|
| Model A (Deep Set, 3-seed avg) | 0.547 | Weaker alone on season-scoped features; still blended in meta |
| Model B (XGB) | **0.760** | Primary signal after excluding `standing_rank_norm` from B features |
| Meta (3 cols: A, B, standings) | — | Global coef ≈ `[6.4, 9.9, 9.7]`; per-conf E/W metas all positive on A/B/standings |
| Standings baseline | — | Strong anchor; hard to beat on MAE |

**Conference split (pooled):** East Spearman **0.89** vs West **0.66** — West is the harder conference to rank.

---

## Model vs fair standings (pooled)

| Source | MAE vs playoff outcome | RMSE | Δ MAE vs standings |
|--------|------------------------|------|---------------------|
| W/L standings (to-date) | **3.967** | 6.082 | — |
| Ensemble | 4.100 | 6.126 | −0.133 |
| Model B | **3.933** | 5.995 | **+0.033** |
| Model A | 6.389 | 8.236 | −2.422 |

Bootstrap vs standings (ensemble): p=0.85 — not significant.

---

## What the deep-dive fixes changed

1. **Fair standings baseline** — MAE rose from 3.13 (full-season EOS standings) to 3.97 (standings at model as-of date). The old “standings beats the model” story was partly an eval artifact.
2. **Model B fixed** — Season-scoped features + dropping `standing_rank_norm` from Model B inputs (standings reach the ensemble via stacking instead).
3. **Model A Spearman dropped** — Expected; previously learned corrupted all-franchise-history signals.
4. **Ensemble stability** — Std of Spearman across checkpoints **0.056** vs **0.140** for run_026.
5. **Stacking simplified** — 3-column rank-transform meta (A, B, standings); confidence columns removed; NaN targets dropped instead of mean-imputed.

---

## Flag ablation (sweep `20260703_151849`)

| Combo | team_rolling | injury | sos_srs | Spearman* |
|-------|--------------|--------|---------|-----------|
| **0003** | ✓ | ✓ | ✗ | **0.824** |
| 0002 | ✓ | ✗ | ✗ | 0.567 |
| 0000 | ✗ | ✗ | ✗ | 0.520 |

\*Single end-of-season snapshot, old standings baseline (3.13 MAE) — not directly comparable to pooled 0.750. Still, rolling + injury is the clear winner; **enabled in config for `improved_07-05`**.

---

## Playoff / calibration

- **Brier (champion odds):** 0.030 (reasonable)
- **Champion in top 4 (pooled):** 0.0 — champion identification still weak
- **Precision@4 (pooled):** 0.0

---

## Next run

**Production:** keep `improved_07-03` artifacts; config flags `team_rolling` / `injury` **off**.

**Completed follow-up:** `improved_07-05` (rolling + injury) — no ensemble gain; see [improved_07-05/ANALYSIS_03.md](../improved_07-05/ANALYSIS_03.md).

**Suggested experiments:** `config/8_spearman_improved_topweighted.yaml`, West-conference analysis, champion/top-4 calibration.

---

## Artifacts

- `outputs/run_025_07-03_2349/eval_report.json` — full pooled metrics
- `outputs/run_025_07-03_2349/eval_report_<season>.json` — per-checkpoint reports
- `outputs/run_025_07-03_2349/ANALYSIS_01.md` — auto-generated eval summary (script 5)
- `outputs/ridgecv_meta*.joblib` — 3-column rank-transform metas
