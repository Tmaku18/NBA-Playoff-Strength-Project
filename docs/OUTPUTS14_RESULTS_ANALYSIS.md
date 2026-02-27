# outputs14 MAP runs — consolidated results analysis

Analysis of all outputs14 MAP variants: main run, strict playoff eval (full + small DB), and 50y. All eval paths under `outputs14_map_run/`.

---

## 1. Run overview and eval target

| Run | Config | EOS rank source | Train seasons | Test seasons |
|-----|--------|------------------|---------------|--------------|
| **map_run** | outputs14_map_run | **eos_final_rank** (playoff outcome) | 2015-16 … 2022-23 | 2023-24, 2024-25 |
| **map_run_strict_playoff_eval** | strict_playoff_eval | **eos_final_rank** | 2015-16 … 2022-23 | 2023-24, 2024-25 |
| **map_run_strict_playoff_eval_small_db** | strict + small DB | **eos_final_rank** | (same idea) | 2023-24, 2024-25 |
| **map_run_50y** | 50y | **standings** (W/L) | 70/20/10 date split | — |

**Important:** The 50y run uses `eos_rank_source: "standings"`, so its metrics are **vs regular-season order**, not vs playoff outcome. Spearman/RMSE there are not comparable to the other runs.

---

## 2. Playoff-outcome runs (eos_final_rank): main metrics

Test metrics below are **ensemble** vs **playoff outcome rank** (same train/test seasons where applicable).

### 2.1 Main MAP run (`map_run`)

- **Spearman:** 0.27  
- **Rank RMSE:** 10.46  
- **Rank MAE:** 8.20  
- **NDCG@4:** 0.041  
- **Champion rank:** 6  

**Vs W/L standings (same report):** Standings MAE 3.13, RMSE 4.45 → ensemble is **worse** than standings (MAE +5.07, RMSE +6.01).  
See `docs/OUTPUTS14_MAP_ANALYSIS.md` for comparison to outputs8 (MAP better on NDCG/champion, outputs8 better on Spearman/RMSE).

### 2.2 Strict playoff eval (`map_run_strict_playoff_eval`)

- **Spearman:** **-0.12**  
- **Rank RMSE:** ~9.79 (from ANALYSIS_01)  
- **Rank MAE:** ~7.89  
- **NDCG@4:** 0.021  
- **Champion rank:** 12  

**Vs W/L standings:** Standings MAE 4.0, RMSE 6.06 → ensemble is **worse** (MAE +3.89, RMSE +3.73). Bootstrap p-value vs standings: 1.0 (no improvement).  
**Conference:** East Spearman 0.06, West **-0.61** → strong West anti-correlation.

### 2.3 Strict playoff eval — small DB (`map_run_strict_playoff_eval_small_db`)

- **Spearman:** 0.25  
- **Rank RMSE:** 10.58  
- **Rank MAE:** 8.20  
- **NDCG@4:** 0.036  
- **Champion rank:** 12  

**Vs W/L standings:** Standings MAE 3.13, RMSE 4.45 → ensemble **worse** (MAE +5.07, RMSE +6.13).

### 2.4 Side-by-side (playoff-outcome runs only)

| Metric | map_run | strict_playoff_eval | strict_small_db |
|--------|---------|---------------------|------------------|
| Spearman | 0.27 | **-0.12** | 0.25 |
| Rank RMSE | 10.46 | ~9.79 | 10.58 |
| Rank MAE | 8.20 | ~7.89 | 8.20 |
| NDCG@4 | 0.041 | 0.021 | 0.036 |
| Champion rank | **6** | 12 | 12 |
| Better than standings? | No | No | No |

Strict run has **lower** Spearman (even negative) and **worse** champion placement (12 vs 6) despite same target and `require_eos_final_rank: true`. So under strict playoff eval the current MAP setup does not improve over the main run and is clearly worse than W/L standings.

---

## 3. 50y run (`map_run_50y`) — not playoff-outcome

- **eos_rank_source:** `standings` (evaluation is vs regular-season order, not playoff outcome).
- **Ensemble:** Spearman 0.98, Rank RMSE 1.77 (excellent vs standings).
- **Model A:** Spearman **-0.77**, Rank RMSE 16.29 (inverted; ListMLE trained on playoff outcome is not fitting standings order).
- **Model B (XGB):** Spearman 0.99, RMSE 1.15 (driving the ensemble).

So 50y is **not** a valid playoff-outcome evaluation; it only shows that XGB matches regular-season order very well on that split.

---

## 4. Takeaways

1. **Playoff-outcome runs (map_run, strict, strict_small_db):** In all cases the **ensemble is worse than W/L standings** on MAE/RMSE vs playoff outcome. MAP is not yet competitive with the standings baseline for global rank accuracy.
2. **Strict playoff eval** (explicit season split + `require_eos_final_rank`) performs **worse** than the main map_run (negative Spearman, champion rank 12), so the stricter setup does not improve and may be more sensitive to limited test seasons.
3. **Champion rank:** Best among playoff runs is **map_run** (6); strict runs give 12. So for “champion in top 4” style metrics, the main run is better.
4. **50y run:** Do not compare its Spearman/RMSE to other runs; it evaluates vs standings, not playoff outcome. Use it only for 50-season training/split experiments.
5. **Recommendation:** Keep **outputs8** as official best for Spearman/rank error vs playoff outcome. Use outputs14 MAP for exploratory work (e.g. NDCG, top-k, champion) and consider outputs16 (MAP + standing rank) to see if adding standing rank improves ordering.

---

## 5. Model A vs Model B

**Model A:** ListMLE / neural ranking (trained on playoff outcome with Spearman surrogate).  
**Model B:** XGBoost ranker.

All metrics below are test-set, same eval target as each run (playoff outcome except 50y = standings).

### 5.1 Main MAP run (`map_run`) — playoff outcome

| Metric | Model A | Model B | Ensemble |
|--------|---------|---------|----------|
| Spearman | **-0.41** | **0.31** | 0.27 |
| Rank MAE | 12.60 | **8.07** | 8.20 |
| Rank RMSE | 14.56 | **10.16** | 10.46 |
| NDCG@4 | ~0.0003 | 0.040 | 0.041 |

**Verdict:** Model B wins on every metric. Model A is anti-correlated (negative Spearman) and hurts the ensemble; ensemble sits between A and B but is pulled down by A.

### 5.2 Strict playoff eval (`map_run_strict_playoff_eval`) — playoff outcome

| Metric | Model A | Model B | Ensemble |
|--------|---------|---------|----------|
| Spearman | -0.40 | 0.02 | -0.12 |
| Rank MAE | 8.84 | **7.26** | 7.89 |
| Rank RMSE | 11.36 | **9.38** | 9.79 |
| NDCG@4 | **0.144** | 0.029 | 0.021 |

**Verdict:** B is better on global ordering (MAE, RMSE) and Spearman (A is negative). A is better only on NDCG@4 (top-4). Ensemble is worse than B alone (negative Spearman), so combining with A degrades.

### 5.3 Strict small DB — playoff outcome

| Metric | Model A | Model B | Ensemble |
|--------|---------|---------|----------|
| Spearman | -0.29 | **0.30** | 0.25 |
| Rank MAE | 11.47 | **8.07** | 8.20 |
| Rank RMSE | 13.91 | **10.24** | 10.58 |
| NDCG@4 | 0.042 | 0.040 | 0.036 |

**Verdict:** B wins on Spearman and rank error; A and B are close on NDCG@4. Ensemble is between them.

### 5.4 50y run — standings (not playoff outcome)

| Metric | Model A | Model B | Ensemble |
|--------|---------|---------|----------|
| Spearman | **-0.77** | **0.99** | 0.98 |
| Rank RMSE | 16.29 | **1.15** | 1.77 |

**Verdict:** Model A is strongly anti-correlated when evaluated vs regular-season order (ListMLE is trained on playoff outcome, so it does not follow standings). Model B drives the ensemble and matches standings almost perfectly.

### 5.5 Summary

- **Playoff-outcome evals:** Model B (XGB) consistently outperforms Model A on Spearman and rank MAE/RMSE. Model A is often negative Spearman (anti-correlated).
- **Top-4 (NDCG@4):** Only in strict playoff eval does A beat B (0.144 vs 0.029); in main and strict_small_db, B is on par or better.
- **Ensemble:** Including Model A in the ensemble usually does not improve over B alone and in strict eval makes it worse. So for playoff-outcome ranking, **Model B alone would likely be stronger than the current A+B ensemble.**

---

## 6. Why does MAP (Model A) underperform?

**Model A** = DeepSet + ListMLE (neural ranker trained with Spearman surrogate on playoff outcome). It “sucks” in the sense: negative or weak Spearman, worse than standings, and it drags the ensemble down. Likely reasons:

### 6.1 Signal at the wrong level

- **Model A** sees only **roster-level** inputs: per-player rolling stats (pts, reb, ast, stl, blk, tov, shooting, TS%, usage) over L10/L30 and player embeddings. No team-level SRS, SOS, Four Factors, or pace.
- **Playoff outcome** is driven by schedule, health, matchups, coaching, and clutch play — not just aggregate player box scores. So A is asked to predict a **team/season outcome** from **player stats** alone. That signal is weak and noisy.
- **Model B (XGB)** gets **team-level** features: Four Factors (eFG, TOV%, FT_rate, ORB%), pace, `standing_rank_norm`, and when enabled SOS/SRS, Elo, DefRtg_L10, etc. Those directly correlate with both regular-season and playoff success, so B has a much easier job.

### 6.2 Same target on every list (repeated label, weak list diversity)

- For `listmle_target=playoff_outcome`, **every** conference-date list in a season uses the **same** relevance: that season’s playoff outcome rank (champion=1, etc.). So from November to March the **ordering** the model is trained to predict is identical; only the roster/stats snapshot changes.
- The model can overfit to “this season’s order” or learn spurious roster→rank mappings that don’t generalize. At test time (new season), that can show up as **anti-correlation** (negative Spearman) or high rank error.

### 6.3 Listwise loss vs full-league evaluation

- Training is **listwise**: many small lists (e.g. 15 teams per conference per date) with ListMLE/Spearman-surrogate loss. Evaluation is **full 30-team** ranking vs playoff outcome (Spearman, RMSE).
- The loss does not directly optimize full-league Spearman; it optimizes ranking within each list. So even if in-list order is good, the global order can be poor or inverted when lists are merged at inference.

### 6.4 Architecture and stability

- DeepSet + attention can **collapse** (constant Z → constant scores → flat loss). The codebase already uses σReparam and gradient clipping to mitigate this (see `.cursor/plans/Attention_Report.md`). Even without full collapse, if the input signal is weak, the encoder can learn uninformative or unstable representations, leading to noisy or anti-correlated scores.

### 6.5 What would help Model A

- **Add team-level context as input:** e.g. standing_rank_norm, SRS, SOS, or Four Factors at list level (as in outputs12/outputs16 “standing rank” experiments) so A sees the same kind of signal B gets.
- **Richer list targets or multi-task:** e.g. mix standings-to-date and playoff outcome, or train on both so the model doesn’t see one fixed order per season.
- **Regularization / capacity:** Smaller model or stronger regularization to reduce overfitting to the repeated playoff order.
- **Evaluate listwise:** Report in-list NDCG/Spearman during training and compare to full-league eval to see if the gap is train vs eval setup.

---

## 7. References

- **Main MAP analysis:** `docs/OUTPUTS14_MAP_ANALYSIS.md`
- **Configs:** `config/outputs14_map_run.yaml`, `config/outputs14_map_run_strict_playoff_eval.yaml`
- **Eval reports:** `outputs14_map_run/<run_name>/eval_report.json`, `run_025/eval_report.json`, `run_025/ANALYSIS_01.md`
