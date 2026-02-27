# outputs14_map_run: MAP run analysis

**Run analyzed:** `outputs14_map_run/map_run` (run_025)  
**Primary report:** `outputs14_map_run/map_run/eval_report.json`  
**Comparison references:** `outputs8/sweeps/20260217_042955/combo_0033/outputs/eval_report.json`, `outputs8/sweeps/20260217_042955/combo_0032/outputs/eval_report.json`

---

## Executive summary

The MAP run is **mixed**:

- Strong on top-focused ranking quality (**NDCG@4 = 0.464**) and champion placement (**champion_rank = 2**).
- Weak on global ordering and rank-distance metrics versus current best outputs8 combos (**Spearman 0.436, RMSE 9.19**).
- Worse than the standings baseline on rank MAE/RMSE in this run.

Bottom line: MAP is promising for top-end relevance, but it is **not competitive** yet as the overall best model.

---

## MAP metrics (outputs14_map_run)

| Metric | MAP value |
|--------|-----------|
| Spearman | 0.4358 |
| Playoff Spearman | 0.4607 |
| NDCG@4 | 0.4644 |
| NDCG@16 | 0.5248 |
| NDCG@30 | 0.5957 |
| Rank MAE (pred vs playoff outcome rank) | 7.40 |
| Rank RMSE (pred vs playoff outcome rank) | 9.19 |
| Brier championship odds | 0.0301 |
| Champion rank | 2 |
| Champion in top 4 | 1.0 |

---

## Comparison vs outputs8 references

`outputs8` has different best combos by objective, so two references are shown:

- `combo_0033`: official best Spearman / rank error combo.
- `combo_0032`: strong NDCG cutoff combo.

| Metric | MAP (outputs14) | outputs8 combo_0033 | outputs8 combo_0032 |
|--------|------------------|---------------------|---------------------|
| Spearman | 0.4358 | **0.7771** | 0.7370 |
| Playoff Spearman | 0.4607 | **0.8020** | 0.7513 |
| NDCG@4 | **0.4644** | 0.0422 | 0.3490 |
| NDCG@16 | **0.5248** | 0.4380 | 0.5218 |
| NDCG@30 | **0.5957** | 0.4381 | 0.5219 |
| Rank MAE | 7.40 | **4.80** | 5.07 |
| Rank RMSE | 9.19 | **5.78** | 6.28 |
| Champion rank | **2** | 7 | 3 |

Interpretation:

- MAP beats outputs8 on NDCG cutoffs and champion placement in this run.
- outputs8 remains clearly better for global ranking fidelity (Spearman/playoff Spearman) and rank error.

---

## Comparison vs standings baseline (same eval report)

From `model_vs_standings_comparison` in MAP eval report:

| Metric | Standings baseline | MAP ensemble |
|--------|--------------------|--------------|
| Rank MAE | **3.13** | 7.40 |
| Rank RMSE | **4.45** | 9.19 |
| Improvement vs standings | — | negative (MAE -4.27, RMSE -4.74) |

Bootstrap significance in the same report indicates MAP is not better than standings on rank MAE for this run.

---

## Conference split (MAP)

MAP shows strong West / weak East asymmetry:

- East Spearman: **0.221**
- West Spearman: **0.646**
- East RMSE (ensemble vs playoff outcome): **11.33**
- West RMSE (ensemble vs playoff outcome): **6.38**

This suggests the MAP run may need East-specific robustness improvements.

---

## Stability across seasons (MAP)

From `metrics_across_seasons`:

- Mean Spearman: **0.4198**, std **0.0227**
- Mean Playoff Spearman: **0.4474**, std **0.0189**
- Mean Rank RMSE: **9.3229**, std **0.1820**
- Number of test seasons: **2**

The run appears stable across these seasons, but stable at a level below outputs8 on core ordering metrics.

---

## Recommendation

Keep outputs8 as official best. Treat outputs14 MAP as an exploratory line:

1. Keep MAP for top-focused objective experiments (NDCG/Final Four behavior).
2. Do not promote MAP to official best until Spearman and rank RMSE close the gap to outputs8.
3. Use outputs16 (`MAP + standing rank`) as the next test to check whether standing-rank input improves MAP ordering metrics.

