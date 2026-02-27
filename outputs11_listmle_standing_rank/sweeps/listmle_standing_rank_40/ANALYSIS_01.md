# Analysis 01 — outputs11 ListMLE + standing rank sweep (listmle_standing_rank_40)

**Sweep:** 40 Optuna trials, objective **spearman**, `listmle_target: playoff_outcome`, rolling [10,30], standing rank as input (stat_dim 22).

---

## Best trial (Optuna objective = spearman)

| Field | Value |
|-------|--------|
| **Best combo** | **combo_012** |
| **Spearman (ensemble)** | 0.459 |
| **playoff_spearman** | 0.467 |
| **rank_mae** (pred vs playoff outcome) | 7.13 |
| **rank_rmse** | 9.01 |
| **NDCG@4** | 0.464 |
| **NDCG@16** | 0.526 |
| **NDCG@30** | 0.596 |
| **Champion in top 4** | 1.0 |
| **Champion rank** | 2 |

**Best combo params:** model_a_epochs 21, max_depth 4, lr 0.08, n_estimators_xgb 291, n_estimators_rf 200, rolling [10,30], subsample 0.8, colsample_bytree 0.7, min_samples_leaf 5.

---

## Comparison to official best (outputs8_spearman_surrogate)

| Metric | outputs11 (ListMLE + standing) best | outputs8 (Spearman surrogate) best | Winner |
|--------|-------------------------------------|-------------------------------------|--------|
| **Spearman** | 0.459 | **0.777** (combo_0033) | outputs8 |
| **playoff_spearman** | 0.467 | **0.854** (combo_0038) | outputs8 |
| **rank_mae** | 7.13 | **4.80** (combo_0033) | outputs8 |
| **rank_rmse** | 9.01 | **5.78** (combo_0033) | outputs8 |
| **NDCG@30** | **0.596** | 0.522 (combo_0032) | outputs11 |
| **NDCG@4** | **0.464** | ~0.46 (combo_0033) | ~tie |

**Summary:** ListMLE + standing rank (outputs11) is **weaker** than the Spearman-surrogate sweep (outputs8) on correlation and rank error (Spearman, playoff_spearman, rank_mae, rank_rmse) but **stronger** on NDCG@30 (and competitive on NDCG@4). Same pattern as outputs9 (ListMLE with Spearman objective, no standing): ListMLE favors NDCG-style ranking at the top; adding standing rank does not close the gap to outputs8 on Spearman/rank error.

---

## vs W/L standings baseline (same test seasons)

- **Standings:** rank_mae 3.13, rank_rmse 4.45 vs playoff outcome.
- **Ensemble (combo_12):** rank_mae 7.13, rank_rmse 9.01 — model is **worse** than W/L standings on rank distance (typical when standings correlate with seeding and the model is tuned for Spearman/NDCG rather than RMSE).
- **spearman_standings** (standings vs playoff outcome): -0.49 — standings rank is **inverted** vs playoff outcome in this test set (East/West or season-specific).

---

## Conference (East vs West)

- **East:** NDCG 0.22–0.31, Spearman 0.24–0.31 across combos; best NDCG combo_13, best Spearman combo_1.
- **West:** NDCG 0.67–0.75, Spearman 0.50–0.61; best NDCG combo_0, best Spearman combo_30.
- West is easier (higher metrics); East has more variance and lower scores.

---

## Files

- **Full results:** `sweep_results.csv`, `sweep_results_summary.json`
- **Best config:** `combo_0012/config.yaml`
- **Optuna:** `optuna_study.json`, `optuna_importances.json` (if generated)

See [docs/OUTPUTS11_LISTMLE_STANDING_RANK.md](../../../docs/OUTPUTS11_LISTMLE_STANDING_RANK.md) and [docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](../../../docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md) for cross-run comparison and baseline (outputs6).
