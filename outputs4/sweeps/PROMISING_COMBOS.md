# Promising Combos (Non-Best) — Sweep Tracking

Configs that were strong but not best, for follow-up experiments or feature-trimming context.

---

## Phase 3 Fine NDCG@16 (phase3_fine_ndcg16_final_rank)

**Best:** Combo 18 — NDCG@16 0.550, Spearman 0.557, playoff_spearman 0.568.

| Combo | epochs | lr    | n_xgb | n_rf | NDCG@16 | Spearman | playoff_spearman |
|-------|--------|-------|-------|------|---------|----------|------------------|
| 2     | 18     | 0.076 | 220   | 171  | 0.543   | 0.487    | 0.491            |
| 6     | 21     | 0.086 | 239   | 202  | 0.545   | 0.513    | 0.490            |

**Optuna importances:** lr 0.55, n_xgb 0.21, epochs 0.15, n_rf 0.10.

---

## Phase 4 Playoff Outcome (phase4_ndcg16_playoff_outcome)

**Best:** Combo 12 — NDCG@16 0.543, Spearman 0.486, playoff_spearman 0.483.

| Combo | epochs | lr    | n_xgb | n_rf | NDCG@16 | Spearman | playoff_spearman |
|-------|--------|-------|-------|------|---------|----------|------------------|
| 0     | 20     | 0.074 | 251   | 172  | 0.536   | 0.528    | 0.525            |
| 7     | 20     | 0.057 | 232   | 186  | 0.534   | 0.534    | 0.521            |

---

## Phase 2 Coarse Spearman (phase2_coarse_spearman_final_rank)

**Best:** Combo 8 — Spearman 0.535, playoff_spearman 0.547, NDCG@16 0.543.

| Combo | epochs | lr    | n_xgb | n_rf | Spearman | playoff_spearman | NDCG@16 |
|-------|--------|-------|-------|------|----------|------------------|---------|
| 7     | 24     | —     | —     | —    | (rolling best) | —           | —       |

Phase 2 coarse had 15 trials; combo 8 was best across objectives. Combo 7 was the rolling sweep best before Phase 2 coarse.

**Optuna importances:** learning_rate 0.40, n_xgb 0.24, n_rf 0.24, model_a_epochs 0.13.

---

## Usage

- Use these combos as starting points for narrower search (e.g. phase2_fine around combo 2 or 6).
- Optuna importances guide which params to vary first in feature-trimming or ablation runs.
- See `docs/METRIC_MATRIX_EXPLORATION_PLAN.md` for the 8-sweep matrix and next-phase plan.
