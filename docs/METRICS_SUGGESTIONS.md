# Evaluation metrics: current set and suggestions

## Currently implemented

### Rank correlation
- **Spearman** — Monotonic rank correlation; primary metric for overall ranking quality.
- **Kendall τ** — Concordant/discordant pairs; robust, interpretable.
- **Pearson** — Linear correlation; complements Spearman when relationship is roughly linear.

### Top‑k quality
- **NDCG** (and NDCG@4, @12, @16, @20, @30) — Graded relevance; emphasizes top of list.
- **Precision@4** — Fraction of actual top‑4 (by outcome) that appear in predicted top‑4. “We got 3 of the final four.”
- **Precision@8** — Same for top‑8 (conference semifinals).
- **MRR (top 2 / top 4)** — 1 / (rank of first “elite” team in predicted order).

### Error and calibration
- **Rank MAE / RMSE** — Against playoff outcome rank; compared to W/L standings baseline.
- **Brier (champion)** — For championship probability predictions.
- **ECE (champion)** — Expected calibration error for championship odds (new).

### Playoff‑specific
- **champion_rank** — 1‑based rank of the actual champion in predicted order (1 = predicted 1st).
- **champion_in_top_4** — 1 if champion was in predicted top‑4, else 0.
- **Spearman / Kendall / NDCG** — Predicted vs playoff outcome rank.

### Other
- **ROC‑AUC (upset)** — Discriminating upset vs non‑upset from model score.
- **Model vs standings** — MAE/RMSE improvement and bootstrap significance.

---

## Optional additions (not yet implemented)

- **Spearman / NDCG improvement vs standings** — Report `ensemble_spearman - spearman_standings` and same for NDCG@4 so “better than standings” is visible for correlation, not only MAE/RMSE.
- **Rank‑biased overlap (RBO)** — Top‑weighted rank similarity; good when “top 4 order” matters most (no scipy built‑in; would need a small implementation).
- **Recall@k** — Same numerator as Precision@k; can be useful if you care about “did we miss any of the true top‑k?” (With same k for both, precision = recall for set overlap.)
- **Pairwise accuracy** — Fraction of team pairs correctly ordered; redundant with Kendall (pairwise accuracy ≈ (Kendall + 1) / 2 for no ties).
- **Log loss** — If you have full probability vectors over outcomes (e.g. P(team i wins title)); you already have Brier for champion.
- **Per‑season metrics** — Spearman/NDCG by season in the report for stability checks.

---

## Where they appear

- **eval_report.json** — `test_metrics_ensemble`, `test_metrics_model_*`, `test_metrics_by_conference`, `playoff_metrics`.
- **ANALYSIS_0x.md** — “Test metrics (ensemble)” list and East vs West table (NDCG, Spearman, Kendall τ, MAE).
