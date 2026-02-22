# outputs10 — Model and role

**Role:** Same methodology as **outputs8** (Spearman-surrogate, playoff_outcome, 40 Optuna trials) but with **standing rank as an input feature** for Model A, B, and C (`standing_rank_norm`; stat_dim 22). Tests whether adding current regular-season rank as input improves over outputs8.

**Model:** Spearman-surrogate (Model A) + Model B + stacking, with **standing_rank_norm** in roster and team-context features. Sweep batch e.g. `standing_rank_spearman_40` under `outputs10/sweeps/standing_rank_spearman_40/combo_*/`. Best Spearman ~0.31 (combo_005); Model A Spearman is negative (inversion) — see docs/MODEL_A_STANDING_RANK_INVERSION.md.

**Difference from outputs9:** outputs9 = ListMLE sweep (with standing in defaults). outputs10 = **Spearman-surrogate** sweep **with standing rank as input** (same as outputs8 setup + standing). Compare to outputs8 (no standing) to isolate effect of standing; compare to outputs9 for ListMLE vs surrogate with standing.

**See also:** [docs/OUTPUTS10_SWEEP_STANDING_RANK.md](../docs/OUTPUTS10_SWEEP_STANDING_RANK.md), [docs/MODEL_A_STANDING_RANK_INVERSION.md](../docs/MODEL_A_STANDING_RANK_INVERSION.md), [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md).
