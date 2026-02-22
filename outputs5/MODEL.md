# outputs5 — Model and role

**Role:** ListMLE **outcome vs standings** comparison. Same pipeline as outputs4 but comparing training on `playoff_outcome` vs `final_rank` (standings) when evaluated on playoff outcome.

**Model:** ListMLE (Model A) + Model B + stacking. Multiple subdirs (e.g. ndcg_outcome, ndcg_standing, spearman_outcome, spearman_standing) each with configs that differ only by `listmle_target`. No standing rank as input (stat_dim 21).

**Difference from outputs4:** outputs4 = single baseline (final_rank); outputs5 = **controlled comparison** of listmle_target (playoff_outcome vs final_rank) on the same eval. Finding: standings-trained ListMLE often matched or beat outcome-trained on playoff metrics when evaluated on playoff outcome.

**See also:** [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md), README § ListMLE outcome vs standings.
