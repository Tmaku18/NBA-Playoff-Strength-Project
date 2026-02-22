# outputs6 — Model and role (baseline)

**Role:** **Project baseline.** Phase 1 outcome runs: pipeline runs that use **existing configs from outputs4** (e.g. WSL combo_0018, Phase 6 combo_0016) with `listmle_target: playoff_outcome`, writing to `outputs6/phase_1/outcome/best_*`. ListMLE, rolling [15,30]; stacking with confidence when used in WSL configs. **Reference for all later comparisons** (superseded by outputs8 for official best metrics, but baseline role = outputs6).

**Model:** Same as outputs4-style ListMLE + Model B + stacking, but target = **playoff_outcome** and output destination = outputs6. Run IDs start at 028 when path contains `outputs6`. Best_rmse / best_ndcg16 etc. refer to which outputs4 combo was used. Spearman ~0.75, rank_rmse ~6.1.

**Difference from outputs5:** outputs5 = outcome vs standings **comparison** (multiple targets, same structure). outputs6 = **baseline** destination for re-runs of outputs4-style configs with playoff_outcome and optional confidence stacking.

**See also:** [docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](../docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md), [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md).
