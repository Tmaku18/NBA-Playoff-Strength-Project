# outputs4 — Model and role

**Role:** Production outputs root and **config source for baseline**. Phase 3 fine NDCG@16 combo 18 (ListMLE, `listmle_target: final_rank`, rolling [15,30], no standing rank) lives here; **baseline runs** that use these configs are in **outputs6** (phase_1 outcome). outputs4 also holds sweeps and runs (e.g. run_025, run_026).

**Model:** ListMLE (Model A) + Model B (XGB) + stacking. Config: `outputs4/sweeps/phase3_fine_ndcg16_final_rank/combo_0018/config.yaml`. Training target = regular-season standings (`final_rank`). **No standing rank as input** (stat_dim 21).

**Difference from outputs3:** outputs3 = exploratory sweeps; outputs4 = **production/sweep root** with fixed best combo (Phase 3 combo 18). The **baseline** reference for comparisons is **outputs6** (phase_1 outcome runs using outputs4-style configs).

**See also:** [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md), [docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](../docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md).
