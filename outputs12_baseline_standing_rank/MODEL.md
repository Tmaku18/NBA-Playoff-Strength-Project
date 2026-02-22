# outputs12_baseline_standing_rank — Model and role

**Folder name:** `outputs12_baseline_standing_rank`. See [OUTPUT_FOLDER_NAMING.md](../docs/OUTPUT_FOLDER_NAMING.md).

**Role:** Baseline-style config (ListMLE, final_rank, rolling [15,30]) with **standing rank as input** (stat_dim 22). Single run, not a sweep. Compare to **baseline (outputs6_baseline)** to isolate the effect of standing rank.

**Model:** Same as outputs4-style baseline (Phase 3 combo 18) but with `model_a.stat_dim: 22` and `model_a.use_standing_rank: true`. Config: `config/outputs4_baseline_standing_rank.yaml`; run to `outputs12_baseline_standing_rank/baseline_standing_rank`.

**Difference from outputs11_listmle_standing_rank:** outputs11 = ListMLE **sweep** (Optuna) with standing rank. outputs12 = **single run** with baseline config + standing rank.

**See also:** [docs/MODEL_LINEUP_AND_NEXT_STEPS.md](../docs/MODEL_LINEUP_AND_NEXT_STEPS.md), [docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md](../docs/OFFICIAL_BEST_CONFIGS_AND_ANALYSIS.md).
