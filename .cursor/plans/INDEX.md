# Cursor Plans Index

One place to see what lives in `.cursor/plans/` and where. Links use paths relative to repo root.

## Root (active / canonical / reference)

| File | Name / summary | Status | Location |
|------|----------------|--------|----------|
| [Plan.md](Plan.md) | NBA True Strength Prediction — project source-of-truth, 10-phase roadmap | canonical | root |
| [Implementation_Plan_Roadmap.md](Implementation_Plan_Roadmap.md) | Phase 0–9 implementation checklist; points to Plan.md | canonical | root |
| [project_scope_and_design_roadmap_document_8fd95820.plan.md](project_scope_and_design_roadmap_document_8fd95820.plan.md) | Project scope + design roadmap; maps Plan, Update1–8, and plan references | reference | root |
| [team-stats_probabilistic_models_1635188e.plan.md](team-stats_probabilistic_models_1635188e.plan.md) | Team-stats track: LR, BayesianRidge, GPR, GMM; uncertainty; optional Model A score | completed | root |
| [run_full_pipeline_foreground_784ee3ec.plan.md](run_full_pipeline_foreground_784ee3ec.plan.md) | Run full pipeline as 8 separate foreground commands; PYTHONPATH and WSL | active | root |
| [Attention_Report.md](Attention_Report.md) | Attention collapse investigation, σReparam, diagnostics, fix_attention link | reference | root |
| [Performance_trajectory_and_hyperparameters.md](Performance_trajectory_and_hyperparameters.md) | How performance evolved across runs; optimal hyperparams; what to re-check | reference | root |

## Archive (completed / superseded / historical)

Plans below are in [archive/](archive/). Use `.cursor/plans/archive/<filename>` when linking.

### Sweeps and runs

| File | Summary | Status |
|------|---------|--------|
| [archive/phased_sweep_roadmap_3hr_6b5aa588.plan.md](archive/phased_sweep_roadmap_3hr_6b5aa588.plan.md) | Phased sweep roadmap; Phase 0 baseline, Phase 1 spearman/ndcg | completed |
| [archive/phased_sweep_execution_plan_d0f3e0a3.plan.md](archive/phased_sweep_execution_plan_d0f3e0a3.plan.md) | Actionable Phase 1 sweep execution (12 sweeps, 4 jobs) | completed |
| [archive/outputs4_phase_i_sweeps_2638c514.plan.md](archive/outputs4_phase_i_sweeps_2638c514.plan.md) | Outputs4 Phase I sweeps; run_id 25; FAST (AMP + batch cache) | completed |
| [archive/outputs2_run_019_sweeps_update8_d5fca612.plan.md](archive/outputs2_run_019_sweeps_update8_d5fca612.plan.md) | Outputs2, run_019; sweeps; Update8 doc | completed |
| [archive/refined_sweep_rerun_bc8afb8f.plan.md](archive/refined_sweep_rerun_bc8afb8f.plan.md) | Rerun sweep around sweet spots; val_frac 0.25; two-phase Model B grid | completed |
| [archive/sweep_rerun_+_attention_check_7aa587c2.plan.md](archive/sweep_rerun_+_attention_check_7aa587c2.plan.md) | Sweep rerun + attention diagnostic | completed |
| [archive/sweep_foreground_timing_early_exit_58c85b55.plan.md](archive/sweep_foreground_timing_early_exit_58c85b55.plan.md) | Sweep foreground timing and early exit | completed |
| [archive/sweep-script-exec_03086ef1.plan.md](archive/sweep-script-exec_03086ef1.plan.md) | Real-data sweep script; DuckDB; first batch | completed |
| [archive/ndcg16_playoff_outcome_sweep_cf6971f0.plan.md](archive/ndcg16_playoff_outcome_sweep_cf6971f0.plan.md) | NDCG@16 playoff-outcome sweep | completed |
| [archive/playoff_outcome_broad_sweep_5289c065.plan.md](archive/playoff_outcome_broad_sweep_5289c065.plan.md) | Broader playoff-outcome search phase | completed |
| [archive/phase1_spearman_rerun_plan_936d5aca.plan.md](archive/phase1_spearman_rerun_plan_936d5aca.plan.md) | Phase 1 Spearman rerun | completed |
| [archive/defaults_narrow_sweep_feature_reduction_ee4ff3a9.plan.md](archive/defaults_narrow_sweep_feature_reduction_ee4ff3a9.plan.md) | Defaults narrow sweep; feature reduction; Optuna importances | completed |
| [archive/PHASE2_SWEEP_PLAN.md](archive/PHASE2_SWEEP_PLAN.md) | Phase 2 sweep plan | completed |
| [archive/PHASE2_GRANULAR_SWEEP_PLAN.md](archive/PHASE2_GRANULAR_SWEEP_PLAN.md) | Phase 2 granular sweep sub-phases | completed |
| [archive/next_full_pipeline_run_b24c7601.plan.md](archive/next_full_pipeline_run_b24c7601.plan.md) | Next end-to-end pipeline run; configs per objective | completed |
| [archive/lock_best_config_and_explain.plan.md](archive/lock_best_config_and_explain.plan.md) | Lock best sweep config; run explain on best run | completed |
| [archive/lock_best_config_and_explain_b37c4a43.plan.md](archive/lock_best_config_and_explain_b37c4a43.plan.md) | Lock best config; 5b_explain; Phase 2 attention hardening | completed |

### Fixes (attention, IG, playoff, pipeline)

| File | Summary | Status |
|------|---------|--------|
| [archive/fix_attention_+_trustworthy_run_d52cdb1c.plan.md](archive/fix_attention_+_trustworthy_run_d52cdb1c.plan.md) | Fix attention collapse; NDCG-first hyperparams; pipeline guardrails | completed |
| [archive/fix_issues_for_final_pipeline_run_c582e145.plan.md](archive/fix_issues_for_final_pipeline_run_c582e145.plan.md) | Attention, playoff ranks, roster, hyperparams; pre-flight checklist | completed |
| [archive/fix_run_009_outputs_56afc397.plan.md](archive/fix_run_009_outputs_56afc397.plan.md) | Empty roster contributors; IG NaNs; EOS rank names | completed |
| [archive/ig_and_playoff_rank_fixes_f6af9789.plan.md](archive/ig_and_playoff_rank_fixes_f6af9789.plan.md) | IG batching; playoff rank bounds; manifest; run_008 quality | completed |
| [archive/model_a_attention_fix_and_phased_roadmap_1e5c219f.plan.md](archive/model_a_attention_fix_and_phased_roadmap_1e5c219f.plan.md) | Model A attention fix; debugger; Optuna sweeps; phased tasks | completed |
| [archive/attention_collapse_and_diagnostics_ab8a6d19.plan.md](archive/attention_collapse_and_diagnostics_ab8a6d19.plan.md) | Attention collapse; create Attention_Report.md | completed |

### Features and models

| File | Summary | Status |
|------|---------|--------|
| [archive/playoff-aware_rankings_&_odds_0cfb10c0.plan.md](archive/playoff-aware_rankings_&_odds_0cfb10c0.plan.md) | Playoff data; championship odds; conference ranks; config switch | completed |
| [archive/75-25_split_and_richer_metrics_78a2db80.plan.md](archive/75-25_split_and_richer_metrics_78a2db80.plan.md) | 75/25 train/test; split_info; train and test metrics | completed |
| [archive/walk_forward_and_eos_final_rank_combined.plan.md](archive/walk_forward_and_eos_final_rank_combined.plan.md) | Walk-forward training; EOS final rank (Option B); per-season inference | completed |
| [archive/validation_season_and_eos_final_rank_3c8a4e4c.plan.md](archive/validation_season_and_eos_final_rank_3c8a4e4c.plan.md) | Validation season and EOS final rank | completed |
| [archive/next_steps_attention_+_playoff_ranks_570e4a25.plan.md](archive/next_steps_attention_+_playoff_ranks_570e4a25.plan.md) | Next steps: attention and playoff ranks | completed |
| [archive/confidence-weighted_ensemble_and_report_d4c5d7ff.plan.md](archive/confidence-weighted_ensemble_and_report_d4c5d7ff.plan.md) | 4-input meta-learner (s_A, s_X, c_A, c_X); confidence from attention and XGB | completed |
| [archive/batch_cache_for_sweeps_4d2251d5.plan.md](archive/batch_cache_for_sweeps_4d2251d5.plan.md) | Disk cache for Model A list/batch building; keyed by config and DB | completed |
| [archive/amp_with_float32_loss_8e415e9a.plan.md](archive/amp_with_float32_loss_8e415e9a.plan.md) | AMP for Model A; loss in float32; GradScaler when CUDA | completed |
| [archive/xgboost_gpu_training_18b9db1e.plan.md](archive/xgboost_gpu_training_18b9db1e.plan.md) | XGBoost GPU (tree_method, device) for Model B | completed |
| [archive/enable_optional_features_7b94a57e.plan.md](archive/enable_optional_features_7b94a57e.plan.md) | Enable Elo, team rolling, motivation, injury, Monte Carlo, SOS/SRS | completed |
| [archive/integrate_raptor_metrics_58a50adc.plan.md](archive/integrate_raptor_metrics_58a50adc.plan.md) | FiveThirtyEight RAPTOR into pipeline; team_context | completed |
| [archive/east_west_conference_training_906d03de.plan.md](archive/east_west_conference_training_906d03de.plan.md) | East/west conference training | completed |
| [archive/centralize_training_config_attention_eval_expansion.plan.md](archive/centralize_training_config_attention_eval_expansion.plan.md) | Centralize training config; attention; eval expansion | completed |
| [archive/comprehensive_feature_and_evaluation_expansion.plan.md](archive/comprehensive_feature_and_evaluation_expansion.plan.md) | Feature and evaluation expansion (Second Order, RAPTOR, calibration) | completed |
| [archive/push_docs_metric_matrix_d37e47dc.plan.md](archive/push_docs_metric_matrix_d37e47dc.plan.md) | Push docs; metric matrix; Spearman/NDCG sweeps; Optuna importances | completed |

### History: Update1–8 and related

| File | Summary | Status |
|------|---------|--------|
| [archive/Update1.md](archive/Update1.md) | Playoff tables; playoff-performance rank; odds; conference rank; plots | historical |
| [archive/Update2.md](archive/Update2.md) | IG batching; latest-team roster; EOS_global_rank; manifest; NaN handling | historical |
| [archive/Update3.md](archive/Update3.md) | Date-range filtering; masked query; attention debug and fallback | historical |
| [archive/Update4.md](archive/Update4.md) | Sweep script with real DB; configurable epochs; Model B grid | historical |
| [archive/Update5.md](archive/Update5.md) | (Historical update) | historical |
| [archive/Update6.md](archive/Update6.md) | (Historical update) | historical |
| [archive/Update7.md](archive/Update7.md) | EOS final rank Option B; per-season inference; walk-forward | historical |
| [archive/Update8.md](archive/Update8.md) | Outputs2; run_id 19; sweeps path; foreground | historical |
| [archive/Plan_and_readme_updates_1.plan.md](archive/Plan_and_readme_updates_1.plan.md) | Plan and README updates (risk analysis; game-level lists; leakage) | completed |
| [archive/Plan_and_readme_updates_2.plan.md](archive/Plan_and_readme_updates_2.plan.md) | Plan and README updates (same themes) | completed |
| [archive/Run_Instructions.md](archive/Run_Instructions.md) | Run full pipeline foreground (duplicate of run_full_pipeline_foreground plan) | archived |
| [archive/Implementation_Plan_RoadMap1.md](archive/Implementation_Plan_RoadMap1.md) | Alternate implementation roadmap; points to Plan.md | completed |
| [archive/GPT_Plan.md](archive/GPT_Plan.md) | GPT-era plan reference | completed |
| [archive/GPT_Fix.md](archive/GPT_Fix.md) | GPT-era fix notes | completed |
| [archive/Opus_Plan.md](archive/Opus_Plan.md) | Opus-era plan reference | completed |
| [archive/Opus_fix.md](archive/Opus_fix.md) | Opus-era fix notes | completed |
| [archive/Notion_update_summary.md](archive/Notion_update_summary.md) | Notion update summary | completed |
| [archive/NBA_ANALYST_METRICS.md](archive/NBA_ANALYST_METRICS.md) | NBA analyst metrics | completed |
| [archive/PlayoffPerformanceLearning.md](archive/PlayoffPerformanceLearning.md) | Playoff performance learning | completed |

---

**Legend:** Root = keep in `.cursor/plans/` for quick access. Archive = completed, superseded, or historical; full content in `archive/`. When linking from `docs/` or other plans, use `.cursor/plans/archive/<filename>` for archived plans.
