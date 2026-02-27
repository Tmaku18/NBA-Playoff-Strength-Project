---
name: Defaults narrow sweep feature reduction
overview: Set Phase 3 combo 18 (standings) and Phase 5 combo 2 (playoff outcome) as the two NDCG@16 defaults; run narrow sweeps for both listmle targets; run feature reduction (export importances, apply config, re-train/eval) and optional feature-subset Optuna; use hardware specs for parallelism; document insights from existing analysis.
todos: []
isProject: false
---

# Defaults, Narrow Sweeps, Feature Reduction, and Parallelism

## Context

- **Hardware (from [~/.cursor/hardware-specs.md](file:///C:/Users/tmaku/.cursor/hardware-specs.md)):** 20 logical processors (i7-13700H), 32 GB RAM. Use **n_jobs=6** (or up to 8) for sweep parallel workers so each full-pipeline trial has headroom; feature_subset stays **n_jobs=1** (each trial changes config).
- **Current defaults:** [config/defaults.yaml](config/defaults.yaml) uses Phase 3 fine NDCG@16 combo 18 params and `listmle_target: final_rank`. No separate config for playoff-outcome default.
- **Narrow phase:** [scripts/sweep_hparams.py](scripts/sweep_hparams.py) already has `phase2_playoff_narrow` and [config/outputs4_phase6_narrow.yaml](config/outputs4_phase6_narrow.yaml); `--listmle-target` is a CLI override, so the same phase can run for both targets with different batch IDs.
- **Feature reduction:** [scripts/export_feature_importances.py](scripts/export_feature_importances.py) exists; [src/features/team_context.py](src/features/team_context.py) and config already support `model_b.include_features` / `model_b.exclude_features`.
- **Insights sources:** [outputs4/sweeps/OUTPUTS4_ANALYSIS.md](outputs4/sweeps/OUTPUTS4_ANALYSIS.md), [outputs4/sweeps/PROMISING_COMBOS.md](outputs4/sweeps/PROMISING_COMBOS.md), [docs/METRIC_MATRIX_EXPLORATION_PLAN.md](docs/METRIC_MATRIX_EXPLORATION_PLAN.md).

---

## 1. Two NDCG@16 defaults

**Goal:** Lock production to (1) standings-tuned default = Phase 3 combo 18, (2) playoff-outcome-tuned default = Phase 5 combo 2.

- **Keep [config/defaults.yaml](config/defaults.yaml) as the standings default:** Already uses combo 18 params (epochs 22, rolling [15,30], XGB 229/lr 0.072, RF 173, `listmle_target: final_rank`). Add a short comment at the top or under `training.listmle_target` that this is the **standings (final_rank) default**; combo 18 path: `outputs4/sweeps/phase3_fine_ndcg16_final_rank/combo_0018/config.yaml`.
- **Add [config/defaults_playoff_outcome.yaml](config/defaults_playoff_outcome.yaml):** New file that extends defaults and overrides only:
  - `training.listmle_target: "playoff_outcome"`
  - `model_a.epochs: 26` (from phase5 combo_002)
  - `model_b.xgb.n_estimators: 248`, `model_b.xgb.learning_rate: 0.0883` (combo_002)
  - `model_b.rf.n_estimators: 172`
  - Optional: `paths.outputs: "outputs4"` (same as defaults); no need to duplicate full YAML—use a minimal override file that the loader merges with defaults (same pattern as [config/outputs4_phase1.yaml](config/outputs4_phase1.yaml)).
- **Document in [README.md](README.md):** In the "Production default" sentence, state: standings default = `config/defaults.yaml` (Phase 3 combo 18, NDCG@16 0.550, final_rank); playoff-outcome default = `config/defaults_playoff_outcome.yaml` (Phase 5 combo 2, NDCG@16 0.543, playoff_outcome). Pipeline: use `--config config/defaults_playoff_outcome.yaml` when running for playoff-outcome-tuned predictions.
- **Document in [outputs4/sweeps/OUTPUTS4_ANALYSIS.md](outputs4/sweeps/OUTPUTS4_ANALYSIS.md):** In "Next steps" or "Recommendations", add one line that the two canonical defaults are combo 18 (standings) and combo 2 (playoff outcome) and point to the two config files.

---

## 2. Narrow sweeps for both listmle targets

**Goal:** Run narrow sweep optimized for playoff outcome (tuned default) and for standings (tuned default), using existing `phase2_playoff_narrow` and config, with higher parallelism.

- **Sweep A — playoff outcome tuned default:**  
Config: `config/outputs4_phase6_narrow.yaml`.  
Command (from project root, WSL or PowerShell with PYTHONPATH set):
  - `python -m scripts.sweep_hparams --config config/outputs4_phase6_narrow.yaml --method optuna --n-trials 20 --n-jobs 6 --objective ndcg16 --phase phase2_playoff_narrow --listmle-target playoff_outcome --batch-id phase6_ndcg16_playoff_narrow`
  - Output: `outputs4/sweeps/phase6_ndcg16_playoff_narrow/`. Use `--no-run-explain` if desired to save time.
- **Sweep B — standings tuned default:**  
Same config and phase; override listmle to standings (final_rank).
  - `python -m scripts.sweep_hparams --config config/outputs4_phase6_narrow.yaml --method optuna --n-trials 20 --n-jobs 6 --objective ndcg16 --phase phase2_playoff_narrow --listmle-target final_rank --batch-id phase6_ndcg16_standings_narrow`
  - Output: `outputs4/sweeps/phase6_ndcg16_standings_narrow/`.
- **Parallelism:** Use **n_jobs=6** (20 logical cores; leave headroom for I/O and Model A). If runs are stable, 8 can be tried; document in OUTPUTS4_ANALYSIS or a short run note.
- **After runs:** Aggregate results (or run `scripts/aggregate_sweep_results.py` if needed), update OUTPUTS4_ANALYSIS with phase6 best combo per target and optional comparison to phase5 combo 2 and phase3 combo 18.

---

## 3. Feature reduction (export, apply config, re-train and evaluate)

**Goal:** Export importances from best combos, apply suggested exclude or include in config, re-train Model B (and full pipeline if desired) and evaluate to confirm metrics with fewer Model B features.

- **Export from both combos:**
  - `python -m scripts.export_feature_importances --combo-dir outputs4/sweeps/phase5_ndcg16_playoff_broad/combo_0002 --threshold 0.05 --out outputs4/sweeps/phase5_ndcg16_playoff_broad/feature_importances_combo_002.json`
  - `python -m scripts.export_feature_importances --combo-dir outputs4/sweeps/phase3_fine_ndcg16_final_rank/combo_0018 --threshold 0.05 --out outputs4/sweeps/phase3_fine_ndcg16_final_rank/feature_importances_combo_18.json`
  - Inspect suggested `exclude_features` / `include_features` in the JSON (and console output). Optionally build a single list: e.g. exclude features that are below threshold in **both** combos, or use combo_002 only for playoff-outcome runs and combo_18 for standings runs.
- **Apply in config:** Create a dedicated config for reduced-feature runs (so the main defaults stay full-feature):
  - Add [config/defaults_reduced_features.yaml](config/defaults_reduced_features.yaml) (or two: one per target) that extends `defaults.yaml` and sets `model_b.exclude_features: [ ... ]` from the suggested list (or `model_b.include_features: [ ... ]` for allowlist). Use the same model_a/model_b/training params as the chosen default (combo 18 or combo 2) so the only change is the feature set.
- **Re-train and evaluate:** Run full pipeline once with the reduced-feature config (e.g. `--config config/defaults_reduced_features.yaml` from script 3 onward, or at least 4→4b→6→5) and record eval_report.json. Compare NDCG@16, Spearman, playoff_spearman to the full-feature default; document in OUTPUTS4_ANALYSIS or a short `outputs4/FEATURE_REDUCTION_RESULTS.md`.
- **Optional:** If reduced-feature metrics are close, set `config/defaults.yaml` (and `defaults_playoff_outcome.yaml`) to use the reduced list so future runs use fewer features by default.

---

## 4. Optuna feature-subset sweep

**Goal:** Run one feature-subset Optuna sweep (expensive: full pipeline per trial); keep n_jobs=1 as in current implementation.

- **Command:**  
`python -m scripts.sweep_hparams --config config/outputs4_phase1.yaml --method optuna --n-trials 10 --n-jobs 1 --phase feature_subset --objective ndcg16 --batch-id phase_feature_subset`
- **Note:** [scripts/sweep_hparams.py](scripts/sweep_hparams.py) already limits feature_subset to 15 trials and n_jobs=1; no code change. Run from project root with PYTHONPATH set. Optional: run a second sweep with `--listmle-target playoff_outcome` and a different batch-id to get a feature subset for playoff-outcome.
- **After run:** Add best feature-subset combo to OUTPUTS4_ANALYSIS or PROMISING_COMBOS; compare to full-feature combo 18 / combo 2.

---

## 5. Insights from phases, sweeps, and analysis

**Goal:** Summarize insights from existing docs and optionally extend exploration.

- **Review and summarize (no code required):**
  - [outputs4/sweeps/OUTPUTS4_ANALYSIS.md](outputs4/sweeps/OUTPUTS4_ANALYSIS.md): phase order, best combos per phase, Optuna importances (phase5: lr, n_xgb, n_rf matter; phase3 fine: lr 0.55, n_xgb 0.21).
  - [outputs4/sweeps/PROMISING_COMBOS.md](outputs4/sweeps/PROMISING_COMBOS.md): non-best combos (e.g. phase3 combo 2, 6; phase4 combo 0, 7) for ablation or narrow search.
  - [docs/METRIC_MATRIX_EXPLORATION_PLAN.md](docs/METRIC_MATRIX_EXPLORATION_PLAN.md): 8-sweep matrix (2 listmle targets × 4 objectives); note that ndcg16 and playoff_spearman are already supported; spearman_standings / ndcg_standings would need eval changes.
- **Optional doc update:** Add a short "Insights" subsection under [outputs4/sweeps/OUTPUTS4_ANALYSIS.md](outputs4/sweeps/OUTPUTS4_ANALYSIS.md) (or a separate `outputs4/sweeps/INSIGHTS.md`) with bullets: e.g. phase3 combo 18 best for standings; phase5 combo 2 best for playoff outcome; narrow phase fixes low-importance params; feature reduction and feature_subset are next steps; metric matrix suggests future 8-sweep comparison once eval supports standings metrics.
- **Optional wider exploration:** If desired, run one or two additional sweeps from the metric matrix (e.g. playoff_spearman objective with final_rank or playoff_outcome) and record in OUTPUTS4_ANALYSIS. This is optional and can be scoped in a follow-up.

---

## 6. Execution order and parallelism summary


| Step | Action                                                                                                    | Parallelism |
| ---- | --------------------------------------------------------------------------------------------------------- | ----------- |
| 1    | Set two defaults (defaults.yaml + new defaults_playoff_outcome.yaml); update README and OUTPUTS4_ANALYSIS | —           |
| 2a   | Narrow sweep, playoff outcome (`phase6_ndcg16_playoff_narrow`)                                            | n_jobs=6    |
| 2b   | Narrow sweep, standings (`phase6_ndcg16_standings_narrow`)                                                | n_jobs=6    |
| 3a   | Export feature importances (combo_002, combo_18)                                                          | —           |
| 3b   | Add defaults_reduced_features.yaml with exclude/include from export                                       | —           |
| 3c   | Run pipeline with reduced-feature config; record metrics                                                  | —           |
| 4    | Feature-subset Optuna sweep (phase_feature_subset)                                                        | n_jobs=1    |
| 5    | Document insights; optional metric-matrix runs                                                            | —           |


Sweeps 2a and 2b can be run sequentially (or in parallel in two terminals if desired). Feature-subset (step 4) is expensive; run after feature reduction if time is limited.

---

## File touch list


| Action                | File                                                                                                                                                         |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Comment or minor edit | [config/defaults.yaml](config/defaults.yaml) (standings default note)                                                                                        |
| Create                | [config/defaults_playoff_outcome.yaml](config/defaults_playoff_outcome.yaml) (phase5 combo 2 params + listmle_target playoff_outcome)                        |
| Create (optional)     | [config/defaults_reduced_features.yaml](config/defaults_reduced_features.yaml) (model_b.exclude_features or include_features from export)                    |
| Update                | [README.md](README.md) (two defaults: standings vs playoff outcome)                                                                                          |
| Update                | [outputs4/sweeps/OUTPUTS4_ANALYSIS.md](outputs4/sweeps/OUTPUTS4_ANALYSIS.md) (phase6 results after sweeps; insights; two defaults; feature reduction result) |
| Optional              | [outputs4/sweeps/INSIGHTS.md](outputs4/sweeps/INSIGHTS.md) or [outputs4/FEATURE_REDUCTION_RESULTS.md](outputs4/FEATURE_REDUCTION_RESULTS.md)                 |
| No code change        | [scripts/sweep_hparams.py](scripts/sweep_hparams.py) (already supports phase2_playoff_narrow, listmle-target, feature_subset; use n_jobs=6 in commands)      |


