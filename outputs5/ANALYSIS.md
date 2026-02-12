# outputs5: Best-Run Configs (No Sweeps)

All pipeline outputs for the **12** best-run configs go to **outputs5**, starting at **run_026** and iterating (run_027, …). No sweeps; each config uses the best hyperparameters from previous sweeps and experimentation.

---

## 1. Metric naming

- **W/L record standings** — End-of-season regular-season rank (1–30 by win %). Stored as `EOS_playoff_standings` in predictions. Baseline metrics: `rank_mae_wl_record_standings_vs_playoff_outcome_rank`, `rank_rmse_wl_record_standings_vs_playoff_outcome_rank`.
- **Playoff Outcome Rank** — Actual playoff finish (champion=1, runner-up=2, …). Stored as `EOS_global_rank` when `eos_rank_source` is `eos_final_rank`. Metrics: `spearman_pred_vs_playoff_outcome_rank`, `rank_mae_pred_vs_playoff_outcome_rank`, `ndcg_at_30_pred_vs_playoff_outcome_rank`, etc.
- **Historic Conference Rank** — Within-conference rank (1–15) used in predictions; axis label in plots (replaces "Actual Conference Rank").
- **NDCG cutoff labels:** ndcg_at_4 = Conference Finals (top 4); ndcg_at_12 = Clinch Playoff (top 12); ndcg_at_16 = One Play-In Tournament (top 16); ndcg_at_20 = Qualify for Playoffs (top 20); ndcg_at_30 = full order. (Documents and eval reports include this note so the numbers are clearly defined.)

---

## 2. Folder layout (12 configs)

Same structure as existing outputs5: each folder has `run_026/`, `eval_report.json`, `split_info.json`, models, and per-run artifacts (predictions, ANALYSIS_01.md, plots). **NDCG@10 removed;** primary ranking metric is **NDCG@30** (full 30-team order).

| Folder | ListMLE target | Best-from |
|--------|----------------|-----------|
| **regular_ndcg** | W/L record standings | outputs5 ndcg_standing |
| **regular_spearman** | W/L record standings | phase6 combo 8 |
| **regular_ndcg16** | W/L record standings | phase6 combo 19 |
| **regular_ndcg30** | W/L record standings | NDCG@30 (phase6-style) |
| **regular_rank_mae** | W/L record standings | phase6 combo 4 |
| **regular_rank_rmse** | W/L record standings | rank RMSE (standings) |
| **playoff_ndcg** | Playoff Outcome Rank | 20260212 combo 15 |
| **playoff_spearman** | Playoff Outcome Rank | 20260212 combo 16 |
| **playoff_ndcg16** | Playoff Outcome Rank | 20260212 combo 18 |
| **playoff_ndcg30** | Playoff Outcome Rank | NDCG@30 (20260212-style) |
| **playoff_rank_mae** | Playoff Outcome Rank | 20260212 combo 15 |
| **playoff_rank_rmse** | Playoff Outcome Rank | 20260212 combo 18 |

---

## 3. How to run

Configs are **overlays**: the pipeline merges them on top of `config/defaults.yaml` (regular_*) or `config/defaults_playoff_outcome.yaml` (playoff_*). All outputs go to `outputs5/<folder>`, first run is **run_026**.

**Single config (e.g. regular_ndcg):**
```bash
python -m scripts.run_pipeline_from_model_a --config config/outputs5_regular_ndcg.yaml
```
Optional: `--outputs outputs5/regular_ndcg` to override the path in the config.

**All 12 (one after another):**
```bash
for c in regular_ndcg regular_spearman regular_ndcg16 regular_ndcg30 regular_rank_mae regular_rank_rmse playoff_ndcg playoff_spearman playoff_ndcg16 playoff_ndcg30 playoff_rank_mae playoff_rank_rmse; do
  python -m scripts.run_pipeline_from_model_a --config config/outputs5_${c}.yaml
done
```

---

## 4. Config files

- `config/outputs5_regular_ndcg.yaml` … `config/outputs5_regular_rank_rmse.yaml` (incl. **regular_ndcg30**, **regular_rank_rmse**) — ListMLE on **W/L record standings** (`listmle_target: final_rank`).
- `config/outputs5_playoff_ndcg.yaml` … `config/outputs5_playoff_rank_rmse.yaml` (incl. **playoff_ndcg30**, **playoff_rank_rmse**) — ListMLE on **Playoff Outcome Rank** (`listmle_target: playoff_outcome`).

Each sets `paths.outputs`, `training.rolling_windows`, `model_a.epochs`, `model_b` (xgb/rf), and `inference.run_id_base: 26`.

---

## 5. Evaluation

After each run, script 5 writes `eval_report.json` (and per-season `eval_report_<season>.json`) with: `ndcg_at_30` (no NDCG@10), `rank_mae_pred_vs_playoff_outcome_rank`, `rank_rmse_*`, `playoff_metrics.spearman_pred_vs_playoff_outcome_rank`, `playoff_metrics.ndcg_at_30_pred_vs_playoff_outcome_rank`. See `outputs/ANALYSIS.md` and `docs/METRICS_USED.md` for full metric definitions.
