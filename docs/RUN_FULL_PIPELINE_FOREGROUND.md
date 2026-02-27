# Run Full Pipeline (Separate Foreground Commands)

Full steps and commands for running the pipeline as 8 separate foreground steps. See [.cursor/plans/run_full_pipeline_foreground_784ee3ec.plan.md](../.cursor/plans/run_full_pipeline_foreground_784ee3ec.plan.md) for the plan summary.

## Setup (once at start)

From the project root, set the environment so `src` is importable:

- **Working directory:** `c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project`
- **PYTHONPATH:** same path (so `import src` works)

In PowerShell, from that directory:

```powershell
cd "c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project"
$env:PYTHONPATH = (Get-Location).Path
```

## Pipeline steps (run in order, one command per step; wait for each to finish)


| Step | Script                                | What it does                                         |
| ---- | ------------------------------------- | ---------------------------------------------------- |
| 1    | `python scripts/1_download_raw.py`    | Download raw logs; write manifest                    |
| 2    | `python scripts/2_build_db.py`        | Build DuckDB from raw                                |
| 3    | `python scripts/3_train_model_a.py`   | Train deep-set model A (slow)                        |
| 4    | `python scripts/4_train_model_b.py`   | Train model B (XGB/RF)                               |
| 5    | `python scripts/4b_train_stacking.py` | Train stacking meta-model                            |
| 6    | `python scripts/5_evaluate.py`        | Evaluate latest run's predictions → eval_report.json |
| 7    | `python scripts/5b_explain.py`        | Explain (SHAP/IG/attention)                          |
| 8    | `python scripts/6_run_inference.py`   | Run inference → new run_NNN and predictions          |


Each command runs in the **foreground**: start it, wait for it to complete (success or failure), then run the next. No chaining with `;` or `&&`, and no background execution.

## Execution approach

When implementing:

1. Run **step 1** in the foreground (with `cd` and `PYTHONPATH` set in that shell).
2. After it exits, run **step 2** in the same shell (cwd and PYTHONPATH already set).
3. Repeat for steps **3 through 8**, one invocation per step, each in the foreground.

Step 3 (train model A) can take several minutes; use a sufficiently long timeout for that step only so it is not cut off (e.g. 15–20 minutes). Other steps can use normal timeouts.

## Notes

- [scripts/run_full_pipeline.py](../scripts/run_full_pipeline.py) exists but runs all steps in one process; for separate foreground steps, **do not** use it. Use 8 separate `python scripts/<script>.py` commands instead.
- If any step exits non-zero, stop and report; do not run the next step.
- The evaluation fix (legacy `actual_rank` / `actual_global_rank` in [scripts/5_evaluate.py](../scripts/5_evaluate.py)) is in place, so step 6 should succeed on existing run_009 predictions.
