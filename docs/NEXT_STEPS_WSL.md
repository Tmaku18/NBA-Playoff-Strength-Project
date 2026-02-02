# Next steps execution (WSL)

## What ran

- **Environment:** `python3.12-venv` installed in WSL; venv created at `/home/tmaku18/nba-venv` (Linux fs) and `pip install -r requirements.txt` completed there (avoids OneDrive write errors on project dir).
- **Config:** Set to best combo (phase2_rolling_on): `training.rolling_windows: [5, 10]`, `model_a.epochs: 16`.
- **Pipeline:** `run_pipeline_from_model_a.py` ran from project root with `PYTHONPATH=.` and `/home/tmaku18/nba-venv/bin/python`.
  - Steps 1–5 **succeeded:** 2_build_db (skipped), run_leakage_tests, 3_train_model_a, 4_train_model_b, 4b_train_stacking. Outputs written to `outputs3/` (oof_model_a.parquet, oof_model_b.parquet, best_deep_set.pt, xgb_model.joblib, rf_model.joblib, ridgecv_meta.joblib, oof_pooled.parquet).
  - Step 6 (6_run_inference.py) **failed:** `[Errno 1] Operation not permitted` when writing `outputs3/run_019/predictions.json`. This is a WSL → Windows mount (e.g. OneDrive) permission issue when writing under the project dir.

## How to complete the pipeline

1. **From Windows (recommended):** In PowerShell or cmd, `cd` to the project, set `config/defaults.yaml` to best combo if not already, then run:
   ```cmd
   set PYTHONPATH=.
   python scripts\run_pipeline_from_model_a.py
   ```
   Or run only steps 6, 5, 5b (inference, evaluate, explain) since 1–5 already produced artifacts in `outputs3/`:
   ```cmd
   python scripts\6_run_inference.py
   python scripts\5_evaluate.py
   python scripts\5b_explain.py
   ```

2. **From WSL with project on Linux fs:** Copy or clone the project to a path under `/home/...` (e.g. `/home/tmaku18/nba-project`), run the full pipeline there with `/home/tmaku18/nba-venv/bin/python` and `PYTHONPATH=.`; outputs will write without permission errors.

3. **WSL venv for future runs:** Use the venv in WSL home so deps are on Linux fs:
   ```bash
   cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project"
   PYTHONPATH=. /home/tmaku18/nba-venv/bin/python scripts/run_pipeline_from_model_a.py
   ```
   Expect step 6 to fail again with "Operation not permitted" unless the project (or at least `outputs3/`) is on a filesystem that WSL can write to (e.g. `\\wsl$` or a Linux path).

## Scripts added

- `scripts/run_next_steps.sh` — WSL: run pipeline (uses `.venv` if present, else `python3`; set config first).
- `scripts/run_next_steps.ps1` — Windows: run pipeline (uses `.venv\Scripts\python.exe` if present).

Config is already set to best combo (rolling_windows [5, 10], epochs 16). To revert defaults, edit `config/defaults.yaml` back to `rolling_windows: [10, 30]` and `epochs: 28`.
