# Best outputs9 config + position-aware ListMLE

- **Source:** outputs9 sweep best combo (combo_16, Spearman 0.477).
- **Change:** `listmle_position_aware: true` (sweep had `false`).
- **Outputs:** `outputs9/best_position_aware/outputs/` (run_025 or next run_id).

## Run pipeline (from project root)

**PowerShell:**
```powershell
cd "c:\Users\tmaku\OneDrive\Documents\GSU\Advanced Machine Learning\NBA Playoff Strentgh Project"
$env:PYTHONPATH = (Get-Location).Path
python -m scripts.run_pipeline_from_model_a --config outputs9/best_position_aware/config.yaml
```

**WSL:**
```bash
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
python -m scripts.run_pipeline_from_model_a --config outputs9/best_position_aware/config.yaml
```

Runs: 2_build_db → leakage → 3 (Model A) → 4 (Models B/C) → 4b (stacking) → 6 (inference) → 5 (evaluate) → 5b (explain). Run in foreground; allow time for training.
