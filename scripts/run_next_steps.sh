#!/usr/bin/env bash
# Execute next steps from Hyperparameter_Testing plan: best combo config + pipeline (foreground).
# Run from project root. Uses venv if present.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Use venv if present (project .venv or WSL-home nba-venv)
if [ -d ".venv" ] && [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
  export PATH="$(pwd)/.venv/bin:$PATH"
elif [ -x "/home/$USER/nba-venv/bin/python" ]; then
  PYTHON="/home/$USER/nba-venv/bin/python"
else
  PYTHON="python3"
fi

# Ensure deps (optional: uncomment to install if missing)
# "$PYTHON" -c "import numpy" 2>/dev/null || { "$PYTHON" -m pip install -r requirements.txt; }

# Best combo from phase2_rolling_on (plan): rolling_windows [5, 10], epochs 16
# Apply via env or by editing config - here we override via temp copy then run
# Option: edit config/defaults.yaml: training.rolling_windows: [5, 10], model_a.epochs: 16
echo "Running pipeline from model A (best combo: rolling_windows [5, 10], epochs 16)..."
echo "Ensure config/defaults.yaml has training.rolling_windows: [5, 10] and model_a.epochs: 16"

PYTHONPATH=. "$PYTHON" scripts/run_pipeline_from_model_a.py

echo "Done. Check outputs3/ for the new run (e.g. run_020)."
