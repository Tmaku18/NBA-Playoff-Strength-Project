# Execute next steps from Hyperparameter_Testing plan: best combo config + pipeline (foreground).
# Run from project root in PowerShell. Uses venv if present.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".venv\Scripts\python.exe"
} else {
    $Python = "python"
}

Write-Host "Running pipeline from model A (best combo: rolling_windows [5, 10], epochs 16)..."
Write-Host "Ensure config/defaults.yaml has training.rolling_windows: [5, 10] and model_a.epochs: 16"
$env:PYTHONPATH = "."
& $Python scripts/run_pipeline_from_model_a.py

Write-Host "Done. Check outputs3/ for the new run (e.g. run_020)."
