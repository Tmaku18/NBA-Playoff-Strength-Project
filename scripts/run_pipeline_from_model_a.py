"""Run pipeline from script 2 (build DB) through explain. Skips script 1 (download raw).

What this does:
- Runs 2→leakage→3→4→4b→6→5→5b. Skips download since raw data assumed present.
- 2_build_db skips rebuild if raw hashes unchanged and DB exists.
- Runs in foreground. Use when raw data already exists and you want to retrain/eval.
- With --config: passes config to scripts 3, 4, 4b, 6, 5, 5b (e.g. config/defaults_reduced_features.yaml)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Scripts that accept --config (2_build_db and run_leakage_tests do not)
CONFIG_SCRIPTS = {"3_train_model_a.py", "4_train_models_b_and_c.py", "4b_train_stacking.py", "6_run_inference.py", "5_evaluate.py", "5b_explain.py"}


def run(script: str, config_path: str | None = None) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script)]
    if config_path and script in CONFIG_SCRIPTS:
        cmd.extend(["--config", config_path])
    return subprocess.run(
        cmd,
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pipeline 2→leakage→3→4→4b→6→5→5b.")
    parser.add_argument("--config", type=str, default=None, help="Config YAML for scripts 3,4,4b,6,5,5b (e.g. config/defaults_reduced_features.yaml)")
    args = parser.parse_args()

    steps = [
        "2_build_db.py",           # conditional: skip if raw unchanged
        "run_leakage_tests.py",
        "3_train_model_a.py",
        "4_train_models_b_and_c.py",
        "4b_train_stacking.py",
        "6_run_inference.py",
        "5_evaluate.py",
        "5b_explain.py",
    ]
    for i, script in enumerate(steps, 1):
        print(f"\n--- Step {i}/{len(steps)}: {script} ---")
        code = run(script, args.config)
        if code != 0:
            print(f"Pipeline failed at {script} (exit {code})")
            return code
    print("\n--- Pipeline complete ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
