"""Run pipeline from train Model A onwards (3 -> 4 -> 4b -> 6 -> 5 -> 5b).

Set model_a.attention_debug: true in config so you see training loss and attention stats each epoch.
Requires DB at config paths.db (run 2_build_db.py first if missing).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    steps = [
        "3_train_model_a.py",
        "4_train_model_b.py",
        "4b_train_stacking.py",
        "6_run_inference.py",
        "5_evaluate.py",
        "5b_explain.py",
    ]
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    for i, script in enumerate(steps, 1):
        print(f"\n--- Step {i}/{len(steps)}: {script} ---")
        code = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            cwd=str(ROOT),
            env=env,
        ).returncode
        if code != 0:
            print(f"Pipeline failed at {script} (exit {code})")
            return code
    print("\n--- Pipeline complete ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
