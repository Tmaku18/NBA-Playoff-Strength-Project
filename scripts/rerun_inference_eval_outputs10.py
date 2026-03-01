"""Re-run inference (6) and evaluate (5) for outputs10 combos that failed at inference.

Usage (WSL, from project root with PYTHONPATH set):
  python -m scripts.rerun_inference_eval_outputs10

Scans outputs10/sweeps/standing_rank_spearman_40/combo_XXXX/ for combos that have
config + outputs (and optionally .current_run) but no eval_report.json in outputs/.
Runs 6_run_inference then 5_evaluate for each such combo.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "output" / "10_spearman_surrogate_standing_rank" / "sweeps" / "standing_rank_spearman_40"


def main() -> int:
    if not BATCH.exists():
        print(f"Batch dir not found: {BATCH}", file=sys.stderr)
        return 1
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    combos_done = 0
    combos_fail = 0
    for i in range(100):
        combo_dir = BATCH / f"combo_{i:04d}"
        if not combo_dir.exists():
            break
        cfg = combo_dir / "config.yaml"
        out_dir = combo_dir / "outputs"
        eval_report = out_dir / "eval_report.json"
        if not cfg.exists() or not out_dir.exists():
            continue
        if eval_report.exists():
            print(f"Combo {i}: already has eval_report.json, skip.")
            combos_done += 1
            continue
        # Need inference + evaluate; run_id from .current_run (set by script 3)
        run_id = "run_025"
        current_run = out_dir / ".current_run"
        if current_run.exists():
            run_id = current_run.read_text().strip() or run_id
        print(f"\n--- Combo {i} (run_id={run_id}) ---")
        rel_cfg = cfg.relative_to(ROOT) if cfg.is_relative_to(ROOT) else cfg
        cfg_str = str(rel_cfg)
        code = subprocess.run(
            [sys.executable, "-m", "scripts.6_run_inference", "--config", cfg_str],
            cwd=str(ROOT),
            env=env,
        ).returncode
        if code != 0:
            print(f"Combo {i}: inference failed (exit {code})", file=sys.stderr)
            combos_fail += 1
            continue
        code = subprocess.run(
            [sys.executable, "-m", "scripts.5_evaluate", "--config", cfg_str, "--run-id", run_id],
            cwd=str(ROOT),
            env=env,
        ).returncode
        if code != 0:
            print(f"Combo {i}: evaluate failed (exit {code})", file=sys.stderr)
            combos_fail += 1
            continue
        combos_done += 1
        print(f"Combo {i}: inference + evaluate OK.")
    print(f"\nDone. OK: {combos_done}, Failed: {combos_fail}")
    return 0 if combos_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
