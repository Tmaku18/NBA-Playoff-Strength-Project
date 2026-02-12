"""Run pipeline from script 2 (build DB) through explain. Skips script 1 (download raw).

What this does:
- Runs 2→leakage→3→4→4b→6→5→5b. Skips download since raw data assumed present.
- 2_build_db skips rebuild if raw hashes unchanged and DB exists.
- Runs in foreground. Use when raw data already exists and you want to retrain/eval.
- With --config: passes config to scripts 3, 4, 4b, 6, 5, 5b (e.g. config/defaults_reduced_features.yaml).
- With --outputs: overrides paths.outputs in the config (e.g. outputs5/ndcg_outcome) so all steps write there."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

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


def _deep_update(base: dict, overlay: dict) -> None:
    """Update base in-place with overlay (recursive for dicts)."""
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pipeline 2→leakage→3→4→4b→6→5→5b.")
    parser.add_argument("--config", type=str, default=None, help="Config YAML for scripts 3,4,4b,6,5,5b (e.g. config/defaults_reduced_features.yaml)")
    parser.add_argument("--outputs", type=str, default=None, help="Override paths.outputs (e.g. outputs5 or outputs5/regular_ndcg)")
    args = parser.parse_args()

    config_path = args.config
    temp_config_path: Path | None = None

    config_path_resolved = (ROOT / args.config if args.config and not Path(args.config).is_absolute() else Path(args.config)) if args.config else None
    # outputs5_*.yaml configs are overlays: merge on top of defaults (and defaults_playoff_outcome for playoff_*).
    if config_path_resolved and config_path_resolved.exists():
        name = config_path_resolved.name
        if "outputs5_" in name:
            with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if "playoff_" in name:
                with open(ROOT / "config" / "defaults_playoff_outcome.yaml", "r", encoding="utf-8") as f:
                    _deep_update(config, yaml.safe_load(f))
            with open(config_path_resolved, "r", encoding="utf-8") as f:
                _deep_update(config, yaml.safe_load(f))
            out_val = args.outputs if args.outputs is not None else config.get("paths", {}).get("outputs", "")
            if out_val:
                out_path = Path(out_val)
                config.setdefault("paths", {})["outputs"] = str(out_path.resolve() if out_path.is_absolute() else (ROOT / out_val).resolve())
            else:
                config.setdefault("paths", {})["outputs"] = str((ROOT / "outputs5").resolve())
            fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="pipeline_config_")
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                config_path = temp_config_path
            except Exception:
                Path(temp_config_path).unlink(missing_ok=True)
                raise
        elif args.outputs is not None:
            if not args.config:
                print("--outputs requires --config.", file=sys.stderr)
                return 1
            with open(config_path_resolved, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            out_val = args.outputs
            out_path = Path(out_val)
            config.setdefault("paths", {})["outputs"] = str(out_path.resolve() if out_path.is_absolute() else (ROOT / out_val).resolve())
            fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="pipeline_config_")
            try:
                with open(fd, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                config_path = temp_config_path
            except Exception:
                Path(temp_config_path).unlink(missing_ok=True)
                raise

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
    try:
        for i, script in enumerate(steps, 1):
            print(f"\n--- Step {i}/{len(steps)}: {script} ---")
            code = run(script, config_path)
            if code != 0:
                print(f"Pipeline failed at {script} (exit {code})")
                return code
        print("\n--- Pipeline complete ---")
        return 0
    finally:
        if temp_config_path and Path(temp_config_path).exists():
            Path(temp_config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
