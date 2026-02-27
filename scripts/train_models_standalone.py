"""Train Model A, Model B (XGB), and optionally Model C (RF) separately — no ensemble (no stacking).

What this does:
- Runs script 3 (train Model A) and script 4 (train Model B and optionally C).
- Does NOT run script 4b (stacking). No RidgeCV meta-learner is trained; no ensemble blend.
- Saves models to the same output dir: best_deep_set.pt, xgb_model.joblib, [rf_model.joblib].
- Optionally writes passthrough meta files so you can run standard inference with a single
  model: ridgecv_meta_model_a_only.joblib and ridgecv_meta_model_b_only.joblib. Copy one
  to ridgecv_meta.joblib before running script 6 to get predictions from that model only.

Usage:
  python -m scripts.train_models_standalone --config config/defaults.yaml
  python -m scripts.train_models_standalone --config config/team_stats_spearman_surrogate.yaml --outputs outputs_standalone/run_001

Requires: DB and split from a prior run, or run script 2 and then this (script 3 creates split_info.json).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_script(script_name: str, config_path: str, cwd: Path, env: dict | None = None) -> int:
    """Run a script with --config. Returns exit code."""
    import subprocess
    if not script_name.endswith(".py"):
        script_name = script_name + ".py"
    script_path = ROOT / "scripts" / script_name
    cmd = [sys.executable, str(script_path), "--config", config_path]
    env = env or {}
    env.setdefault("PYTHONPATH", str(ROOT))
    full_env = {**__import__("os").environ, **env}
    return subprocess.run(cmd, cwd=str(cwd), env=full_env).returncode


def write_passthrough_metas(output_dir: Path) -> None:
    """Write Ridge meta that passes through column 0 (A) or column 1 (B) only.
    So predict([sa, sx]) = sa for A-only, or sx for B-only."""
    # Fit with dummy 2-col input so sklearn Ridge has coef_/intercept_/n_features_in_
    X_dummy = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    y_dummy = np.array([1.0, 1.0])
    meta_a = Ridge(alpha=1.0, fit_intercept=False)
    meta_a.fit(X_dummy, y_dummy)
    meta_a.coef_ = np.array([1.0, 0.0], dtype=np.float64)
    meta_a.intercept_ = 0.0
    joblib.dump(meta_a, output_dir / "ridgecv_meta_model_a_only.joblib")
    meta_b = Ridge(alpha=1.0, fit_intercept=False)
    meta_b.fit(X_dummy, y_dummy)
    meta_b.coef_ = np.array([0.0, 1.0], dtype=np.float64)
    meta_b.intercept_ = 0.0
    joblib.dump(meta_b, output_dir / "ridgecv_meta_model_b_only.joblib")


def write_readme(output_dir: Path) -> None:
    readme = """# Standalone models (no ensemble)

Models in this directory were trained **separately**; no stacking meta-learner was fit.

## Artifacts

- `best_deep_set.pt` — Model A (Deep Set)
- `xgb_model.joblib` — Model B (XGBoost)
- `rf_model.joblib` — Model C (Random Forest), only if `training.train_model_c: true`
- `ridgecv_meta_model_a_only.joblib` — Passthrough: ensemble_score = Model A score only
- `ridgecv_meta_model_b_only.joblib` — Passthrough: ensemble_score = Model B score only

## Single-model inference

To run inference (script 6) using **only Model A** or **only Model B**:

1. **Model A only:**  
   Copy `ridgecv_meta_model_a_only.joblib` to `ridgecv_meta.joblib`, then run script 6.  
   (Or temporarily move/rename `xgb_model.joblib` so only A is loaded; ensemble_score will be A/2.)

2. **Model B only:**  
   Copy `ridgecv_meta_model_b_only.joblib` to `ridgecv_meta.joblib`, then run script 6.  
   (Or temporarily move `best_deep_set.pt` so only B is loaded; ensemble_score will be B/2.)

Evaluation (script 5) will then report metrics for that single model's ranking.
"""
    (output_dir / "STANDALONE_MODELS_README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train Model A, B, and optionally C separately; do not train ensemble (no stacking)."
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML (default: config/defaults.yaml)")
    parser.add_argument("--outputs", type=str, default=None, help="Override paths.outputs")
    parser.add_argument("--skip-passthrough-meta", action="store_true", help="Do not write passthrough meta files")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else ROOT / "config" / "defaults.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_val = args.outputs or config.get("paths", {}).get("outputs", "outputs_standalone")
    out_path = Path(out_val)
    if not out_path.is_absolute():
        out_path = (ROOT / out_val).resolve()
    config.setdefault("paths", {})["outputs"] = str(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Write merged config to temp file so scripts 3 and 4 see correct paths.outputs
    fd, temp_config = tempfile.mkstemp(suffix=".yaml", prefix="standalone_config_")
    try:
        with open(fd, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    except Exception:
        import os
        os.close(fd)
        Path(temp_config).unlink(missing_ok=True)
        raise

    try:
        print("Training Model A (script 3)...", flush=True)
        code3 = run_script("3_train_model_a.py", temp_config, ROOT)
        if code3 != 0:
            print(f"Script 3 exited with code {code3}", file=sys.stderr)
            return code3

        print("Training Model B (and optionally C) (script 4)...", flush=True)
        code4 = run_script("4_train_models_b_and_c.py", temp_config, ROOT)
        if code4 != 0:
            print(f"Script 4 exited with code {code4}", file=sys.stderr)
            return code4

        # Not running 4b_train_stacking.py — no ensemble.

        if not args.skip_passthrough_meta:
            write_passthrough_metas(out_path)
            print("Wrote ridgecv_meta_model_a_only.joblib and ridgecv_meta_model_b_only.joblib", flush=True)
        write_readme(out_path)
        print(f"Wrote {out_path / 'STANDALONE_MODELS_README.md'}", flush=True)
        print("Done. Models trained separately; no ensemble (stacking skipped).", flush=True)
        return 0
    finally:
        Path(temp_config).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
