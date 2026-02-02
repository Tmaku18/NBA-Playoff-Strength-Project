"""Hyperparameter sweep: Model A epochs, Model B grid, rolling windows.

Runs in the foreground (no background/daemon mode). Writes results to
<config.paths.outputs>/sweeps/<batch_id>/ (e.g. outputs3/sweeps/<batch_id>/).
No artificial timeout.

Usage:
  python -m scripts.sweep_hparams [--batch-id BATCH_ID] [--dry-run] [--max-combos N]

--dry-run: Print combo count and config overrides without running.
--max-combos: Limit number of combos to run (for testing).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


def _load_config() -> dict:
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _run_cmd(script_path: str, extra_args: list[str], cwd: Path | None = None) -> int:
    env = {**__import__("os").environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run(
        [sys.executable, str(ROOT / script_path)] + extra_args,
        cwd=str(cwd or ROOT),
        env=env,
    )
    return r.returncode


def _collect_clone_metrics(report_path: Path) -> dict:
    """Read clone_classifier_report.json and extract AUC, Brier for sweep results."""
    if not report_path.exists():
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for key in ("val_auc", "val_brier", "holdout_auc", "holdout_brier", "train_auc", "train_brier"):
        if key in data and isinstance(data[key], (int, float)):
            out[f"clone_{key}"] = data[key]
    return out


def _collect_metrics(eval_path: Path) -> dict:
    """Read eval_report.json and extract metrics for sweep results."""
    if not eval_path.exists():
        return {}
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for key in ("test_metrics_ensemble", "test_metrics_model_a", "test_metrics_xgb", "test_metrics_rf"):
        m = data.get(key, {})
        if isinstance(m, dict):
            for k, v in m.items():
                if k == "playoff_metrics":
                    out[f"{key}_{k}"] = v
                elif isinstance(v, (int, float)):
                    out[f"{key}_{k}"] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--batch-id", type=str, default=None, help="Batch folder name; default: timestamp")
    parser.add_argument("--dry-run", action="store_true", help="Print combos without running")
    parser.add_argument("--max-combos", type=int, default=None, help="Limit number of combos (for testing)")
    args = parser.parse_args()

    config = _load_config()
    out_name = config.get("paths", {}).get("outputs", "outputs")
    out_dir = Path(out_name) if Path(out_name).is_absolute() else ROOT / out_name
    sweeps_dir = out_dir / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)

    batch_id = args.batch_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_dir = sweeps_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    sweep_cfg = config.get("sweep", {})
    epochs_list = sweep_cfg.get("model_a_epochs", [8, 12, 16, 20, 24, 28])
    rolling_list = sweep_cfg.get("rolling_windows", [[10, 30]])
    mb = sweep_cfg.get("model_b", {})
    max_depth_list = mb.get("max_depth", [4])
    lr_list = mb.get("learning_rate", [0.08])
    n_xgb_list = mb.get("n_estimators_xgb", [250])
    n_rf_list = mb.get("n_estimators_rf", [200])
    subsample_list = mb.get("subsample", [0.8])
    colsample_list = mb.get("colsample_bytree", [0.7])
    min_leaf_list = mb.get("min_samples_leaf", [5])
    if not isinstance(subsample_list, list):
        subsample_list = [subsample_list]
    if not isinstance(colsample_list, list):
        colsample_list = [colsample_list]
    if not isinstance(min_leaf_list, list):
        min_leaf_list = [min_leaf_list]

    combos = list(itertools.product(
        rolling_list,
        epochs_list,
        max_depth_list,
        lr_list,
        n_xgb_list,
        n_rf_list,
        subsample_list,
        colsample_list,
        min_leaf_list,
    ))
    if args.max_combos:
        combos = combos[: args.max_combos]

    print(f"Sweep: {len(combos)} combos in {batch_dir}", flush=True)
    if args.dry_run:
        for i, c in enumerate(combos[:5]):
            rw, ep, md, lr, nx, nr = c[0], c[1], c[2], c[3], c[4], c[5]
            sub = c[6] if len(c) > 6 else 0.8
            col = c[7] if len(c) > 7 else 0.7
            mleaf = c[8] if len(c) > 8 else 5
            print(f"  {i}: rolling={rw}, epochs={ep}, max_depth={md}, lr={lr}, n_xgb={nx}, n_rf={nr}, subsample={sub}, colsample={col}, min_leaf={mleaf}")
        if len(combos) > 5:
            print(f"  ... and {len(combos) - 5} more")
        return 0

    results = []
    for i, (rolling_windows, epochs, max_depth, lr, n_xgb, n_rf, subsample, colsample, min_leaf) in enumerate(combos):
        combo_dir = batch_dir / f"combo_{i:04d}"
        combo_dir.mkdir(parents=True, exist_ok=True)
        combo_out = combo_dir / "outputs"
        combo_out.mkdir(parents=True, exist_ok=True)

        cfg = copy.deepcopy(config)
        cfg["training"] = cfg.get("training", {})
        cfg["training"]["rolling_windows"] = list(rolling_windows)
        cfg["model_a"] = cfg.get("model_a", {})
        cfg["model_a"]["epochs"] = int(epochs)
        cfg["model_b"] = cfg.get("model_b", {})
        cfg["model_b"]["xgb"] = cfg["model_b"].get("xgb", {})
        cfg["model_b"]["xgb"]["max_depth"] = int(max_depth)
        cfg["model_b"]["xgb"]["learning_rate"] = float(lr)
        cfg["model_b"]["xgb"]["n_estimators"] = int(n_xgb)
        cfg["model_b"]["xgb"]["subsample"] = float(subsample)
        cfg["model_b"]["xgb"]["colsample_bytree"] = float(colsample)
        cfg["model_b"]["rf"] = cfg["model_b"].get("rf", {})
        cfg["model_b"]["rf"]["n_estimators"] = int(n_rf)
        cfg["model_b"]["rf"]["min_samples_leaf"] = int(min_leaf)
        cfg["paths"] = cfg.get("paths", {})
        cfg["paths"]["outputs"] = str(combo_out.resolve())

        config_path = combo_dir / "config.yaml"
        _write_config(config_path, cfg)
        config_arg = str(config_path)

        print(f"[{i+1}/{len(combos)}] rolling={rolling_windows}, epochs={epochs}, xgb d={max_depth} lr={lr} n={n_xgb} sub={subsample} col={colsample}, rf n={n_rf} min_leaf={min_leaf}", flush=True)

        include_clone = sweep_cfg.get("include_clone_classifier", False)
        pipeline = [
            ("scripts/3_train_model_a.py", "Model A"),
            ("scripts/4_train_model_b.py", "Model B"),
            ("scripts/4b_train_stacking.py", "Stacking"),
            ("scripts/6_run_inference.py", "Inference"),
            ("scripts/5_evaluate.py", "Eval"),
        ]
        if include_clone:
            pipeline.append(("scripts/4c_train_classifier_clone.py", "Clone Classifier"))
        for script, name in pipeline:
            code = _run_cmd(script, ["--config", config_arg])
            if code != 0:
                print(f"  FAILED at {name} (exit {code})", flush=True)
                results.append({
                    "combo": i,
                    "rolling_windows": str(rolling_windows),
                    "epochs": epochs,
                    "max_depth": max_depth,
                    "learning_rate": lr,
                    "n_xgb": n_xgb,
                    "n_rf": n_rf,
                    "subsample": subsample,
                    "colsample_bytree": colsample,
                    "min_samples_leaf": min_leaf,
                    "error": name,
                })
                break
        else:
            metrics = _collect_metrics(combo_out / "eval_report.json")
            row = {
                "combo": i,
                "rolling_windows": str(rolling_windows),
                "epochs": epochs,
                "max_depth": max_depth,
                "learning_rate": lr,
                "n_xgb": n_xgb,
                "n_rf": n_rf,
                **{k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            }
            results.append(row)

    # Write results
    if results:
        import csv
        cols = list(results[0].keys())
        with open(batch_dir / "sweep_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)

    # Summary: best by spearman, ndcg, rank_mae (lower is better)
    ensemble_key = "test_metrics_ensemble_spearman"
    ndcg_key = "test_metrics_ensemble_ndcg"
    rank_mae_key = "test_metrics_ensemble_rank_mae_pred_vs_playoff"
    valid = [r for r in results if ensemble_key in r and r.get(ensemble_key) is not None]
    summary = {}
    if valid:
        best_sp = max(valid, key=lambda x: float(x.get(ensemble_key, -2)))
        best_ndcg = max(valid, key=lambda x: float(x.get(ndcg_key, -1)))
        summary["best_by_spearman"] = best_sp
        summary["best_by_ndcg"] = best_ndcg
    valid_mae = [
        r for r in results
        if rank_mae_key in r
        and isinstance(r.get(rank_mae_key), (int, float))
        and math.isfinite(r.get(rank_mae_key))
    ]
    if valid_mae:
        best_mae = min(valid_mae, key=lambda x: float(x.get(rank_mae_key, float("inf"))))
        summary["best_by_rank_mae"] = best_mae
    with open(batch_dir / "sweep_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(batch_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump({"batch_id": batch_id, "n_combos": len(combos), "config_outputs": str(out_dir)}, f, indent=2)

    print(f"Wrote {batch_dir / 'sweep_results.csv'}, {batch_dir / 'sweep_results_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
