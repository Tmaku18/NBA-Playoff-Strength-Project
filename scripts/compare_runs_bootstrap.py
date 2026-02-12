"""Paired bootstrap significance between two run/configs over test seasons.

Loads per-season eval reports (eval_report_<season>.json) from two run directories,
computes metric differences per season, then bootstrap resamples seasons with replacement
to get 95% CI and p-value for each metric.

Usage:
  python -m scripts.compare_runs_bootstrap --run-dir-a path/to/run_a --run-dir-b path/to/run_b [--out results.json] [--B 2000]
  Or pass combo outputs dirs; script will find run_* subdir containing eval_report_*.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Metric keys to compare (from test_metrics_ensemble or playoff_metrics subdict)
OUTCOME_METRIC_KEYS = [
    "spearman_pred_vs_playoff_outcome_rank",
    "ndcg_at_4_final_four",
    "ndcg_at_30_pred_vs_playoff_outcome_rank",
    "rank_rmse_pred_vs_playoff_outcome_rank",
]
STANDINGS_METRIC_KEYS = [
    "spearman_standings",
    "ndcg_at_4_standings",
    "ndcg_at_16_standings",
    "ndcg_at_30_standings",
    "rank_rmse_standings",
]
# Lower-is-better metrics (for p-value: proportion of bootstrap diffs <= 0 when run_a - run_b)
LOWER_IS_BETTER = {"rank_rmse_pred_vs_playoff_outcome_rank", "rank_rmse_standings"}


def _find_run_dir(path: Path) -> Path:
    """If path contains run_* subdir with eval_report_*.json, return that subdir; else return path."""
    path = Path(path).resolve()
    if not path.is_dir():
        return path
    for sub in sorted(path.iterdir()):
        if sub.is_dir() and sub.name.startswith("run_") and list(sub.glob("eval_report_*.json")):
            return sub
    if list(path.glob("eval_report_*.json")):
        return path
    return path


def _load_per_season_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    """Load per-season metrics from eval_report_<season>.json. Returns {season: {metric_key: value}}."""
    run_dir = _find_run_dir(run_dir)
    out: dict[str, dict[str, float]] = {}
    for report_path in sorted(run_dir.glob("eval_report_*.json")):
        # Skip plain eval_report.json (aggregate)
        if report_path.stem == "eval_report":
            continue
        season = report_path.stem.replace("eval_report_", "")
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        ens = data.get("test_metrics_ensemble") or {}
        row = {}
        for key in OUTCOME_METRIC_KEYS + STANDINGS_METRIC_KEYS:
            if key in ens and isinstance(ens[key], (int, float)) and math.isfinite(ens[key]):
                row[key] = float(ens[key])
        # playoff_metrics sub-dict (overlay outcome metrics when present)
        pm = ens.get("playoff_metrics") or {}
        for key in OUTCOME_METRIC_KEYS:
            if key in pm and isinstance(pm[key], (int, float)) and math.isfinite(pm[key]):
                row[key] = float(pm[key])
        # Fallback: some reports have outcome metrics at top level
        for key in OUTCOME_METRIC_KEYS + STANDINGS_METRIC_KEYS:
            if key not in row and key in ens and isinstance(ens[key], (int, float)) and math.isfinite(ens[key]):
                row[key] = float(ens[key])
        if row:
            out[season] = row
    return out


def _paired_bootstrap(
    seasons: list[str],
    metrics_a: dict[str, dict[str, float]],
    metrics_b: dict[str, dict[str, float]],
    metric_keys: list[str],
    B: int = 2000,
    seed: int = 42,
) -> dict[str, dict]:
    """Paired bootstrap over seasons. Returns per-metric: mean_diff, ci_low, ci_high, p_value."""
    rng = np.random.default_rng(seed)
    n = len(seasons)
    if n == 0:
        return {}
    results = {}
    for key in metric_keys:
        va = [metrics_a.get(s, {}).get(key) for s in seasons]
        vb = [metrics_b.get(s, {}).get(key) for s in seasons]
        valid = [i for i in range(n) if va[i] is not None and vb[i] is not None]
        if len(valid) < 2:
            results[key] = {"mean_diff": None, "ci_low": None, "ci_high": None, "p_value": None, "n_seasons": len(valid)}
            continue
        va = np.array([va[i] for i in valid], dtype=np.float64)
        vb = np.array([vb[i] for i in valid], dtype=np.float64)
        diffs = va - vb
        mean_diff = float(np.mean(diffs))
        boot_diffs = []
        for _ in range(B):
            idx = rng.integers(0, len(valid), size=len(valid))
            boot_diffs.append(float(np.mean(diffs[idx])))
        boot_diffs = np.array(boot_diffs)
        ci_low = float(np.percentile(boot_diffs, 2.5))
        ci_high = float(np.percentile(boot_diffs, 97.5))
        if key in LOWER_IS_BETTER:
            # H0: no improvement. We want run_a better = lower metric. So diff = a - b; if a < b, diff < 0. p = proportion of boot >= 0 (no improvement or worse)
            p_value = float(np.mean(boot_diffs >= 0))
        else:
            # Higher is better. p = proportion of boot <= 0
            p_value = float(np.mean(boot_diffs <= 0))
        results[key] = {
            "mean_diff": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": p_value,
            "n_seasons": len(valid),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired bootstrap significance between two runs")
    parser.add_argument("--run-dir-a", type=str, required=True, help="First run directory (or combo outputs dir)")
    parser.add_argument("--run-dir-b", type=str, required=True, help="Second run directory (or combo outputs dir)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path; default: bootstrap_significance.json in run-dir-a")
    parser.add_argument("--B", type=int, default=2000, help="Bootstrap replicates (default 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_a = Path(args.run_dir_a).resolve()
    run_b = Path(args.run_dir_b).resolve()
    metrics_a = _load_per_season_metrics(run_a)
    metrics_b = _load_per_season_metrics(run_b)

    seasons_a = set(metrics_a.keys())
    seasons_b = set(metrics_b.keys())
    seasons_common = sorted(seasons_a & seasons_b)
    if not seasons_common:
        print("No common seasons with metrics between the two runs.", file=sys.stderr)
        return 1

    all_keys = list(dict.fromkeys(OUTCOME_METRIC_KEYS + STANDINGS_METRIC_KEYS))
    results = _paired_bootstrap(seasons_common, metrics_a, metrics_b, all_keys, B=args.B, seed=args.seed)

    out_data = {
        "run_dir_a": str(run_a),
        "run_dir_b": str(run_b),
        "seasons": seasons_common,
        "n_seasons": len(seasons_common),
        "B": args.B,
        "metrics": results,
    }
    out_path = Path(args.out) if args.out else _find_run_dir(run_a) / "bootstrap_significance.json"
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)
    print(f"Wrote {out_path}")
    for key, r in results.items():
        if r.get("mean_diff") is not None:
            print(f"  {key}: mean_diff={r['mean_diff']:.4f} 95% CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}] p={r['p_value']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
