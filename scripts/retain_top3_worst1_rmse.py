"""Retain only top 3 and worst 1 runs by RMSE per sweep batch (outputs other than outputs8).

Usage:
  python -m scripts.retain_top3_worst1_rmse [--dry-run] [--outputs OUTPUTS]
  --outputs: specific output root (e.g. output/outputs4) or "all" (default) for all output/outputs* except outputs8.
  --dry-run: print what would be kept/deleted without deleting.

Policy: For each sweep batch, rank combos by RMSE (lower is better). Keep top 3 and worst 1;
delete all other combo dirs. Uses only RMSE (not MAE). outputs8 is never modified.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Flattened keys from eval_report (current and older formats)
RMSE_KEYS = [
    "test_metrics_ensemble_rank_rmse_pred_vs_playoff_outcome_rank",
    "test_metrics_ensemble_rank_rmse_standings",
    "test_metrics_ensemble_rank_rmse_pred_vs_playoff",  # older format (outputs2, outputs3)
]


def collect_metrics(eval_path: Path) -> dict:
    """Read eval_report.json and extract flattened metrics (RMSE keys used for retention)."""
    if not eval_path.exists():
        return {}
    try:
        with open(eval_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for key in ("test_metrics_ensemble", "test_metrics_model_a", "test_metrics_model_b", "test_metrics_model_c", "test_metrics_xgb", "test_metrics_rf"):
        m = data.get(key, {})
        if isinstance(m, dict):
            for k, v in m.items():
                if k == "playoff_metrics" and isinstance(v, dict):
                    for subk, subv in v.items():
                        if isinstance(subv, (int, float)):
                            out[f"{key}_playoff_{subk}"] = subv
                elif isinstance(v, (int, float)):
                    out[f"{key}_{k}"] = v
    return out


def get_rmse(metrics: dict) -> float | None:
    """Return RMSE value for ranking (lower is better). Prefer playoff outcome, then standings."""
    for k in RMSE_KEYS:
        v = metrics.get(k)
        if v is not None and isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    return None


OUTPUT_PARENT = "output"


def iter_output_roots(root: Path, exclude: set[str]) -> list[Path]:
    """List output/outputs* directories under root (i.e. root/output/), excluding given names."""
    parent = root / OUTPUT_PARENT
    if not parent.is_dir():
        return []
    out = []
    for d in sorted(parent.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith("outputs"):
            continue
        if name in exclude:
            continue
        # outputs2, outputs10, outputs11_listmle_standing_rank, etc.
        rest = name[7:]  # after "outputs"
        if rest and (rest[0].isdigit() or rest.startswith("_")):
            out.append(d)
    return out


def process_batch(batch_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Process one sweep batch: keep top 3 + worst 1 by RMSE, delete rest. Returns (kept, deleted)."""
    combo_dirs = sorted(
        (batch_dir / c.name for c in batch_dir.iterdir() if c.is_dir() and re.match(r"combo_\d{4}$", c.name)),
        key=lambda p: p.name,
    )
    if not combo_dirs:
        return 0, 0

    # Collect (combo_dir, rmse) for combos that have eval_report and valid RMSE
    eval_file = "outputs/eval_report.json"  # under combo dir
    ranked: list[tuple[Path, float]] = []
    for combo_dir in combo_dirs:
        eval_path = combo_dir / eval_file
        metrics = collect_metrics(eval_path)
        rmse = get_rmse(metrics)
        if rmse is not None:
            ranked.append((combo_dir, rmse))

    if not ranked:
        return 0, 0

    # Sort by RMSE ascending (lower is better)
    ranked.sort(key=lambda x: x[1])
    # Indices to keep: top 3 (0,1,2) and worst 1 (last)
    n = len(ranked)
    top3 = ranked[:3]
    worst1 = [ranked[-1]] if n > 0 and (n == 1 or ranked[-1] not in top3) else []
    to_keep = set(p for p, _ in top3) | set(p for p, _ in worst1)
    to_delete = [combo_dir for combo_dir, _ in ranked if combo_dir not in to_keep]

    kept_names = [p.name for p in to_keep]
    for combo_dir in to_delete:
        if dry_run:
            print(f"  [dry-run] would delete {combo_dir.relative_to(ROOT)}", flush=True)
        else:
            try:
                shutil.rmtree(combo_dir)
                print(f"  deleted {combo_dir.relative_to(ROOT)}", flush=True)
            except OSError as e:
                print(f"  error deleting {combo_dir}: {e}", file=sys.stderr, flush=True)

    return len(to_keep), len(to_delete)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retain top 3 + worst 1 by RMSE per sweep batch (except outputs8)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only, do not delete")
    parser.add_argument("--outputs", type=str, default="all", help="Output root (e.g. output/outputs4) or 'all'")
    args = parser.parse_args()

    if args.outputs == "all":
        roots = iter_output_roots(ROOT, exclude={"outputs8"})
    else:
        out_dir = ROOT / args.outputs
        if not out_dir.is_dir():
            print(f"Not a directory: {out_dir}", file=sys.stderr)
            return 1
        if args.outputs.endswith("outputs8") or args.outputs == "outputs8":
            print("outputs8 is excluded from retention; nothing to do.", file=sys.stderr)
            return 0
        roots = [out_dir]

    total_kept = 0
    total_deleted = 0
    for out_root in roots:
        sweeps_dir = out_root / "sweeps"
        if not sweeps_dir.is_dir():
            continue
        for batch_dir in sorted(sweeps_dir.iterdir()):
            if not batch_dir.is_dir():
                continue
            kept, deleted = process_batch(batch_dir, args.dry_run)
            if kept or deleted:
                rel = batch_dir.relative_to(ROOT)
                print(f"{rel}: kept {kept}, deleted {deleted}", flush=True)
            total_kept += kept
            total_deleted += deleted

    print(f"Total: kept {total_kept}, deleted {total_deleted}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
