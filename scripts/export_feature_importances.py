"""Export Model B feature importances from a trained combo (XGB + RF).

Reads xgb_model.joblib and rf_model.joblib from a sweep combo's outputs/,
extracts .feature_importances_ and writes feature_importances.json.
Feature names come from get_team_context_feature_cols(config) using the combo's config.yaml.

Usage:
  python -m scripts.export_feature_importances --config path/to/combo/config.yaml
  python -m scripts.export_feature_importances --combo-dir outputs4/sweeps/phase5_ndcg16_playoff_broad/combo_0002
  python -m scripts.export_feature_importances --combo-dir ... --threshold 0.02  # suggest exclude list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Export XGB/RF feature importances from a sweep combo")
    parser.add_argument("--config", type=str, default=None, help="Path to combo config.yaml")
    parser.add_argument("--combo-dir", type=str, default=None, help="Path to combo dir (e.g. outputs4/sweeps/.../combo_0002)")
    parser.add_argument("--threshold", type=float, default=None, help="If set, suggest exclude_features for importance below this (avg of xgb+rf)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: <combo_dir>/outputs/feature_importances.json or batch/feature_importances_<combo>.json)")
    args = parser.parse_args()

    config_path: Path | None = None
    outputs_dir: Path
    combo_name: str = ""

    if args.combo_dir:
        combo_dir = Path(args.combo_dir)
        if not combo_dir.is_absolute():
            combo_dir = ROOT / combo_dir
        if not combo_dir.is_dir():
            print(f"Combo dir not found: {combo_dir}", file=sys.stderr)
            return 1
        config_path = combo_dir / "config.yaml"
        outputs_dir = combo_dir / "outputs"
        combo_name = combo_dir.name
    elif args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        if not config_path.exists():
            print(f"Config not found: {config_path}", file=sys.stderr)
            return 1
        # Assume config is inside a combo dir (e.g. .../combo_0002/config.yaml)
        outputs_dir = config_path.parent / "outputs"
        combo_name = config_path.parent.name
    else:
        print("Provide --config or --combo-dir.", file=sys.stderr)
        return 1

    if not config_path or not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        return 1
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from src.features.team_context import get_team_context_feature_cols

    feat_cols = get_team_context_feature_cols(config)
    if not feat_cols:
        print("No feature columns from config.", file=sys.stderr)
        return 1

    xgb_path = outputs_dir / "xgb_model.joblib"
    rf_path = outputs_dir / "rf_model.joblib"
    if not xgb_path.exists() or not rf_path.exists():
        print(f"Models not found in {outputs_dir}. Need xgb_model.joblib and rf_model.joblib.", file=sys.stderr)
        return 1

    import joblib

    xgb_model = joblib.load(xgb_path)
    rf_model = joblib.load(rf_path)

    def get_importances(model, name: str) -> dict[str, float]:
        imp = getattr(model, "feature_importances_", None)
        if imp is None:
            return {}
        n = len(imp)
        if n != len(feat_cols):
            print(f"Warning: {name} has {n} importance values but config has {len(feat_cols)} feature names; using first min(n, len(feat_cols)).", file=sys.stderr)
        size = min(n, len(feat_cols))
        return {feat_cols[i]: float(imp[i]) for i in range(size)}

    xgb_imp = get_importances(xgb_model, "XGB")
    rf_imp = get_importances(rf_model, "RF")

    out_obj = {
        "xgb": xgb_imp,
        "rf": rf_imp,
        "feature_names": feat_cols,
        "combo": combo_name,
        "config_path": str(config_path),
    }

    if args.threshold is not None:
        avg_imp = {}
        for f in feat_cols:
            xv = xgb_imp.get(f, 0.0)
            rv = rf_imp.get(f, 0.0)
            avg_imp[f] = (xv + rv) / 2.0
        below = [f for f, v in avg_imp.items() if v < args.threshold]
        above = [f for f, v in avg_imp.items() if v >= args.threshold]
        out_obj["suggested_exclude_features"] = below
        out_obj["suggested_include_features"] = above
        out_obj["threshold"] = args.threshold
        print(f"Threshold {args.threshold}: exclude {len(below)} features, include {len(above)}")
        if below:
            print("Suggested model_b.exclude_features:", below)
        if above:
            print("Suggested model_b.include_features (if using allowlist):", above)

    out_path: Path
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    else:
        out_path = outputs_dir / "feature_importances.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
