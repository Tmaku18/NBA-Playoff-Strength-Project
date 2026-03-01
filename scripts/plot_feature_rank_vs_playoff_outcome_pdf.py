"""Plot 2D probability density (KDE) of feature rank vs playoff outcome rank for each feature.

Uses the same data as plot_feature_rank_vs_playoff_outcome. For each feature, produces one file
with two side-by-side charts (East and West) showing the estimated PDF as filled contours.

Run from project root with PYTHONPATH set, e.g.:
  python -m scripts.plot_feature_rank_vs_playoff_outcome_pdf [--config CONFIG] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "manifest.json"
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gaussian_kde

from scripts.plot_feature_rank_vs_playoff_outcome import (
    _load_config,
    _resolve_db_path,
    build_feature_rank_data,
)


def _resolve_db_from_config(config: dict) -> Path:
    """Resolve DB path using same fallbacks as plot_feature_rank_vs_playoff_outcome."""
    env_db = os.environ.get("NBA_DB_PATH", "").strip()
    if env_db:
        return Path(env_db).resolve()
    db_raw = config.get("paths", {}).get("db", "data/processed/nba_build.duckdb")
    if not isinstance(db_raw, str):
        db_raw = "data/processed/nba_build.duckdb"
    db_path = _resolve_db_path(db_raw)
    if not db_path.exists():
        fallback = ROOT / "data" / "processed" / "nba_build.duckdb"
        if fallback.exists():
            return fallback
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest_db = (manifest.get("db_path") or "").strip()
                if manifest_db:
                    p = Path(manifest_db)
                    if not p.is_absolute():
                        p = ROOT / p
                    elif sys.platform != "win32" and re.match(r"^[A-Za-z]:", manifest_db):
                        drive = manifest_db[0].lower()
                        rest = manifest_db[2:].lstrip("/\\").replace("\\", "/")
                        p = Path(f"/mnt/{drive}/{rest}")
                    if p.exists():
                        return p.resolve()
            except (json.JSONDecodeError, OSError):
                pass
        for candidate in sorted((ROOT / "data" / "processed").glob("*.duckdb")):
            if candidate.is_file():
                return candidate.resolve()
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PDF (2D KDE) of feature rank vs playoff outcome rank (East/West)"
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML (merged over defaults)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: docs/feature_rank_vs_playoff_outcome_pdf)",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if config_path and not config_path.is_absolute():
        config_path = ROOT / config_path
    config = _load_config(config_path)

    db_path = _resolve_db_from_config(config)
    if not db_path.exists():
        print("Database not found. Run 1_download_raw and 2_build_db or set NBA_DB_PATH.", file=sys.stderr)
        sys.exit(1)

    feat_df, plot_cols = build_feature_rank_data(config, db_path)
    if not plot_cols or feat_df.empty:
        print("No feature data to plot.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    if not out_dir:
        out_dir = ROOT / "docs" / "feature_rank_vs_playoff_outcome_pdf"
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Grid for KDE evaluation (feature rank 1-15, playoff outcome rank 1-15)
    x_min, x_max = 0.5, 15.5
    y_min, y_max = 0.5, 15.5
    n_grid = 80
    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    Xi, Yi = np.meshgrid(xi, yi)
    grid_points = np.vstack([Xi.ravel(), Yi.ravel()])

    for feature in plot_cols:
        rank_col = f"{feature}_rank"
        safe_name = feature.replace("%", "pct").replace(" ", "_")
        fig, (ax_east, ax_west) = plt.subplots(1, 2, figsize=(14, 6))
        for ax, conf_name, conf_code in [(ax_east, "East", "E"), (ax_west, "West", "W")]:
            sub = feat_df[feat_df["conference"] == conf_code][[rank_col, "playoff_outcome_rank_conf"]].dropna()
            if sub.shape[0] < 3:
                ax.set_title(f"{feature} — {conf_name} (insufficient data for KDE)")
                ax.set_xlim(x_min - 0.5, x_max + 0.5)
                ax.set_ylim(y_min - 0.5, y_max + 0.5)
                continue
            x_data = sub[rank_col].values.astype(float)
            y_data = sub["playoff_outcome_rank_conf"].values.astype(float)
            try:
                kde = gaussian_kde(np.vstack([x_data, y_data]), bw_method="scott")
                Zi = kde(grid_points).reshape(Xi.shape)
                # Normalize so max is 1 for consistent colormap across plots
                Zi = Zi / (Zi.max() + 1e-12)
                ax.contourf(Xi, Yi, Zi, levels=np.linspace(0, 1, 12), cmap="viridis", alpha=0.85)
                ax.contour(Xi, Yi, Zi, levels=5, colors="k", linewidths=0.4, alpha=0.5)
            except np.linalg.LinAlgError:
                ax.text(0.5, 0.5, "KDE failed (e.g. singular matrix)", ha="center", va="center", transform=ax.transAxes)
            ax.scatter(x_data, y_data, s=12, c="white", edgecolors="k", linewidths=0.3, alpha=0.6, zorder=5)
            ax.set_xlabel("Feature rank (1–15)")
            ax.set_ylabel("Playoff outcome rank (1–15)")
            ax.set_title(f"{feature} — {conf_name}")
            ax.set_xlim(x_min - 0.5, x_max + 0.5)
            ax.set_ylim(y_min - 0.5, y_max + 0.5)
            ax.set_aspect("equal")
            ax.grid(True, linestyle="--", alpha=0.5)
        fig.suptitle(f"{feature} — probability density (feature rank vs playoff outcome rank)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = out_dir / f"{safe_name}_pdf.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}", flush=True)

    print(f"Done. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
