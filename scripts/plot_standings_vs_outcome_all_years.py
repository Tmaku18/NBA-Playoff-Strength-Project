"""Plot EOS standings (x) vs playoff outcome rank (y) for all seasons in the dataset on one graph.

Uses config seasons and DB; writes a single PNG with all team-seasons. Run from project root with
  PYTHONPATH set, e.g. python -m scripts.plot_standings_vs_outcome_all_years [--config CONFIG] [--out PATH]

If the DB is not found (e.g. running in WSL while the DB was created on Windows), set the path explicitly:
  NBA_DB_PATH=/path/to/nba_build.duckdb python -m scripts.plot_standings_vs_outcome_all_years
  or on Windows: $env:NBA_DB_PATH = "C:\...\data\processed\nba_build.duckdb"; python -m ...
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

import pandas as pd
import yaml

from src.data.db_loader import load_playoff_data, load_training_data
from src.evaluation.playoffs import compute_eos_final_rank, compute_eos_playoff_standings


def _resolve_db_path(raw: str) -> Path:
    """Resolve DB path from config: support relative paths and Windows paths when running under WSL."""
    raw = " ".join(raw.split()).strip()  # normalize newlines and multiple spaces to single space
    p = Path(raw)
    # Windows-style absolute (e.g. C:/Users/...) on Linux/WSL is not Path.is_absolute(); convert to /mnt/c/...
    if sys.platform != "win32" and re.match(r"^[A-Za-z]:", raw):
        drive = raw[0].lower()
        rest = raw[2:].lstrip("/\\").replace("\\", "/")
        p = Path(f"/mnt/{drive}/{rest}")
    if p.is_absolute():
        return p
    return ROOT / p


def _load_config(config_path: Path | None) -> dict:
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if config_path and config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k] = {**config.get(k, {}), **v}
            else:
                config[k] = v
    return config


def _seasons_from_config(config: dict) -> list[str]:
    seasons_cfg = config.get("seasons") or {}
    out = []
    for k, v in seasons_cfg.items():
        if not isinstance(v, dict) or "start" not in v or "end" not in v:
            continue
        if isinstance(k, str) and len(k) >= 4 and "-" in k:
            out.append(k)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot standings vs outcome for all years (one graph)")
    parser.add_argument("--config", type=str, default=None, help="Config YAML (merged over defaults)")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: docs/standings_vs_outcome_all_years.png)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    if config_path and not config_path.is_absolute():
        config_path = ROOT / config_path
    config = _load_config(config_path)

    env_db = os.environ.get("NBA_DB_PATH", "").strip()
    if env_db:
        db_path = Path(env_db).resolve()
    else:
        db_raw = config.get("paths", {}).get("db", "data/processed/nba_build.duckdb")
        if not isinstance(db_raw, str):
            db_raw = "data/processed/nba_build.duckdb"
        db_path = _resolve_db_path(db_raw)
    if not db_path.exists():
        fallback = ROOT / "data" / "processed" / "nba_build.duckdb"
        if fallback.exists():
            db_path = fallback
        elif MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest_db = (manifest.get("db_path") or "").strip()
                if manifest_db:
                    # manifest may store Windows path; resolve for current OS
                    p = Path(manifest_db)
                    if not p.is_absolute():
                        p = ROOT / p
                    elif sys.platform != "win32" and re.match(r"^[A-Za-z]:", manifest_db):
                        drive = manifest_db[0].lower()
                        rest = manifest_db[2:].lstrip("/\\").replace("\\", "/")
                        p = Path(f"/mnt/{drive}/{rest}")
                    if p.exists():
                        db_path = p.resolve()
            except (json.JSONDecodeError, OSError):
                pass
        if not db_path.exists():
            # last resort: any .duckdb in data/processed
            processed_dir = ROOT / "data" / "processed"
            if processed_dir.is_dir():
                for candidate in sorted(processed_dir.glob("*.duckdb")):
                    if candidate.is_file():
                        db_path = candidate.resolve()
                        break
    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        if env_db:
            print("NBA_DB_PATH was set but path does not exist.", file=sys.stderr)
        print("Tried config path, fallback nba_build.duckdb, manifest db_path, and data/processed/*.duckdb.", file=sys.stderr)
        print("Run 1_download_raw and 2_build_db to create the DB, or set NBA_DB_PATH to an existing DB.", file=sys.stderr)
        sys.exit(1)

    games, tgl, teams, pgl = load_training_data(db_path)
    pg, ptgl, _ = load_playoff_data(db_path)
    if games.empty or tgl.empty:
        print("DB has no games/tgl.", file=sys.stderr)
        sys.exit(1)

    seasons = _seasons_from_config(config)
    if not seasons:
        print("No seasons with start/end in config.seasons.", file=sys.stderr)
        sys.exit(1)

    rows = []
    seasons_cfg = config.get("seasons") or {}
    for season in seasons:
        rng = seasons_cfg.get(season)
        if not rng:
            continue
        ss, se = rng.get("start"), rng.get("end")
        standings = compute_eos_playoff_standings(
            games, tgl, season, season_start=ss, season_end=se
        )
        outcome = compute_eos_final_rank(
            pg, ptgl, games, tgl, season,
            season_start=ss, season_end=se,
        )
        if not outcome or len(outcome) < 16:
            continue
        for tid, stand_r in standings.items():
            out_r = outcome.get(tid)
            if out_r is not None:
                rows.append({"season": season, "standings_rank": stand_r, "outcome_rank": out_r})

    if not rows:
        print("No (standings, outcome) pairs found. Check playoff data and config seasons.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    # Numeric year for colormap (e.g. 2024 for "2023-24")
    def season_to_year(s: str) -> int:
        try:
            return int(s.split("-")[1])  # "2023-24" -> 24 -> use as 2000+24
        except Exception:
            return 0
    df["year"] = df["season"].apply(lambda s: 2000 + int(s.split("-")[1]) if len(s.split("-")) > 1 and s.split("-")[1].isdigit() else 0)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    seasons_sorted = sorted(df["season"].unique())
    cmap = plt.get_cmap("viridis")
    n_seasons = max(len(seasons_sorted), 1)
    for i, season in enumerate(seasons_sorted):
        sub = df[df["season"] == season]
        color = cmap(i / n_seasons)
        ax.scatter(
            sub["standings_rank"],
            sub["outcome_rank"],
            label=season,
            alpha=0.7,
            s=28,
            c=[color],
            edgecolors="k",
            linewidths=0.3,
        )
    max_r = max(df["standings_rank"].max(), df["outcome_rank"].max(), 30) + 1
    ax.plot([0, max_r], [0, max_r], "k--", alpha=0.6, linewidth=1.5, label="identity (agreement)")
    ax.set_xlabel("EOS standings rank (1–30)")
    ax.set_ylabel("Playoff outcome rank (1–30)")
    ax.set_title("Standings vs outcome — all years (original dataset)")
    ax.set_xlim(-0.5, max_r)
    ax.set_ylim(-0.5, max_r)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    out_path = args.out
    if not out_path:
        out_path = ROOT / "docs" / "standings_vs_outcome_all_years.png"
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
