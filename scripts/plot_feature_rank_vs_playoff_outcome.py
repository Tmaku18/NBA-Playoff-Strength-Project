"""Plot feature rank (1-15 within conference) vs playoff outcome rank (1-15 within conference) for all seasons.

For each input feature (eFG, TOV_pct, FT_rate, ORB_pct, pace, standing_rank_norm, etc.),
produces one file with two side-by-side charts: East and West. All years in the dataset are plotted on each chart.

Run from project root with PYTHONPATH set, e.g.:
  python -m scripts.plot_feature_rank_vs_playoff_outcome [--config CONFIG] [--out-dir DIR]

If the DB is not found, set NBA_DB_PATH explicitly (see plot_standings_vs_outcome_all_years).
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
import pandas as pd
import yaml

from src.data.db_loader import load_playoff_data, load_training_data
from src.evaluation.playoffs import compute_playoff_performance_rank
from src.features.team_context import (
    build_team_context_as_of_dates,
    get_team_context_feature_cols,
)

# Rank 1 = best. Lower value is better for these features.
LOWER_IS_BETTER: set[str] = {"TOV_pct"}

# Fallback when teams.conference is null (same as build_lists)
TEAM_CONFERENCE: dict[str, str] = {
    "BOS": "E", "BKN": "E", "NYK": "E", "PHI": "E", "TOR": "E",
    "CHI": "E", "CLE": "E", "DET": "E", "IND": "E", "MIL": "E",
    "ATL": "E", "CHA": "E", "MIA": "E", "ORL": "E", "WAS": "E",
    "DAL": "W", "HOU": "W", "MEM": "W", "NOP": "W", "SAS": "W",
    "DEN": "W", "MIN": "W", "OKC": "W", "POR": "W", "UTA": "W",
    "GSW": "W", "LAC": "W", "LAL": "W", "PHX": "W", "SAC": "W",
}


def _resolve_db_path(raw: str) -> Path:
    """Resolve DB path from config; support Windows paths when running under WSL."""
    raw = " ".join(raw.split()).strip()
    p = Path(raw)
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


def _season_for_date(d: object, seasons_cfg: dict) -> str | None:
    """Return season key (e.g. '2023-24') if d falls within that season's range."""
    d = pd.to_datetime(d).date() if d is not None else None
    if d is None or not seasons_cfg:
        return None
    for season, rng in seasons_cfg.items():
        start = pd.to_datetime(rng.get("start")).date()
        end = pd.to_datetime(rng.get("end")).date()
        if start <= d <= end:
            return season
    return None


def _team_ids_in_season(games: pd.DataFrame, tgl: pd.DataFrame, season_start: str, season_end: str) -> list[int]:
    """Return sorted list of team_ids that played in the given season date range."""
    start_d = pd.to_datetime(season_start).date()
    end_d = pd.to_datetime(season_end).date()
    g = games.copy()
    if "game_date" not in g.columns:
        return []
    g["_gd"] = pd.to_datetime(g["game_date"]).dt.date
    g = g[(g["_gd"] >= start_d) & (g["_gd"] <= end_d)]
    game_ids = set(g["game_id"].astype(str).tolist())
    if not game_ids:
        return []
    t = tgl[tgl["game_id"].astype(str).isin(game_ids)]
    if t.empty or "team_id" not in t.columns:
        return []
    return sorted(t["team_id"].astype(int).unique().tolist())


def build_feature_rank_data(
    config: dict,
    db_path: Path,
    *,
    games: pd.DataFrame | None = None,
    tgl: pd.DataFrame | None = None,
    teams: pd.DataFrame | None = None,
    pgl: pd.DataFrame | None = None,
    pg: pd.DataFrame | None = None,
    ptgl: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load DB (if frames not provided), build per-season feature table with feature ranks and
    playoff outcome rank within conference (1-15). Returns (feat_df, plot_cols).
    """
    if games is None or tgl is None:
        games, tgl, teams, pgl = load_training_data(db_path)
        pg, ptgl, _ = load_playoff_data(db_path)
    if teams is None:
        _, _, teams, _ = load_training_data(db_path)
    if pgl is None:
        _, _, _, pgl = load_training_data(db_path)
    if pg is None or ptgl is None:
        pg, ptgl, _ = load_playoff_data(db_path)

    seasons = _seasons_from_config(config)
    if not seasons:
        return pd.DataFrame(), []
    seasons_cfg = config.get("seasons") or {}

    team_dates: list[tuple[int, str]] = []
    for season in seasons:
        rng = seasons_cfg.get(season)
        if not rng:
            continue
        ss, se = rng.get("start"), rng.get("end")
        if not ss or not se:
            continue
        as_of_date = str(pd.to_datetime(se).date())
        for tid in _team_ids_in_season(games, tgl, ss, se):
            team_dates.append((tid, as_of_date))
    if not team_dates:
        return pd.DataFrame(), []

    feat_df = build_team_context_as_of_dates(
        tgl, games, team_dates, config=config, teams=teams, pgl=pgl
    )
    if feat_df.empty:
        return pd.DataFrame(), []

    feat_df["season"] = feat_df["as_of_date"].apply(lambda d: _season_for_date(d, seasons_cfg))
    feat_df = feat_df[feat_df["season"].notna()].copy()

    if "conference" in teams.columns and teams["conference"].notna().any():
        team_conf = teams[["team_id", "conference"]].drop_duplicates()
        team_conf["team_id"] = team_conf["team_id"].astype(int)
        feat_df = feat_df.merge(team_conf, on="team_id", how="left")
        feat_df["conference"] = feat_df["conference"].fillna("")
    else:
        feat_df["conference"] = ""
    if feat_df["conference"].eq("").any():
        abbr_col = "abbreviation" if "abbreviation" in teams.columns else "abbrev"
        if abbr_col in teams.columns:
            team_abbr = teams[["team_id", abbr_col]].drop_duplicates()
            team_abbr["team_id"] = team_abbr["team_id"].astype(int)
            feat_df = feat_df.merge(team_abbr, on="team_id", how="left")
            feat_df["_conf_fb"] = feat_df[abbr_col].map(TEAM_CONFERENCE)
            feat_df["conference"] = feat_df["conference"].where(
                feat_df["conference"].ne(""), feat_df["_conf_fb"]
            )
            feat_df = feat_df.drop(columns=["_conf_fb", abbr_col], errors="ignore")
        feat_df["conference"] = feat_df["conference"].fillna("E")

    feat_df["playoff_outcome_rank"] = float("nan")
    for season in feat_df["season"].unique():
        rng = seasons_cfg.get(season)
        if not rng:
            continue
        ss, se = rng.get("start"), rng.get("end")
        season_team_ids = feat_df.loc[feat_df["season"] == season, "team_id"].unique().tolist()
        rank_map = compute_playoff_performance_rank(
            pg, ptgl, games, tgl, season,
            all_team_ids=season_team_ids, season_start=ss, season_end=se,
        )
        if not rank_map:
            continue
        mask = feat_df["season"] == season
        feat_df.loc[mask, "playoff_outcome_rank"] = feat_df.loc[mask, "team_id"].map(rank_map)

    feat_df = feat_df[feat_df["playoff_outcome_rank"].notna()].copy()
    if feat_df.empty:
        return pd.DataFrame(), []

    feat_df["playoff_outcome_rank_conf"] = feat_df.groupby(["season", "conference"])["playoff_outcome_rank"].rank(
        ascending=True, method="min"
    ).astype(int)

    all_feat_cols = get_team_context_feature_cols(config)
    plot_cols = [c for c in all_feat_cols if c in feat_df.columns and pd.api.types.is_numeric_dtype(feat_df[c])]
    for col in plot_cols:
        ascending = col in LOWER_IS_BETTER
        feat_df[f"{col}_rank"] = feat_df.groupby(["season", "conference"])[col].rank(
            ascending=ascending, method="min"
        ).astype(int)
    return feat_df, plot_cols


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature rank vs playoff outcome rank (East/West, all years)"
    )
    parser.add_argument("--config", type=str, default=None, help="Config YAML (merged over defaults)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for PNGs (default: docs/feature_rank_vs_playoff_outcome)",
    )
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
            for candidate in sorted((ROOT / "data" / "processed").glob("*.duckdb")):
                if candidate.is_file():
                    db_path = candidate.resolve()
                    break
    if not db_path.exists():
        print("Database not found. Run 1_download_raw and 2_build_db or set NBA_DB_PATH.", file=sys.stderr)
        sys.exit(1)

    feat_df, plot_cols = build_feature_rank_data(config, db_path)
    if not plot_cols or feat_df.empty:
        print("No feature data to plot. Check playoff data and config seasons.", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir
    if not out_dir:
        out_dir = ROOT / "docs" / "feature_rank_vs_playoff_outcome"
    else:
        out_dir = Path(out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons_sorted = sorted(feat_df["season"].unique())
    n_seasons = max(len(seasons_sorted), 1)
    cmap = plt.get_cmap("viridis")

    for feature in plot_cols:
        rank_col = f"{feature}_rank"
        safe_name = feature.replace("%", "pct").replace(" ", "_")
        fig, (ax_east, ax_west) = plt.subplots(1, 2, figsize=(14, 6))
        for ax, conf_name, conf_code in [(ax_east, "East", "E"), (ax_west, "West", "W")]:
            sub = feat_df[feat_df["conference"] == conf_code].copy()
            if sub.empty:
                ax.set_title(f"{feature} — {conf_name} (no data)")
                continue
            for i, season in enumerate(seasons_sorted):
                row = sub[sub["season"] == season]
                if row.empty:
                    continue
                color = cmap(i / n_seasons)
                ax.scatter(
                    row[rank_col],
                    row["playoff_outcome_rank_conf"],
                    label=season,
                    alpha=0.7,
                    s=28,
                    c=[color],
                    edgecolors="k",
                    linewidths=0.3,
                )
            ax.set_xlabel(f"Feature rank (1–15)")
            ax.set_ylabel("Playoff outcome rank (1–15)")
            ax.set_title(f"{feature} — {conf_name}")
            ax.set_xlim(-0.5, 16)
            ax.set_ylim(-0.5, 16)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, ncol=1)
        fig.suptitle(f"{feature} rank vs playoff outcome rank (within conference) — all years", fontsize=11)
        fig.tight_layout(rect=[0, 0, 0.88, 0.96])
        out_path = out_dir / f"{safe_name}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}", flush=True)

    print(f"Done. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
