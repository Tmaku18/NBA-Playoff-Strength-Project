"""Load RAPTOR (FiveThirtyEight) player metrics. Maps BRef IDs to our player_id via name matching."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _normalize_name(name: str) -> str:
    """Normalize for matching: strip Jr./III, lowercase."""
    s = str(name).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r",?\s*(Jr\.?|III?|IV|II)\s*$", "", s, flags=re.I)
    return s.lower()


def _raptor_season_to_ours(yr: str) -> str:
    """RAPTOR '2017' -> our '2017-18'."""
    yr = str(yr).strip()
    if "-" in yr:
        return yr
    try:
        y = int(yr)
        return f"{y}-{str((y + 1) % 100).zfill(2)}"
    except ValueError:
        return yr


def load_raptor_by_player(
    path: str | Path,
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load RAPTOR CSV (historical or modern by_player/by_team). Map to our player_id.
    Expects: player_name, player_id (BRef), season, raptor_offense, raptor_defense, raptor_total.
    For by_team files: filter season_type=RS, use first row per (player, season) when multiple teams.
    Returns: player_id (NBA), season (our format), raptor_offense, raptor_defense, raptor_total.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "season", "raptor_offense", "raptor_defense", "raptor_total"])

    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: str(c).strip())
    for dst, cands in [
        ("player_name", ["player_name", "Player Name"]),
        ("season", ["season", "Season"]),
        ("raptor_offense", ["raptor_offense", "raptor_offense"]),
        ("raptor_defense", ["raptor_defense", "raptor_defense"]),
        ("raptor_total", ["raptor_total", "raptor_total"]),
        ("season_type", ["season_type", "season_type"]),
    ]:
        for c in cands:
            if c in df.columns and c != dst:
                df = df.rename(columns={c: dst})
    if "player_name" not in df.columns or "season" not in df.columns:
        return pd.DataFrame(columns=["player_id", "season", "raptor_offense", "raptor_defense", "raptor_total"])

    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper() == "RS"].copy()
    df["season"] = df["season"].astype(str).apply(_raptor_season_to_ours)
    df["_name_norm"] = df["player_name"].apply(_normalize_name)

    name_col = "player_name" if "player_name" in players_df.columns else "PLAYER_NAME"
    pid_col = "player_id" if "player_id" in players_df.columns else "PLAYER_ID"
    if name_col not in players_df.columns or pid_col not in players_df.columns:
        return pd.DataFrame(columns=["player_id", "season", "raptor_offense", "raptor_defense", "raptor_total"])

    players_df = players_df.copy()
    players_df["_name_norm"] = players_df[name_col].astype(str).apply(_normalize_name)
    name_to_nba_id = dict(zip(players_df["_name_norm"], players_df[pid_col].astype(int)))

    df["player_id"] = df["_name_norm"].map(name_to_nba_id)
    df = df.dropna(subset=["player_id"])
    df["player_id"] = df["player_id"].astype(int)

    agg_cols = ["raptor_offense", "raptor_defense", "raptor_total"]
    for c in agg_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[agg_cols] = df[agg_cols].fillna(0)

    out = df.groupby(["player_id", "season"], as_index=False)[agg_cols].mean()
    return out
