"""Kaggle SOS/SRS loader. Expects CSV(s) already downloaded (no credentials in repo).

Data source: Kaggle — NBA Season Records from Every Year (boonpalipatana)
https://www.kaggle.com/datasets/boonpalipatana/nba-season-records-from-every-year
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_kaggle_sos_srs(
    path: str | Path,
    *,
    team_col: str = "Team",
    season_col: str = "Season",
    sos_col: str = "SOS",
    srs_col: str = "SRS",
) -> pd.DataFrame:
    """
    Load a Kaggle (or similar) CSV with SOS and SRS.
    Returns DataFrame with normalized team/season keys and sos, srs columns.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: str(c).strip())
    # map to expected names if present
    for dst, candidates in [
        (team_col, ["Team", "team", "TEAM", "Team Abbreviation", "Abbreviation"]),
        (season_col, ["Season", "season", "SEASON", "Year"]),
        (sos_col, ["SOS", "sos", "Strength of Schedule"]),
        (srs_col, ["SRS", "srs", "Simple Rating"]),
    ]:
        for c in candidates:
            if c in df.columns and c != dst:
                df = df.rename(columns={c: dst})
    return df


def normalize_sos_srs_to_team_season(
    df: pd.DataFrame,
    *,
    team_col: str = "Team",
    season_col: str = "Season",
    sos_col: str = "SOS",
    srs_col: str = "SRS",
) -> pd.DataFrame:
    """
    Normalize to team_id/season where possible. If team_col is abbreviation, keep it;
    caller can join to teams.team_id via teams.abbreviation.
    Returns DataFrame with columns: team (or team_abbreviation), season, sos, srs.
    """
    out = df[[c for c in [team_col, season_col, sos_col, srs_col] if c in df.columns]].copy()
    out = out.rename(columns={team_col: "team_abbreviation", season_col: "season", sos_col: "sos", srs_col: "srs"})
    out = out.dropna(subset=["team_abbreviation", "season"], how="all")
    return out


def load_team_records_srs(path: str | Path, teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load Team_Records.csv (Season, Team, SRS). Map Team name to team_id via teams table.
    Strips '*' from Team names. Returns DataFrame with columns: team_id, season, srs, sos.
    When SOS is absent, sos is set equal to srs (plan allows SRS-only).
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["team_id", "season", "srs", "sos"])
    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: str(c).strip())
    for dst, candidates in [
        ("Team", ["Team", "team", "TEAM"]),
        ("Season", ["Season", "season", "SEASON", "Year"]),
        ("SRS", ["SRS", "srs", "Simple Rating"]),
        ("SOS", ["SOS", "sos", "Strength of Schedule"]),
    ]:
        for c in candidates:
            if c in df.columns and c != dst:
                df = df.rename(columns={c: dst})
    if "Team" not in df.columns or "Season" not in df.columns:
        return pd.DataFrame(columns=["team_id", "season", "srs", "sos"])
    df["Team"] = df["Team"].astype(str).str.replace(r"\*$", "", regex=True).str.strip()
    df["Season"] = df["Season"].astype(str).str.strip()
    srs_col = "SRS" if "SRS" in df.columns else "srs"
    df["srs"] = pd.to_numeric(df.get(srs_col, 0), errors="coerce").fillna(0)
    df["sos"] = pd.to_numeric(df.get("SOS", df.get("sos", 0)), errors="coerce")
    df["sos"] = df["sos"].fillna(df["srs"])
    team_id_col = "team_id" if "team_id" in teams_df.columns else "TEAM_ID"
    name_col = "name" if "name" in teams_df.columns else "NAME"
    abbrev_col = "abbreviation" if "abbreviation" in teams_df.columns else "ABBREVIATION"
    teams_map: dict[str, int] = {}
    for _, r in teams_df.iterrows():
        tid = int(r[team_id_col])
        if name_col in teams_df.columns and pd.notna(r.get(name_col)):
            teams_map[str(r[name_col]).strip()] = tid
        if abbrev_col in teams_df.columns and pd.notna(r.get(abbrev_col)):
            teams_map[str(r[abbrev_col]).strip()] = tid
    if not teams_map:
        return pd.DataFrame(columns=["team_id", "season", "srs", "sos"])
    df["team_id"] = df["Team"].map(teams_map)
    df = df.dropna(subset=["team_id"])
    df["team_id"] = df["team_id"].astype(int)
    return df[["team_id", "Season", "srs", "sos"]].rename(columns={"Season": "season"})
