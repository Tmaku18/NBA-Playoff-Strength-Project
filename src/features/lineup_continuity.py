"""Lineup continuity: pct_min_returning per team as-of date."""
from __future__ import annotations

import pandas as pd


def get_prior_season_roster(
    pgl: pd.DataFrame,
    team_id: int,
    season_start: str | pd.Timestamp,
    *,
    date_col: str = "game_date",
    player_id_col: str = "player_id",
    team_id_col: str = "team_id",
    min_col: str = "min",
) -> set[int]:
    """Return set of player_ids who played (min > 0) for team_id in prior season."""
    pgl = pgl.copy()
    pgl[date_col] = pd.to_datetime(pgl[date_col])
    ss = pd.to_datetime(season_start)
    start = ss - pd.Timedelta(days=400)  # ~prior season
    end = ss - pd.Timedelta(days=1)
    mask = (pgl[team_id_col] == team_id) & (pgl[date_col] >= start) & (pgl[date_col] <= end)
    prior = pgl.loc[mask]
    if prior.empty:
        return set()
    played = prior[prior[min_col].fillna(0) > 0]
    return set(played[player_id_col].unique().astype(int))


def pct_min_returning_per_team(
    pgl: pd.DataFrame,
    games: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    team_ids: list[int] | None = None,
    *,
    date_col: str = "game_date",
    player_id_col: str = "player_id",
    team_id_col: str = "team_id",
    min_col: str = "min",
    season_start: str | pd.Timestamp | None = None,
) -> dict[int, float]:
    """
    For each team, compute pct_min_returning = (minutes from returning players) / (total minutes)
    as-of date. Returning = players who were on this team in the prior season.
    Uses games for season boundaries when provided.
    """
    ad = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    pgl = pgl.copy()
    pgl[date_col] = pd.to_datetime(pgl[date_col])
    dates = pgl[date_col].dt.date
    mask = dates < ad
    if season_start is not None:
        ss = pd.to_datetime(season_start).date() if isinstance(season_start, str) else season_start
        mask &= dates >= ss
    past = pgl.loc[mask]
    if past.empty:
        return {tid: 0.0 for tid in (team_ids or [])}

    if team_ids is None:
        team_ids = past[team_id_col].unique().astype(int).tolist()

    out: dict[int, float] = {}
    for tid in team_ids:
        team_games = past[past[team_id_col] == tid]
        if team_games.empty:
            out[tid] = 0.0
            continue
        team_games = team_games[team_games[min_col].fillna(0) > 0]
        if team_games.empty:
            out[tid] = 0.0
            continue
        total_min = team_games[min_col].sum()
        if total_min <= 0:
            out[tid] = 0.0
            continue

        returning = get_prior_season_roster(pgl, int(tid), season_start) if season_start else set()
        returning_min = team_games[team_games[player_id_col].isin(returning)][min_col].sum()
        out[tid] = float(returning_min / total_min)

    return out
