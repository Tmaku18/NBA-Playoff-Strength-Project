"""Motivation vectors: tanking and seeding — days_until_playoffs, elimination_status, late_season."""
from __future__ import annotations

import pandas as pd


def get_motivation_features(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    teams: pd.DataFrame,
    team_dates: list[tuple[int, str]],
    seasons_cfg: dict,
    *,
    date_col: str = "game_date",
    team_id_col: str = "team_id",
    wl_col: str = "wl",
    late_season_games: int = 15,
    playoff_wins_threshold: int = 42,
) -> pd.DataFrame:
    """
    Compute motivation features per (team_id, as_of_date):
    - days_until_playoffs: days from as_of_date to season end (regular season)
    - elimination_status: 1 if mathematically eliminated (wins + games_remaining < playoff_wins_threshold)
    - late_season: 1 if within last late_season_games of season
    - eliminated_x_late_season: elimination_status * late_season
    """
    if not team_dates or not seasons_cfg:
        return pd.DataFrame(columns=[
            team_id_col, "as_of_date",
            "days_until_playoffs", "elimination_status", "late_season", "eliminated_x_late_season",
        ])

    tgl = tgl.copy()
    if date_col not in tgl.columns and "game_id" in tgl.columns and date_col in games.columns:
        gd = games[["game_id", date_col]].drop_duplicates()
        tgl = tgl.merge(gd, on="game_id", how="left")
    tgl[date_col] = pd.to_datetime(tgl[date_col]).dt.date
    tgl["w"] = (tgl[wl_col].astype(str).str.upper() == "W").astype(int)

    games_played = tgl.groupby([team_id_col, date_col]).agg(
        wins=("w", "sum"),
        n_games=("game_id", "nunique"),
    ).reset_index()

    rows = []
    for tid, as_of in team_dates:
        ad = pd.to_datetime(as_of).date() if isinstance(as_of, str) else as_of
        season = _season_for_date(ad, seasons_cfg)
        season_end = pd.to_datetime(seasons_cfg.get(season, {}).get("end", "2099-12-31")).date()
        days_left = (season_end - ad).days
        days_until_playoffs = max(0, days_left)

        past = games_played[
            (games_played[team_id_col] == tid) &
            (games_played[date_col] < ad)
        ]
        wins_to_date = int(past["wins"].sum()) if not past.empty else 0
        n_to_date = int(past["n_games"].sum()) if not past.empty else 0
        games_remaining = max(0, 82 - n_to_date)

        eliminated = 1 if wins_to_date + games_remaining < playoff_wins_threshold else 0
        late_season = 1 if games_remaining <= late_season_games else 0
        eliminated_x_late = eliminated * late_season

        rows.append({
            team_id_col: tid,
            "as_of_date": as_of,
            "days_until_playoffs": days_until_playoffs,
            "elimination_status": eliminated,
            "late_season": late_season,
            "eliminated_x_late_season": eliminated_x_late,
        })
    return pd.DataFrame(rows)


def _season_for_date(d: object, seasons_cfg: dict) -> str | None:
    d = pd.to_datetime(d).date() if d is not None else None
    if d is None or not seasons_cfg:
        return None
    for season, rng in seasons_cfg.items():
        start = pd.to_datetime(rng.get("start")).date()
        end = pd.to_datetime(rng.get("end")).date()
        if start <= d <= end:
            return season
    return None
