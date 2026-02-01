"""Game-level differential features: Home_Rolling_eFG - Away_Rolling_eFG, etc."""
from __future__ import annotations

import pandas as pd


def build_game_differentials(
    team_rolling_df: pd.DataFrame,
    games: pd.DataFrame,
    *,
    date_col: str = "game_date",
    team_id_col: str = "team_id",
) -> pd.DataFrame:
    """
    Build Home-Away differentials per game. team_rolling_df must have
    (team_id, as_of_date, eFG_L10, DefRtg_L10, won_prev_game).
    Returns one row per game with home_team_id, away_team_id, game_date, and
    eFG_L10_diff (home - away), DefRtg_L10_diff (home - away), won_prev_diff.
    """
    if team_rolling_df.empty or games.empty:
        return pd.DataFrame()

    g = games[["game_id", "game_date", "home_team_id", "away_team_id"]].copy()
    g[date_col] = pd.to_datetime(g[date_col]).dt.date
    team_rolling_df = team_rolling_df.copy()
    team_rolling_df["as_of_date"] = pd.to_datetime(team_rolling_df["as_of_date"]).dt.date

    home_feats = team_rolling_df.rename(columns={
        team_id_col: "home_team_id",
        "as_of_date": date_col,
        "eFG_L10": "home_eFG_L10",
        "DefRtg_L10": "home_DefRtg_L10",
        "won_prev_game": "home_won_prev",
    })
    away_feats = team_rolling_df.rename(columns={
        team_id_col: "away_team_id",
        "as_of_date": date_col,
        "eFG_L10": "away_eFG_L10",
        "DefRtg_L10": "away_DefRtg_L10",
        "won_prev_game": "away_won_prev",
    })

    out = g.merge(
        home_feats[["home_team_id", date_col, "home_eFG_L10", "home_DefRtg_L10", "home_won_prev"]],
        left_on=["home_team_id", date_col],
        right_on=["home_team_id", date_col],
        how="left",
    )
    out = out.merge(
        away_feats[["away_team_id", date_col, "away_eFG_L10", "away_DefRtg_L10", "away_won_prev"]],
        left_on=["away_team_id", date_col],
        right_on=["away_team_id", date_col],
        how="left",
        suffixes=("", "_away"),
    )
    out["eFG_L10_diff"] = out["home_eFG_L10"].fillna(0.5) - out["away_eFG_L10"].fillna(0.5)
    out["DefRtg_L10_diff"] = out["home_DefRtg_L10"].fillna(110) - out["away_DefRtg_L10"].fillna(110)
    out["won_prev_diff"] = out["home_won_prev"].fillna(0.5) - out["away_won_prev"].fillna(0.5)

    return out
