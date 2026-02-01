"""Lineup-based +/- approximation (RAPM-lite). Without stint data, approximate on-court contribution."""
from __future__ import annotations

import pandas as pd


def compute_on_court_pm_per_game(
    pgl: pd.DataFrame,
    tgl: pd.DataFrame,
    games: pd.DataFrame | None = None,
    *,
    game_id_col: str = "game_id",
    team_id_col: str = "team_id",
    player_id_col: str = "player_id",
    min_col: str = "min",
    plus_minus_col: str = "plus_minus",
) -> pd.DataFrame:
    """
    For each (game_id, player_id, team_id), compute approximate on-court plus-minus contribution:
    (player_min / team_total_min) * team_plus_minus.
    Only for players with min > 0. Team total_min = sum of minutes for players on court (min>0) for that team in that game.
    """
    pgl = pgl.copy()
    tgl = tgl.copy()
    pgl[min_col] = pd.to_numeric(pgl[min_col], errors="coerce").fillna(0)
    pgl[plus_minus_col] = pd.to_numeric(pgl[plus_minus_col], errors="coerce").fillna(0)
    tgl[plus_minus_col] = pd.to_numeric(tgl[plus_minus_col], errors="coerce").fillna(0)

    on_court = pgl[pgl[min_col] > 0].copy()
    if on_court.empty:
        return pd.DataFrame(columns=[game_id_col, player_id_col, team_id_col, "on_court_pm_approx"])

    team_min_per_game = on_court.groupby([game_id_col, team_id_col])[min_col].transform("sum")
    on_court["team_min"] = team_min_per_game
    on_court["team_plus_minus"] = on_court.merge(
        tgl[[game_id_col, team_id_col, plus_minus_col]],
        on=[game_id_col, team_id_col],
        how="left",
        suffixes=("", "_tgl"),
    )[plus_minus_col]
    on_court["min_share"] = on_court[min_col] / on_court["team_min"].replace(0, 1)
    on_court["on_court_pm_approx"] = on_court["min_share"] * on_court["team_plus_minus"].fillna(0)

    return on_court[[game_id_col, player_id_col, team_id_col, "on_court_pm_approx"]].copy()


def get_on_court_pm_rolling(
    pgl: pd.DataFrame,
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    *,
    game_id_col: str = "game_id",
    date_col: str = "game_date",
    player_id_col: str = "player_id",
    window: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling average of on_court_pm_approx per player, as-of date (t-1).
    Returns DataFrame with player_id and on_court_pm_approx_L{window}.
    Requires games for game_date lookup.
    """
    per_game = compute_on_court_pm_per_game(pgl, tgl)
    if per_game.empty:
        return pd.DataFrame(columns=[player_id_col, f"on_court_pm_approx_L{window}"])

    games_dates = games[[game_id_col, date_col]].drop_duplicates()
    merged = per_game.merge(games_dates, on=game_id_col, how="left")
    if date_col not in merged.columns or merged[date_col].isna().all():
        return pd.DataFrame(columns=[player_id_col, f"on_court_pm_approx_L{window}"])

    merged = merged.sort_values([player_id_col, date_col])
    merged[date_col] = pd.to_datetime(merged[date_col])
    ad = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    merged = merged[merged[date_col].dt.date < ad]

    col = f"on_court_pm_approx_L{window}"
    rolled = (
        merged.groupby(player_id_col)["on_court_pm_approx"]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
    )
    merged[col] = rolled
    out = merged.groupby(player_id_col, as_index=False)[col].last()
    return out


def get_on_court_pm_as_of_date(
    pgl: pd.DataFrame,
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    *,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Rolling on_court_pm_approx for L10 and L30 (or configured windows).
    Returns one row per player with on_court_pm_approx_L{w} for each window.
    """
    if windows is None:
        windows = [10, 30]
    out: pd.DataFrame | None = None
    for w in windows:
        df = get_on_court_pm_rolling(pgl, tgl, games, as_of_date=as_of_date, window=w)
        if df.empty:
            continue
        col = f"on_court_pm_approx_L{w}"
        if out is None:
            out = df.copy()
        else:
            out = out.merge(df, on="player_id", how="outer")
    if out is None:
        return pd.DataFrame(columns=["player_id"] + [f"on_court_pm_approx_L{w}" for w in windows])
    for w in windows:
        col = f"on_court_pm_approx_L{w}"
        if col not in out.columns:
            out[col] = 0.0
    return out
