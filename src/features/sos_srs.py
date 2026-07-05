"""Causal to-date SRS/SOS.

SRS is defined by rating_i = avg_margin_i + avg(opponent ratings), which is the
same linear system as the Massey rating (D - N) r = p_total with a sum-to-zero
constraint. So SRS-to-date = Massey rating computed on games before the as-of
date, and SOS = SRS - average point margin to date.

This replaces the previous end-of-season Kaggle values, which leaked the full
season's results into mid-season feature rows.
"""
from __future__ import annotations

import pandas as pd

from .massey import compute_massey_per_season


def get_sos_srs_as_of_dates(
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    team_dates: list[tuple[int, str]],
    seasons_cfg: dict,
    *,
    team_id_col: str = "team_id",
    date_col: str = "game_date",
    season_col: str = "season",
    pts_col: str = "pts",
) -> pd.DataFrame:
    """Return srs and sos per (team_id, as_of_date), using only games before as_of within the season."""
    if not team_dates or games.empty or tgl.empty:
        return pd.DataFrame(columns=[team_id_col, "as_of_date", "srs", "sos"])
    if season_col not in games.columns and date_col in games.columns:
        def _game_date_to_season(d):
            if pd.isna(d):
                return None
            dt = pd.to_datetime(d)
            y, m = dt.year, dt.month
            if m >= 10:
                return f"{y}-{str((y + 1) % 100).zfill(2)}"
            return f"{y - 1}-{str(y % 100).zfill(2)}"
        games = games.copy()
        games[season_col] = games[date_col].apply(_game_date_to_season)

    # Precompute per-game margins joined with dates for avg-margin-to-date
    g = games[["game_id", date_col, season_col, "home_team_id", "away_team_id"]].copy()
    t = tgl[["game_id", team_id_col, pts_col]].copy()
    t[team_id_col] = t[team_id_col].astype(int)
    home = g.merge(t, left_on=["game_id", "home_team_id"], right_on=["game_id", team_id_col], how="inner")
    home = home.rename(columns={pts_col: "home_pts"})
    away = t.rename(columns={team_id_col: "away_team_id", pts_col: "away_pts"})
    g = home.merge(away, on=["game_id", "away_team_id"], how="inner")
    g["_d"] = pd.to_datetime(g[date_col])

    cache: dict[tuple[str, str], tuple[dict[int, float], dict[int, float]]] = {}
    rows = []
    for tid, as_of in team_dates:
        ad = pd.to_datetime(as_of)
        y, m = ad.year, ad.month
        season = f"{y - 1}-{str(y % 100).zfill(2)}" if m < 10 else f"{y}-{str((y + 1) % 100).zfill(2)}"
        key = (season, str(ad.date()))
        if key not in cache:
            srs_df = compute_massey_per_season(
                games, tgl, season, season_col=season_col, end_date=ad, date_col=date_col,
            )
            srs_map = {int(r[team_id_col]): float(r["massey_rating"]) for _, r in srs_df.iterrows()}
            gs = g[(g[season_col].astype(str) == season) & (g["_d"] < ad)]
            margin_map: dict[int, float] = {}
            if not gs.empty:
                diff = gs["home_pts"].astype(float) - gs["away_pts"].astype(float)
                per_team = pd.concat([
                    pd.DataFrame({team_id_col: gs["home_team_id"].astype(int), "margin": diff}),
                    pd.DataFrame({team_id_col: gs["away_team_id"].astype(int), "margin": -diff}),
                ])
                margin_map = per_team.groupby(team_id_col)["margin"].mean().to_dict()
            cache[key] = (srs_map, margin_map)
        srs_map, margin_map = cache[key]
        srs = srs_map.get(int(tid), 0.0)
        sos = srs - margin_map.get(int(tid), 0.0)
        rows.append({team_id_col: tid, "as_of_date": as_of, "srs": srs, "sos": sos})
    return pd.DataFrame(rows)
