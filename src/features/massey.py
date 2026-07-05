"""Massey ratings: solve Mr = p (diagonal = games played, off-diagonal = -1, p = point differential)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_massey_per_season(
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    season: str,
    *,
    team_id_col: str = "team_id",
    game_id_col: str = "game_id",
    pts_col: str = "pts",
    season_col: str = "season",
    end_date: object | None = None,
    date_col: str = "game_date",
) -> pd.DataFrame:
    """
    Compute Massey rating per team for one season. M_ii = games_played_i, M_ij = -n_games(i vs j), p = point diff.
    Solve Mr = p (last row = 1s for sum-to-zero). Returns DataFrame with team_id, season, massey_rating.
    end_date: if given, only games with game_date < end_date are used (causal as-of-date rating).
    """
    if games.empty or tgl.empty:
        return pd.DataFrame(columns=[team_id_col, "season", "massey_rating"])
    gsel = games[games[season_col].astype(str) == str(season)].copy()
    if end_date is not None and date_col in gsel.columns:
        cutoff = pd.to_datetime(end_date)
        gsel = gsel[pd.to_datetime(gsel[date_col]) < cutoff]
    g = gsel[[game_id_col, "home_team_id", "away_team_id"]].copy() if not gsel.empty else gsel
    if g.empty or "home_team_id" not in g.columns or "away_team_id" not in g.columns:
        return pd.DataFrame(columns=[team_id_col, "season", "massey_rating"])
    t = tgl[[game_id_col, team_id_col, pts_col]].copy()
    t[team_id_col] = t[team_id_col].astype(int)
    home_pts = g.merge(t, left_on=[game_id_col, "home_team_id"], right_on=[game_id_col, team_id_col], how="inner")[[game_id_col, "home_team_id", "away_team_id", pts_col]].rename(columns={pts_col: "home_pts"})
    away_pts = t.rename(columns={team_id_col: "away_team_id", pts_col: "away_pts"})
    g = home_pts.merge(away_pts, left_on=[game_id_col, "away_team_id"], right_on=[game_id_col, "away_team_id"], how="inner")
    if g.empty:
        return pd.DataFrame(columns=[team_id_col, "season", "massey_rating"])
    team_ids = sorted(set(g["home_team_id"].astype(int).tolist()) | set(g["away_team_id"].astype(int).tolist()))
    n = len(team_ids)
    tid_to_idx = {tid: i for i, tid in enumerate(team_ids)}
    hi = g["home_team_id"].astype(int).map(tid_to_idx).to_numpy()
    ai = g["away_team_id"].astype(int).map(tid_to_idx).to_numpy()
    diff = (g["home_pts"].astype(float) - g["away_pts"].astype(float)).to_numpy()
    M = np.zeros((n, n))
    p_vec = np.zeros(n)
    np.add.at(p_vec, hi, diff)
    np.add.at(p_vec, ai, -diff)
    np.add.at(M, (hi, hi), 1.0)
    np.add.at(M, (ai, ai), 1.0)
    np.add.at(M, (hi, ai), -1.0)
    np.add.at(M, (ai, hi), -1.0)
    M[-1, :] = 1
    p_vec[-1] = 0
    try:
        r = np.linalg.solve(M, p_vec)
    except np.linalg.LinAlgError:
        r, *_ = np.linalg.lstsq(M, p_vec, rcond=None)
    return pd.DataFrame({team_id_col: team_ids, "season": season, "massey_rating": r.tolist()})


def get_massey_as_of_dates(
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    team_dates: list[tuple[int, str]],
    seasons_cfg: dict,
    *,
    team_id_col: str = "team_id",
    date_col: str = "game_date",
    season_col: str = "season",
) -> pd.DataFrame:
    """Return Massey rating per (team_id, as_of_date). Uses season from as_of_date."""
    if not team_dates:
        return pd.DataFrame(columns=[team_id_col, "as_of_date", "massey_rating"])
    if games.empty or tgl.empty:
        return pd.DataFrame(columns=[team_id_col, "as_of_date", "massey_rating"])
    if season_col not in games.columns and "game_date" in games.columns:
        def _game_date_to_season(d):
            if pd.isna(d):
                return None
            dt = pd.to_datetime(d)
            y, m = dt.year, dt.month
            if m >= 10:
                return f"{y}-{str((y + 1) % 100).zfill(2)}"
            return f"{y - 1}-{str(y % 100).zfill(2)}"
        games = games.copy()
        games[season_col] = games["game_date"].apply(_game_date_to_season)
    # Causal: rating for (team, as_of) uses only that season's games before as_of.
    massey_by_season_date: dict[tuple[str, str], pd.DataFrame] = {}
    rows = []
    for tid, as_of in team_dates:
        ad = pd.to_datetime(as_of).date() if isinstance(as_of, str) else as_of
        y, m = ad.year, ad.month
        season = f"{y - 1}-{str(y % 100).zfill(2)}" if m < 10 else f"{y}-{str((y + 1) % 100).zfill(2)}"
        key = (season, str(ad))
        if key not in massey_by_season_date:
            massey_by_season_date[key] = compute_massey_per_season(
                games, tgl, season, season_col=season_col, end_date=ad, date_col=date_col,
            )
        df_season = massey_by_season_date[key]
        if df_season.empty:
            rows.append({team_id_col: tid, "as_of_date": as_of, "massey_rating": 0.0})
            continue
        row = df_season[df_season[team_id_col] == tid]
        if row.empty:
            rows.append({team_id_col: tid, "as_of_date": as_of, "massey_rating": 0.0})
        else:
            rows.append({team_id_col: tid, "as_of_date": as_of, "massey_rating": float(row["massey_rating"].iloc[0])})
    return pd.DataFrame(rows)
