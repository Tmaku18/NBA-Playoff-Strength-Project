"""Team-level rolling features: L10 eFG%, L10 DefRtg, won_prev_game. Strict t-1."""
from __future__ import annotations

import pandas as pd


def compute_team_rolling(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    *,
    window: int = 10,
    date_col: str = "game_date",
    team_id_col: str = "team_id",
    wl_col: str = "wl",
) -> pd.DataFrame:
    """
    Compute team-level L10 eFG, DefRtg, and won_prev_game.
    tgl must have game_date (from join with games). Uses shift(1) for t-1.
    DefRtg = pts_allowed per 100 possessions. Need opponent pts from other team in game.
    """
    if tgl.empty or games.empty:
        return pd.DataFrame()

    tgl = tgl.copy()
    if date_col not in tgl.columns and "game_id" in tgl.columns and date_col in games.columns:
        gd = games[["game_id", date_col]].drop_duplicates()
        tgl = tgl.merge(gd, on="game_id", how="left")
    tgl[date_col] = pd.to_datetime(tgl[date_col])
    tgl = tgl.sort_values([team_id_col, date_col])

    # Opponent pts: merge games for home/away, then get other team's pts
    g = games[["game_id", "home_team_id", "away_team_id"]].drop_duplicates()
    tgl = tgl.merge(g, on="game_id", how="left")
    tgl["opp_team_id"] = tgl.apply(
        lambda r: r["away_team_id"] if r[team_id_col] == r["home_team_id"] else r["home_team_id"],
        axis=1,
    )
    opp_pts = tgl[["game_id", team_id_col, "pts"]].rename(
        columns={team_id_col: "opp_team_id", "pts": "opp_pts"}
    )
    tgl = tgl.merge(opp_pts, on=["game_id", "opp_team_id"], how="left")
    tgl["pts_allowed"] = tgl["opp_pts"].fillna(0)

    # Possessions: FGA + 0.44*FTA - ORB + TOV (one team)
    if "fga" in tgl.columns:
        tgl["_poss"] = (
            tgl["fga"].fillna(0) + 0.44 * tgl["fta"].fillna(0)
            - tgl["oreb"].fillna(0) + tgl["tov"].fillna(0)
        )
    else:
        tgl["_poss"] = 100.0
    tgl["_poss"] = tgl["_poss"].replace(0, 1)

    # eFG = (FGM + 0.5*FG3M)/FGA
    if "fgm" in tgl.columns and "fga" in tgl.columns:
        fga = tgl["fga"].fillna(0).replace(0, 1)
        tgl["eFG"] = (tgl["fgm"].fillna(0) + 0.5 * tgl["fg3m"].fillna(0)) / fga
    else:
        tgl["eFG"] = 0.5
    tgl["DefRtg"] = 100 * tgl["pts_allowed"] / tgl["_poss"]

    # Won
    tgl["won"] = (tgl[wl_col].astype(str).str.upper() == "W").astype(float)

    by_team = tgl.groupby(team_id_col, sort=False)
    tgl["eFG_L10"] = by_team["eFG"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    tgl["DefRtg_L10"] = by_team["DefRtg"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
    )
    tgl["won_prev_game"] = by_team["won"].transform(lambda s: s.shift(1))
    tgl["won_prev_game"] = tgl["won_prev_game"].fillna(0.5)

    return tgl


def get_team_rolling_as_of_dates(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    team_dates: list[tuple[int, str]],
    *,
    window: int = 10,
    date_col: str = "game_date",
    team_id_col: str = "team_id",
) -> pd.DataFrame:
    """
    Return eFG_L10, DefRtg_L10, won_prev_game per (team_id, as_of_date).
    Uses latest row before as_of_date for each team.
    """
    if not team_dates:
        return pd.DataFrame(columns=[team_id_col, "as_of_date", "eFG_L10", "DefRtg_L10", "won_prev_game"])

    rolled = compute_team_rolling(tgl, games, window=window)
    if rolled.empty:
        return pd.DataFrame(columns=[team_id_col, "as_of_date", "eFG_L10", "DefRtg_L10", "won_prev_game"])

    rolled[date_col] = pd.to_datetime(rolled[date_col]).dt.date
    rows = []
    for tid, as_of in team_dates:
        ad = pd.to_datetime(as_of).date() if isinstance(as_of, str) else as_of
        past = rolled[(rolled[team_id_col] == tid) & (rolled[date_col] < ad)]
        if past.empty:
            rows.append({
                team_id_col: tid, "as_of_date": as_of,
                "eFG_L10": 0.5, "DefRtg_L10": 110.0, "won_prev_game": 0.5,
            })
            continue
        last = past.sort_values(date_col).iloc[-1]
        rows.append({
            team_id_col: tid, "as_of_date": as_of,
            "eFG_L10": float(last["eFG_L10"]),
            "DefRtg_L10": float(last["DefRtg_L10"]),
            "won_prev_game": float(last["won_prev_game"]),
        })
    return pd.DataFrame(rows)
