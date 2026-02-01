"""Elo rating with cold start: seed first N games with prior season Elo regressed to mean."""
from __future__ import annotations

import pandas as pd


DEFAULT_MEAN = 1500.0
DEFAULT_K = 20.0
DEFAULT_RATING_SCALE = 400.0


def compute_elo_per_game(
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    seasons_cfg: dict,
    *,
    cold_start_games: int = 10,
    regression_to_mean: float = 0.25,
    k_base: float = 20.0,
    rating_scale: float = 400.0,
) -> pd.DataFrame:
    """
    Compute Elo per team per game date. Returns DataFrame with game_id, team_id, game_date, elo.
    Cold start: For first cold_start_games of each season, seed with prior season final Elo
    regressed regression_to_mean toward 1500: R_seed = (1 - r) * R_prior + r * 1500.
    Update: R_new = R_old + K * (S_actual - S_expected).
    """
    if games.empty or tgl.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "game_date", "elo"])

    g = games[["game_id", "game_date", "season", "home_team_id", "away_team_id"]].copy()
    g["game_date"] = pd.to_datetime(g["game_date"]).dt.date
    g = g.sort_values("game_date").reset_index(drop=True)

    # Get winner from tgl (home team row: wl W = home won)
    home_tgl = tgl.merge(
        g[["game_id", "home_team_id", "away_team_id"]],
        on="game_id",
        how="inner",
        suffixes=("", "_g"),
    )
    home_tgl = home_tgl[home_tgl["team_id"] == home_tgl["home_team_id"]]
    home_tgl = home_tgl[["game_id", "wl"]].rename(columns={"wl": "home_wl"})
    g = g.merge(home_tgl, on="game_id", how="left")
    g["home_won"] = (g["home_wl"].astype(str).str.upper() == "W").astype(float)

    season_order = _season_order(seasons_cfg)
    elo: dict[int, float] = {}
    rows: list[dict] = []
    games_played_this_season: dict[tuple[str, int], int] = {}
    seen_this_season: set[tuple[str, int]] = set()

    for _, row in g.iterrows():
        gid = row["game_id"]
        d = row["game_date"]
        season = str(row.get("season", ""))
        home = int(row["home_team_id"])
        away = int(row["away_team_id"])
        home_won = row.get("home_won", 0.5)

        def games_played(tid: int) -> int:
            return games_played_this_season.get((season, tid), 0)

        # Cold start: for first game of season, regress prior Elo toward mean
        for tid in (home, away):
            if tid not in elo:
                prior_elo = _prior_season_final_elo(elo, tid, season, season_order, g)
                r = regression_to_mean
                elo[tid] = (1 - r) * prior_elo + r * DEFAULT_MEAN
            elif (season, tid) not in seen_this_season and games_played(tid) == 0:
                r = regression_to_mean
                elo[tid] = (1 - r) * elo[tid] + r * DEFAULT_MEAN
            seen_this_season.add((season, tid))

        R_home = elo[home]
        R_away = elo[away]
        S_expected_home = 1.0 / (1.0 + 10 ** (-(R_home - R_away) / rating_scale))
        S_actual = home_won
        delta = S_actual - S_expected_home

        # Dynamic K: higher early in season
        k = _dynamic_k(games_played(home) + games_played(away), k_base)
        elo[home] = elo[home] + k * delta
        elo[away] = elo[away] - k * delta

        for tid in (home, away):
            games_played_this_season[(season, tid)] = games_played(tid) + 1
        rows.append(
            {"game_id": gid, "team_id": home, "game_date": d, "elo": elo[home]}
        )
        rows.append(
            {"game_id": gid, "team_id": away, "game_date": d, "elo": elo[away]}
        )

    return pd.DataFrame(rows)


def _season_order(seasons_cfg: dict) -> list[str]:
    """Return seasons sorted by start date."""
    def start(s: str) -> str:
        return str(seasons_cfg.get(s, {}).get("start", ""))
    return sorted(seasons_cfg.keys(), key=start)


def _prior_season_final_elo(
    elo: dict[int, float],
    team_id: int,
    current_season: str,
    season_order: list[str],
    _games_df: pd.DataFrame,
) -> float:
    """Get team's final Elo from prior season. Elo dict holds latest after each game."""
    return elo.get(team_id, DEFAULT_MEAN)


def _dynamic_k(games_played: int, k_base: float) -> float:
    """Higher K early in season."""
    if games_played < 10:
        return k_base * 1.5
    if games_played < 30:
        return k_base
    return k_base * 0.8


def get_elo_as_of_dates(
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    team_dates: list[tuple[int, str]],
    seasons_cfg: dict,
    *,
    date_col: str = "game_date",
    team_id_col: str = "team_id",
    cold_start_games: int = 10,
    regression_to_mean: float = 0.25,
) -> pd.DataFrame:
    """
    Return Elo per (team_id, as_of_date). Uses latest Elo before as_of_date for each team.
    """
    if not team_dates:
        return pd.DataFrame(columns=[team_id_col, "as_of_date", "elo"])
    elo_per_game = compute_elo_per_game(
        games, tgl, seasons_cfg,
        cold_start_games=cold_start_games,
        regression_to_mean=regression_to_mean,
    )
    if elo_per_game.empty:
        return pd.DataFrame(columns=[team_id_col, "as_of_date", "elo"])
    elo_per_game[date_col] = pd.to_datetime(elo_per_game[date_col]).dt.date
    rows = []
    for tid, as_of in team_dates:
        ad = pd.to_datetime(as_of).date() if isinstance(as_of, str) else as_of
        past = elo_per_game[
            (elo_per_game[team_id_col] == tid) &
            (elo_per_game[date_col] < ad)
        ]
        if past.empty:
            rows.append({team_id_col: tid, "as_of_date": as_of, "elo": DEFAULT_MEAN})
            continue
        last = past.sort_values(date_col).iloc[-1]
        rows.append({team_id_col: tid, "as_of_date": as_of, "elo": float(last["elo"])})
    return pd.DataFrame(rows)
