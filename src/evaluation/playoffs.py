"""Playoff performance ground truth: wins per team, rank 1-30 (playoff wins, tiebreak reg-season win %, lottery 17-30)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

MIN_PLAYOFF_TEAMS = 16


def _to_date(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.to_datetime(value).date()


def _filter_by_date(
    df: pd.DataFrame,
    *,
    date_col: str,
    season_start: str | pd.Timestamp | None,
    season_end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    start = _to_date(season_start)
    end = _to_date(season_end)
    if start is None or end is None:
        return df
    dates = pd.to_datetime(df[date_col]).dt.date
    return df[(dates >= start) & (dates <= end)]


def _filtered_playoff_tgl(
    playoff_games: pd.DataFrame,
    playoff_tgl: pd.DataFrame,
    season: str,
    *,
    season_start: str | pd.Timestamp | None = None,
    season_end: str | pd.Timestamp | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Filter playoff games for one season. Uses the season column (not date range) because
    playoff games occur AFTER the regular-season end (April 15+); config season_end is
    the regular-season end, so date filtering would incorrectly exclude all playoff games.
    """
    if playoff_games.empty or playoff_tgl.empty:
        return pd.DataFrame()
    pg = playoff_games.copy()
    if "season" not in pg.columns and "game_date" in pg.columns:
        def _game_date_to_season(d):
            dt = pd.to_datetime(d)
            y, m = dt.year, dt.month
            if m >= 10:
                return f"{y}-{str((y + 1) % 100).zfill(2)}"
            return f"{y - 1}-{str(y % 100).zfill(2)}"
        pg = pg.copy()
        pg["season"] = pg["game_date"].apply(_game_date_to_season)
    if "season" in pg.columns:
        season_str = str(season).strip()
        gids = set(pg.loc[pg["season"].astype(str) == season_str, "game_id"].astype(str))
        # Fallback: DB may store "2024" or "24" for "2023-24" (playoff year)
        if not gids and "-" in season_str:
            end_part = season_str.split("-")[-1].strip()
            try:
                end_year = int(end_part)
                if end_year < 100:
                    alt = str(2000 + end_year)  # "24" -> "2024"
                else:
                    alt = str(end_year)
                gids = set(pg.loc[pg["season"].astype(str).isin((alt, end_part, season_str)), "game_id"].astype(str))
            except ValueError:
                pass
        if debug:
            print(
                f"Playoff filtering: season={season}, games_matching={len(gids)}",
                flush=True,
            )
    else:
        gids = set(pg["game_id"].astype(str))
    if not gids:
        return pd.DataFrame()
    pt = playoff_tgl[playoff_tgl["game_id"].astype(str).isin(gids)].copy()
    return pt


def get_playoff_wins(
    playoff_games: pd.DataFrame,
    playoff_tgl: pd.DataFrame,
    season: str,
    *,
    season_start: str | pd.Timestamp | None = None,
    season_end: str | pd.Timestamp | None = None,
) -> dict[int, int]:
    """
    Playoff wins per team for one season (exclude Play-In; we use only games in playoff_* tables).
    Returns team_id -> count of wins (WL='W' in playoff_team_game_logs).
    """
    pt = _filtered_playoff_tgl(
        playoff_games,
        playoff_tgl,
        season,
        season_start=season_start,
        season_end=season_end,
    )
    if pt.empty:
        return {}
    wl_col = "wl" if "wl" in pt.columns else "WL"
    if wl_col not in pt.columns:
        return {}
    pt["win"] = (pt[wl_col].astype(str).str.upper() == "W").astype(int)
    wins = pt.groupby("team_id")["win"].sum().to_dict()
    return {int(k): int(v) for k, v in wins.items()}


def get_reg_season_win_pct(
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    season: str,
    *,
    season_start: str | pd.Timestamp | None = None,
    season_end: str | pd.Timestamp | None = None,
) -> dict[int, float]:
    """Regular-season win % per team for one season. Returns team_id -> win rate (0-1)."""
    if games.empty or tgl.empty:
        return {}
    g = games.copy()
    t = tgl.copy()
    use_range = season_start is not None and season_end is not None
    if use_range:
        g = _filter_by_date(g, date_col="game_date", season_start=season_start, season_end=season_end)
    else:
        if "season" not in g.columns:
            g["game_date"] = pd.to_datetime(g["game_date"])
            g["season"] = g["game_date"].dt.year.astype(str)
        g = g[g["season"].astype(str) == str(season)]
    if g.empty:
        return {}
    gids = set(g["game_id"].astype(str))
    t = t[t["game_id"].astype(str).isin(gids)]
    wl_col = "wl" if "wl" in t.columns else "WL"
    if wl_col not in t.columns:
        return {}
    t["win"] = (t[wl_col].astype(str).str.upper() == "W").astype(int)
    agg = t.groupby("team_id").agg({"win": "sum", "game_id": "count"}).rename(columns={"game_id": "gp"})
    agg["win_pct"] = agg["win"] / agg["gp"].clip(lower=1)
    return agg["win_pct"].to_dict()


def compute_playoff_performance_rank(
    playoff_games: pd.DataFrame,
    playoff_tgl: pd.DataFrame,
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    season: str,
    all_team_ids: list[int] | None = None,
    *,
    season_start: str | pd.Timestamp | None = None,
    season_end: str | pd.Timestamp | None = None,
    debug: bool = False,
) -> dict[int, int]:
    """
    Playoff performance rank 1-30 for one season.
    Phase 1: Rank playoff teams by total playoff wins (desc).
    Phase 2: Tie-break by regular-season win % (desc).
    Phase 3: Teams with 0 playoff wins ranked 17-30 by regular-season win %.
    Returns team_id -> rank (1-30) for the top 30 teams by this scheme.
    """
    pw = get_playoff_wins(
        playoff_games,
        playoff_tgl,
        season,
        season_start=season_start,
        season_end=season_end,
    )
    reg_wp = get_reg_season_win_pct(
        games,
        tgl,
        season,
        season_start=season_start,
        season_end=season_end,
    )
    pt = _filtered_playoff_tgl(
        playoff_games,
        playoff_tgl,
        season,
        season_start=season_start,
        season_end=season_end,
        debug=debug,
    )
    playoff_team_ids = set(pt["team_id"].astype(int).tolist()) if not pt.empty else set()
    if all_team_ids is None:
        all_team_ids = sorted(set(list(pw.keys()) + list(reg_wp.keys())))
    if not all_team_ids:
        return {}
    if len(playoff_team_ids) < MIN_PLAYOFF_TEAMS:
        if debug:
            print(
                f"Warning: Only {len(playoff_team_ids)} playoff teams found (min {MIN_PLAYOFF_TEAMS}). "
                "Skipping playoff rank/metrics.",
                flush=True,
            )
        return {}

    def _safe_wp(tid: int) -> float:
        v = reg_wp.get(tid, 0.0)
        return float(v) if pd.notna(v) else 0.0

    # Playoff teams: anyone who appeared in playoff logs (includes 0-win teams)
    playoff_teams = [tid for tid in all_team_ids if tid in playoff_team_ids]
    playoff_teams.sort(key=lambda t: (-pw.get(t, 0), -_safe_wp(t), t))

    # Lottery teams (0 playoff wins): sort by reg_win_pct desc, team_id asc
    lottery = [tid for tid in all_team_ids if tid not in playoff_team_ids]
    lottery.sort(key=lambda t: (-_safe_wp(t), t))

    playoff_top = playoff_teams[:16]
    lottery_top = lottery[:14]

    out: dict[int, int] = {}
    for r, tid in enumerate(playoff_top, start=1):
        out[tid] = r
    for r, tid in enumerate(lottery_top, start=17):
        out[tid] = r
    return out


def _playoff_round_rank(wins: int) -> int:
    """
    NBA playoff structure (best-of-7, 4 wins to advance) implies strict ordering:
    - 16 = champion; 12-15 = Finals; 8-11 = Conference Finals; 4-7 = second round; 0-3 = first round.
    More wins always means went further, so ranking by wins alone is correct.
    This helper returns a secondary key for tie-break: higher = went further (same win count).
    """
    if wins >= 16:
        return 5
    if wins >= 12:
        return 4
    if wins >= 8:
        return 3
    if wins >= 4:
        return 2
    return 1


def compute_eos_final_rank(
    playoff_games: pd.DataFrame,
    playoff_tgl: pd.DataFrame,
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    season: str,
    all_team_ids: list[int] | None = None,
    *,
    season_start: str | pd.Timestamp | None = None,
    season_end: str | pd.Timestamp | None = None,
    debug: bool = False,
) -> dict[int, int]:
    """
    End-of-season final rank 1-30 per user spec:
    - Rank 1: Champion (most playoff wins).
    - Rank 2: Finals runner-up (2nd-most playoff wins).
    - Ranks 3-15: Other playoff teams by playoff wins (desc), then reg-season win %.
    - Ranks 16-28: Lottery teams by reg-season win %.
    - Ranks 29-30: First 2 eliminated from playoffs (fewest playoff wins; tie-break: worse reg-season).
    Tie-break everywhere: playoff wins, then reg-season win %.

    NBA invariant: In best-of-7 (4 wins to advance), more playoff wins always means the team
    went further (first round exit max 3, second round max 7, Conf Finals max 11, Finals 12-15,
    champion 16). So a team cannot have more wins and have been eliminated earlier; ranking
    by wins alone ensures proper ordering. Tie-break for same win count: reg-season win %.
    """
    pw = get_playoff_wins(
        playoff_games,
        playoff_tgl,
        season,
        season_start=season_start,
        season_end=season_end,
    )
    reg_wp = get_reg_season_win_pct(
        games,
        tgl,
        season,
        season_start=season_start,
        season_end=season_end,
    )
    pt = _filtered_playoff_tgl(
        playoff_games,
        playoff_tgl,
        season,
        season_start=season_start,
        season_end=season_end,
        debug=debug,
    )
    playoff_team_ids = set(pt["team_id"].astype(int).tolist()) if not pt.empty else set()
    if all_team_ids is None:
        all_team_ids = sorted(set(list(pw.keys()) + list(reg_wp.keys())))
    if not all_team_ids:
        return {}
    if len(playoff_team_ids) < MIN_PLAYOFF_TEAMS:
        if debug:
            print(
                f"compute_eos_final_rank: Only {len(playoff_team_ids)} playoff teams (min {MIN_PLAYOFF_TEAMS})",
                flush=True,
            )
        return {}

    def _safe_wp(tid: int) -> float:
        v = reg_wp.get(tid, 0.0)
        return float(v) if pd.notna(v) else 0.0

    # Playoff teams sorted by wins desc (champion first). In NBA, more wins = went further,
    # so no team can have more wins and have been eliminated earlier. Tie-break: reg-season win %.
    playoff_teams = [tid for tid in all_team_ids if tid in playoff_team_ids]
    playoff_teams.sort(
        key=lambda t: (-pw.get(t, 0), -_playoff_round_rank(pw.get(t, 0)), -_safe_wp(t), t),
    )
    # First 2 eliminated: 2 playoff teams with fewest wins (tie-break: worse reg-season gets 30, then 29)
    first_two_eliminated = sorted(
        playoff_teams,
        key=lambda t: (pw.get(t, 0), _playoff_round_rank(pw.get(t, 0)), _safe_wp(t), t),
    )[:2]
    remaining_playoff = [t for t in playoff_teams if t not in first_two_eliminated]
    remaining_playoff.sort(
        key=lambda t: (-pw.get(t, 0), -_playoff_round_rank(pw.get(t, 0)), -_safe_wp(t), t),
    )
    lottery = [tid for tid in all_team_ids if tid not in playoff_team_ids]
    lottery.sort(key=lambda t: (-_safe_wp(t), t))

    out: dict[int, int] = {}
    r = 1
    for tid in remaining_playoff:
        out[tid] = r
        r += 1
    n_lottery_slots = 28 - len(remaining_playoff)
    for tid in lottery[:n_lottery_slots]:
        out[tid] = r
        r += 1
    # First 2 eliminated: worse reg-season gets 30, then 29
    fe_sorted = sorted(first_two_eliminated, key=lambda t: (_safe_wp(t), t))
    for i, tid in enumerate(fe_sorted):
        out[tid] = 30 - i
    return out


def compute_eos_playoff_standings(
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    season: str,
    *,
    season_start: str | pd.Timestamp | None = None,
    season_end: str | pd.Timestamp | None = None,
    all_team_ids: list[int] | None = None,
) -> dict[int, int]:
    """
    EOS playoff standings = final regular-season rank (1-30 by final reg-season win %).
    Used for EOS_playoff_standings in outputs. Returns {} if no games.
    """
    reg_wp = get_reg_season_win_pct(
        games,
        tgl,
        season,
        season_start=season_start,
        season_end=season_end,
    )
    if not reg_wp:
        return {}
    if all_team_ids is None:
        all_team_ids = sorted(reg_wp.keys())
    teams_sorted = sorted(
        all_team_ids,
        key=lambda t: (-float(reg_wp.get(t, 0.0) or 0.0), t),
    )
    return {tid: r for r, tid in enumerate(teams_sorted[:30], start=1)}


def get_playoff_finish_label(
    playoff_wins: dict[int, int],
    team_id: int,
) -> str:
    """Human-readable finish for reporting, e.g. 'NBA Finals Runner-Up (13 Playoff Wins)'."""
    w = playoff_wins.get(team_id, 0)
    if w == 0:
        return "Did not qualify"
    if w >= 16:
        return "Champion (16 Playoff Wins)"
    if w >= 12:
        return "Finals Runner-Up"
    if w >= 8:
        return "Conference Finals"
    if w >= 4:
        return "Second Round"
    return f"First Round ({w} Playoff Wins)"


def compute_playoff_contribution_per_player(
    playoff_games: pd.DataFrame,
    playoff_pgl: pd.DataFrame,
    season: str | None = None,
) -> pd.DataFrame:
    """
    Compute per-player playoff contribution (Game Score style) per game, aggregated by (player_id, team_id, season).
    Game Score = PTS + 0.5*REB + 1.5*AST + 2*STL + 2*BLK - TOV.
    Returns DataFrame with player_id, team_id, season, games_played, contribution_per_game, total_contribution.
    """
    if playoff_games.empty or playoff_pgl.empty:
        return pd.DataFrame()
    pg_season = playoff_games[["game_id", "season"]].drop_duplicates()
    ppgl = playoff_pgl.merge(pg_season, on="game_id", how="inner")
    if season is not None:
        ppgl = ppgl[ppgl["season"] == season]
    if ppgl.empty:
        return pd.DataFrame()
    # Game Score: PTS + 0.5*REB + 1.5*AST + 2*STL + 2*BLK - TOV
    for c in ("pts", "reb", "ast", "stl", "blk", "tov"):
        if c not in ppgl.columns:
            ppgl[c] = 0
    ppgl["_game_score"] = (
        ppgl["pts"].fillna(0)
        + 0.5 * ppgl["reb"].fillna(0)
        + 1.5 * ppgl["ast"].fillna(0)
        + 2.0 * ppgl["stl"].fillna(0)
        + 2.0 * ppgl["blk"].fillna(0)
        - ppgl["tov"].fillna(0)
    )
    agg = ppgl.groupby(["player_id", "team_id", "season"], as_index=False).agg(
        games_played=("game_id", "nunique"),
        total_contribution=("_game_score", "sum"),
    )
    agg["contribution_per_game"] = agg["total_contribution"] / agg["games_played"].clip(1)
    return agg


def load_playoff_rank_for_season(
    db_path: str | Path,
    season: str,
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    teams: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    """
    Load playoff data from DB and compute playoff rank, wins, and finish labels for one season.
    Returns (team_id -> playoff_rank, team_id -> playoff_wins, team_id -> finish_label).
    """
    from src.data.db_loader import load_playoff_data

    path = Path(db_path)
    if not path.exists():
        return {}, {}, {}
    pg, ptgl, _ = load_playoff_data(db_path)
    if pg.empty or ptgl.empty:
        return {}, {}, {}
    all_team_ids = sorted(teams["team_id"].astype(int).unique().tolist()) if not teams.empty else None
    rank = compute_playoff_performance_rank(pg, ptgl, games, tgl, season, all_team_ids=all_team_ids)
    wins = get_playoff_wins(pg, ptgl, season)
    labels = {tid: get_playoff_finish_label(wins, tid) for tid in rank}
    return rank, wins, labels
