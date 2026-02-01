"""Rolling features with strict t-1: shift(1) before rolling. L10/L30, DNP, availability fraction."""
from __future__ import annotations

import pandas as pd


def compute_rolling_stats(
    df: pd.DataFrame,
    *,
    player_id_col: str = "player_id",
    date_col: str = "game_date",
    windows: list[int] | None = None,
    stat_cols: list[str] | None = None,
    as_of_date: str | None = None,
    min_col: str = "min",
) -> pd.DataFrame:
    """
    Compute rolling per-game averages over L10/L30 using only past data (t-1).
    Apply shift(1) before rolling so the value for row i uses rows 0..i-1.
    DNP: compute over games played; add availability = fraction of games played in window.
    If as_of_date is set, filter to rows with date_col < as_of_date before computing.
    """
    if windows is None:
        windows = [10, 30]
    if stat_cols is None:
        stat_cols = ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]
    # normalize column names to lower for lookups
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if as_of_date is not None:
        ad = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
        df = df[df[date_col].dt.date < ad].copy()

    out_cols = [player_id_col, date_col]
    by_player = df.sort_values([player_id_col, date_col]).groupby(player_id_col, sort=False)

    for w in windows:
        for c in stat_cols:
            if c not in df.columns:
                continue
            # shifted then rolling: value at row i = mean of rows i-w..i-1
            roll = by_player[c].transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
            df[f"{c}_L{w}"] = roll
            if f"{c}_L{w}" not in out_cols:
                out_cols.append(f"{c}_L{w}")

        # availability: fraction of games in window with min > 0 (or not null)
        if min_col in df.columns:
            played = (df[min_col].fillna(0) > 0).astype(float)
            avail = by_player[played.name].transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
            df[f"availability_L{w}"] = avail
            if f"availability_L{w}" not in out_cols:
                out_cols.append(f"availability_L{w}")

    return df


# Default 7 stat cols for DeepSetRank stat_dim=7 (build_roster_set legacy)
PLAYER_STAT_COLS_L10: list[str] = [
    "pts_L10", "reb_L10", "ast_L10", "stl_L10", "blk_L10", "tov_L10", "availability_L10",
]

# L30 stat cols (7) for second rolling window
PLAYER_STAT_COLS_L30: list[str] = [
    "pts_L30", "reb_L30", "ast_L30", "stl_L30", "blk_L30", "tov_L30", "availability_L30",
]

# L10 + L30 (14 total) for stat_dim=14
PLAYER_STAT_COLS_L10_L30: list[str] = PLAYER_STAT_COLS_L10 + PLAYER_STAT_COLS_L30

# On-court +/- approximation (from on_off.py), L10 and L30 windows
ON_OFF_STAT_COLS: list[str] = ["on_court_pm_approx_L10", "on_court_pm_approx_L30"]

# Full stat cols for Model A: 14 base + 2 on_off = 16. pct_min_returning (team scalar) added in build_roster_set -> 17
PLAYER_STAT_COLS_WITH_ON_OFF: list[str] = PLAYER_STAT_COLS_L10_L30 + ON_OFF_STAT_COLS


def get_prior_season_stats(
    pgl: pd.DataFrame,
    season_start: str | pd.Timestamp,
    *,
    player_id_col: str = "player_id",
    date_col: str = "game_date",
    stat_cols: list[str] | None = None,
    lookback_days: int = 365,
    min_col: str = "min",
) -> pd.DataFrame:
    """
    Compute prior season averages for all players.
    
    For a given season_start, compute season-long averages from games in the prior year
    (season_start - lookback_days, season_start). Returns DataFrame with player_id and
    stat columns (pts_L10, reb_L10, etc.) containing the prior season averages.
    
    This is used as a baseline for early-season predictions when players have 0 current-season stats.
    """
    if stat_cols is None:
        stat_cols = PLAYER_STAT_COLS_L10
    
    # Map L10/L30 stat names to base stat names
    base_stat_map = {
        "pts_L10": "pts", "reb_L10": "reb", "ast_L10": "ast",
        "stl_L10": "stl", "blk_L10": "blk", "tov_L10": "tov",
        "pts_L30": "pts", "reb_L30": "reb", "ast_L30": "ast",
        "stl_L30": "stl", "blk_L30": "blk", "tov_L30": "tov",
    }
    
    pgl = pgl.copy()
    pgl[date_col] = pd.to_datetime(pgl[date_col])
    ss = pd.to_datetime(season_start)
    start_date = ss - pd.Timedelta(days=lookback_days)
    
    # Filter to prior season games
    mask = (pgl[date_col] >= start_date) & (pgl[date_col] < ss)
    prior = pgl.loc[mask]
    
    if prior.empty:
        return pd.DataFrame(columns=[player_id_col] + stat_cols)
    
    # Compute per-player averages for base stats
    agg_cols = {}
    for col in stat_cols:
        base = base_stat_map.get(col)
        if base and base in prior.columns:
            agg_cols[col] = (base, "mean")
    
    # Handle availability separately (L10 and L30)
    for avail_col in ("availability_L10", "availability_L30"):
        if avail_col in stat_cols and min_col in prior.columns:
            prior = prior.copy()
            if "_played" not in prior.columns:
                prior["_played"] = (prior[min_col].fillna(0) > 0).astype(float)
            agg_cols[avail_col] = ("_played", "mean")
    
    if not agg_cols:
        return pd.DataFrame(columns=[player_id_col] + stat_cols)
    
    out = prior.groupby(player_id_col, as_index=False).agg(**agg_cols)
    
    # Ensure all stat_cols are present
    for c in stat_cols:
        if c not in out.columns:
            out[c] = 0.0
    
    return out[[player_id_col] + stat_cols]


def get_player_stats_as_of_date(
    pgl: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    *,
    player_id_col: str = "player_id",
    date_col: str = "game_date",
    windows: list[int] | None = None,
    stat_cols: list[str] | None = None,
    prior_season_stats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Rolling stats as-of a date: one row per player with L10 (or first window) stats.
    Uses compute_rolling_stats then keeps the latest row per player (game_date < as_of_date).
    stat_cols default is PLAYER_STAT_COLS_L10_L30 (14 stats) when stat_dim=14.
    
    If prior_season_stats is provided, players with all-zero stats will have their stats
    filled from the prior season averages (used as baseline for early-season predictions).
    """
    if windows is None:
        windows = [10]
    rolled = compute_rolling_stats(
        pgl,
        player_id_col=player_id_col,
        date_col=date_col,
        as_of_date=as_of_date,
        windows=windows,
    )
    cols = stat_cols or PLAYER_STAT_COLS_L10_L30
    
    if rolled.empty:
        # No current season data - use prior season stats if available
        if prior_season_stats is not None and not prior_season_stats.empty:
            out = prior_season_stats[[player_id_col] + [c for c in cols if c in prior_season_stats.columns]].copy()
            for c in cols:
                if c not in out.columns:
                    out[c] = 0.0
            return out[[player_id_col] + cols]
        return pd.DataFrame(columns=[player_id_col] + cols)
    
    rolled = rolled.sort_values([player_id_col, date_col])
    # Latest row per player (last game before as_of_date)
    out = rolled.groupby(player_id_col, as_index=False).last()
    out = out[[player_id_col] + [c for c in cols if c in out.columns]].copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0.0
    out = out[[player_id_col] + cols]
    
    # Fill zeros with prior season stats if available
    if prior_season_stats is not None and not prior_season_stats.empty:
        # Identify players with all-zero stats (no current season data)
        numeric_cols = [c for c in cols if c != player_id_col]
        all_zero_mask = (out[numeric_cols] == 0).all(axis=1)
        
        if all_zero_mask.any():
            # Merge with prior season stats
            prior_indexed = prior_season_stats.set_index(player_id_col)
            for idx in out[all_zero_mask].index:
                pid = out.loc[idx, player_id_col]
                if pid in prior_indexed.index:
                    for c in numeric_cols:
                        if c in prior_indexed.columns:
                            prior_val = prior_indexed.loc[pid, c]
                            if pd.notna(prior_val) and prior_val != 0:
                                out.loc[idx, c] = prior_val
    
    return out
