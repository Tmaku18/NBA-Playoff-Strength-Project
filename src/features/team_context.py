"""Team-context features for Model B: Four Factors, pace, SOS/SRS, Elo, Massey, rolling, motivation, injury. FORBIDDEN: net_rating."""

from __future__ import annotations

from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

from .four_factors import FF_COUNT_COLS, four_factors_from_team_logs

FORBIDDEN = {"net_rating", "NET_RATING", "net rating"}


def season_start_for_date(as_of: str | pd.Timestamp | _date, seasons_cfg: dict | None = None) -> _date:
    """
    Return the regular-season start date for the season containing as_of.
    Uses config seasons ranges when available; otherwise falls back to an Aug 1
    boundary (NBA seasons start in the fall, so this excludes all prior seasons).
    """
    ad = pd.to_datetime(as_of).date() if not isinstance(as_of, _date) else as_of
    for _season, rng in (seasons_cfg or {}).items():
        if not isinstance(rng, dict):
            continue
        try:
            start = pd.to_datetime(rng.get("start")).date()
            end = pd.to_datetime(rng.get("end")).date()
        except (TypeError, ValueError):
            continue
        if start <= ad <= end:
            return start
    y = ad.year if ad.month >= 8 else ad.year - 1
    return _date(y, 8, 1)


def standing_rank_as_of_date(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    *,
    date_col: str = "game_date",
    team_id_col: str = "team_id",
    wl_col: str = "wl",
    season_start: str | pd.Timestamp | _date | None = None,
) -> dict[int, int]:
    """
    Compute regular-season standing rank (1-30, global) as of a date.
    Uses only games with season_start <= game_date < as_of_date so records are
    season-to-date, not all-franchise-history. Rank 1 = best (highest win rate).
    Tie-break: team_id ascending for stability. Teams with no games get rank 30.
    """
    ad = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    if tgl.empty or games.empty:
        return {}
    g = games[[c for c in ("game_id", date_col) if c in games.columns]].copy()
    if date_col not in g.columns:
        return {}
    ss = pd.to_datetime(season_start).date() if season_start is not None else season_start_for_date(ad)
    g[date_col] = pd.to_datetime(g[date_col]).dt.date
    g = g[(g[date_col] >= ss) & (g[date_col] < ad)]
    valid_game_ids = set(g["game_id"].tolist())
    past = tgl[tgl["game_id"].isin(valid_game_ids)].copy()
    if past.empty:
        return {}
    if wl_col not in past.columns:
        return {}
    past["w"] = (past[wl_col].astype(str).str.upper() == "W").astype(int)
    agg = past.groupby(team_id_col).agg({"w": "sum", "game_id": "nunique"}).rename(columns={"game_id": "g"})
    agg["win_rate"] = agg["w"] / agg["g"].replace(0, 1)
    agg = agg.sort_values(["win_rate", team_id_col], ascending=[False, True])
    rank_map = {int(tid): (r + 1) for r, tid in enumerate(agg.index)}
    return rank_map


def standing_rank_norm(rank: int) -> float:
    """Normalize standing rank 1-30 to [0,1] with 1 = best. (31 - rank) / 30."""
    r = max(1, min(30, int(rank)))
    return (31.0 - r) / 30.0


def _as_of_to_season(as_of: str | pd.Timestamp) -> str:
    """Derive season string from as_of_date (e.g. 2024-01-15 -> 2023-24, 2024-11-01 -> 2024-25)."""
    ad = pd.to_datetime(as_of).date()
    y, m = ad.year, ad.month
    if m >= 10:
        return f"{y}-{str((y + 1) % 100).zfill(2)}"
    return f"{y - 1}-{str(y % 100).zfill(2)}"

# Extended feature cols when optional modules enabled
EXTENDED_FEATURE_COLS: list[str] = [
    "elo", "eFG_L10", "DefRtg_L10", "won_prev_game",
    "days_until_playoffs", "elimination_status", "late_season", "eliminated_x_late_season",
    "proj_available_rating",
]


def build_team_context(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    *,
    sos_srs: pd.DataFrame | None = None,
    team_key: str = "team_id",
    season_key: str = "season",
) -> pd.DataFrame:
    """
    Build Model B feature set: Four Factors (eFG, TOV%, FT_rate, ORB%), pace, SOS, SRS.
    sos_srs: optional with team_abbreviation or team_id, season, sos, srs.
    Enforce: no net_rating in the output. FORBIDDEN is checked by leakage_tests.
    """
    ff = four_factors_from_team_logs(tgl, games)

    # pace: from games we need poss. Approx: 0.96 * (FGA + 0.44*FTA - ORB + TOV) per team; sum both teams per game / 2?
    # Simpler: use POSS from tgl if available; else approximate from tgl: FGA + 0.44*FTA - ORB + TOV (one team).
    if "fga" in tgl.columns and "fta" in tgl.columns and "oreb" in tgl.columns and "tov" in tgl.columns:
        tgl = tgl.copy()
        tgl["_poss"] = tgl["fga"].fillna(0) + 0.44 * tgl["fta"].fillna(0) - tgl["oreb"].fillna(0) + tgl["tov"].fillna(0)
        pace = tgl.groupby("game_id")["_poss"].sum().reset_index()
        pace = pace.rename(columns={"_poss": "pace"})
    else:
        pace = pd.DataFrame(columns=["game_id", "pace"])

    out = ff.merge(pace, on="game_id", how="left")

    if sos_srs is not None and not sos_srs.empty:
        # join on team+season. games has game_id, season; tgl has game_id, team_id. We need team->abbreviation from elsewhere or sos_srs has team_id.
        if "team_abbreviation" in sos_srs.columns and "team_id" not in sos_srs.columns:
            # would need teams table to map; for now skip if we don't have team_id in sos_srs
            pass
        elif "team_id" in sos_srs.columns and season_key in sos_srs.columns:
            gs = games[["game_id", "season"]].drop_duplicates()
            tgl_s = tgl[["game_id", "team_id"]].drop_duplicates().merge(gs, on="game_id")
            tgl_s = tgl_s.merge(sos_srs, left_on=["team_id", "season"], right_on=["team_id", season_key], how="left")
            out = out.merge(tgl_s[["game_id", "team_id", "sos", "srs"]], on=["game_id", "team_id"], how="left", suffixes=("", "_s"))

    for c in list(out.columns):
        if any(f in str(c).lower() for f in FORBIDDEN):
            raise ValueError(f"Model B must not include net_rating; found: {c}")

    return out


# Model B feature column names (no net_rating). standing_rank_norm = current regular-season rank as input (1=best).
TEAM_CONTEXT_FEATURE_COLS: list[str] = ["eFG", "TOV_pct", "FT_rate", "ORB_pct", "pace", "standing_rank_norm"]

# All feature cols (base + extended when enabled). Use for Model B when building feat_cols.
def get_team_context_feature_cols(config: dict | None = None) -> list[str]:
    """Return feature columns for Model B. Includes extended when config enables them.
    If model_b.include_features is set (non-null), returns intersection with full list (order by full list).
    If model_b.exclude_features is set, returns full list minus those names."""
    base = list(TEAM_CONTEXT_FEATURE_COLS)
    cfg = config or {}
    if cfg.get("elo", {}).get("enabled", False):
        base.append("elo")
    if cfg.get("massey", {}).get("enabled", False):
        base.append("massey_rating")
    if cfg.get("team_rolling", {}).get("enabled", True):
        base.extend(["eFG_L10", "DefRtg_L10", "won_prev_game"])
    if cfg.get("motivation", {}).get("enabled", False):
        base.extend(["days_until_playoffs", "elimination_status", "late_season", "eliminated_x_late_season"])
    if cfg.get("injury", {}).get("enabled", False):
        base.append("proj_available_rating")
    if cfg.get("sos_srs", {}).get("enabled", False):
        base.extend(["sos", "srs"])
    if cfg.get("raptor", {}).get("enabled", False):
        base.extend(["raptor_offense_sum_top5", "raptor_defense_sum_top5"])

    mb = cfg.get("model_b") or {}
    include = mb.get("include_features")
    exclude = mb.get("exclude_features") or []
    if include is not None:
        include_set = set(include)
        base = [c for c in base if c in include_set]
    if exclude:
        exclude_set = set(exclude)
        base = [c for c in base if c not in exclude_set]
    return base


def build_team_context_as_of_dates(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    team_dates: list[tuple[int, str]],
    *,
    date_col: str = "game_date",
    team_id_col: str = "team_id",
    teams: pd.DataFrame | None = None,
    pgl: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Build Model B features per (team_id, as_of_date): season-to-date mean of eFG, TOV_pct, FT_rate, ORB_pct, pace.
    Optionally merge Elo, team_rolling (eFG_L10, DefRtg_L10, won_prev_game), motivation, injury when config enables.
    """
    if not team_dates:
        return pd.DataFrame(columns=[team_id_col, "as_of_date"] + TEAM_CONTEXT_FEATURE_COLS)
    cfg = config or {}
    seasons_cfg = cfg.get("seasons") or {}
    # Standing rank (current regular-season rank as of date) for each unique as_of_date
    unique_dates = list(dict.fromkeys(as_of for _, as_of in team_dates))
    standing_by_date: dict[str, dict[int, int]] = {}
    for as_of in unique_dates:
        standing_by_date[str(as_of)] = standing_rank_as_of_date(
            tgl, games, as_of, date_col=date_col, team_id_col=team_id_col,
            season_start=season_start_for_date(as_of, seasons_cfg),
        )
    ctx = build_team_context(tgl, games)
    if "game_date" not in ctx.columns and "game_id" in games.columns and "game_date" in games.columns:
        ctx = ctx.merge(games[["game_id", "game_date"]].drop_duplicates(), on="game_id", how="left")
    ctx["_d64"] = pd.to_datetime(ctx[date_col])
    ctx = ctx.sort_values([team_id_col, "_d64"]).reset_index(drop=True)
    feat_cols = [c for c in TEAM_CONTEXT_FEATURE_COLS if c in ctx.columns]
    count_cols = [c for c in FF_COUNT_COLS if c in ctx.columns]
    has_counts = len(count_cols) == len(FF_COUNT_COLS)
    ctx_by_team: dict[int, pd.DataFrame] = {int(t): grp for t, grp in ctx.groupby(team_id_col)}
    rows = []
    for tid, as_of in team_dates:
        ad64 = np.datetime64(pd.to_datetime(as_of))
        ss64 = np.datetime64(pd.to_datetime(season_start_for_date(as_of, seasons_cfg)))
        grp = ctx_by_team.get(int(tid))
        row = {team_id_col: tid, "as_of_date": as_of, **{c: 0.0 for c in TEAM_CONTEXT_FEATURE_COLS}}
        if grp is not None and feat_cols:
            dts = grp["_d64"].to_numpy()
            lo = int(np.searchsorted(dts, ss64, side="left"))
            hi = int(np.searchsorted(dts, ad64, side="left"))
            past = grp.iloc[lo:hi]
            if not past.empty:
                if has_counts:
                    # Season rates as ratio-of-sums (not mean of per-game rates)
                    s = past[count_cols].sum()
                    fga = float(s["_ff_fga"])
                    tov_denom = fga + 0.44 * float(s["_ff_fta"]) + float(s["_ff_tov"])
                    orb_denom = float(s["_ff_oreb"]) + float(s["_ff_opp_dreb"])
                    if fga > 0:
                        row["eFG"] = (float(s["_ff_fgm"]) + 0.5 * float(s["_ff_fg3m"])) / fga
                        row["FT_rate"] = float(s["_ff_fta"]) / fga
                    if tov_denom > 0:
                        row["TOV_pct"] = float(s["_ff_tov"]) / tov_denom
                    if orb_denom > 0:
                        row["ORB_pct"] = float(s["_ff_oreb"]) / orb_denom
                    other = [c for c in feat_cols if c not in ("eFG", "TOV_pct", "FT_rate", "ORB_pct")]
                else:
                    other = feat_cols
                if other:
                    agg = past[other].mean()
                    for c in other:
                        row[c] = float(agg[c]) if pd.notna(agg[c]) else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in TEAM_CONTEXT_FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0.0
    # Fill standing_rank_norm from precomputed ranks (current regular-season standing as of date)
    def _norm_for_row(r: pd.Series) -> float:
        ranks = standing_by_date.get(str(r["as_of_date"]), {})
        rank = ranks.get(int(r[team_id_col]), 30)
        return standing_rank_norm(rank)
    out["standing_rank_norm"] = out.apply(_norm_for_row, axis=1)

    if cfg.get("elo", {}).get("enabled", False):
        from .elo import get_elo_as_of_dates
        elo_df = get_elo_as_of_dates(
            games, tgl, team_dates, seasons_cfg,
            cold_start_games=cfg.get("elo", {}).get("cold_start_games", 10),
            regression_to_mean=cfg.get("elo", {}).get("regression_to_mean", 0.25),
        )
        if not elo_df.empty:
            out = out.merge(elo_df, on=[team_id_col, "as_of_date"], how="left")
            out["elo"] = out["elo"].fillna(1500.0)

    if cfg.get("massey", {}).get("enabled", False):
        from .massey import get_massey_as_of_dates
        massey_df = get_massey_as_of_dates(games, tgl, team_dates, seasons_cfg)
        if not massey_df.empty:
            out = out.merge(massey_df, on=[team_id_col, "as_of_date"], how="left")
            out["massey_rating"] = out["massey_rating"].fillna(0.0)

    if cfg.get("team_rolling", {}).get("enabled", True):
        from .team_rolling import get_team_rolling_as_of_dates
        roll_df = get_team_rolling_as_of_dates(tgl, games, team_dates, window=10)
        if not roll_df.empty:
            out = out.merge(roll_df, on=[team_id_col, "as_of_date"], how="left")
            for c in ["eFG_L10", "DefRtg_L10", "won_prev_game"]:
                if c in out.columns:
                    out[c] = out[c].fillna(0.5 if c == "won_prev_game" else (0.5 if c == "eFG_L10" else 110.0))

    if cfg.get("motivation", {}).get("enabled", False):
        from .motivation import get_motivation_features
        mot_df = get_motivation_features(
            tgl, games, teams if teams is not None else pd.DataFrame(), team_dates, seasons_cfg,
            late_season_games=cfg.get("motivation", {}).get("late_season_games", 15),
            playoff_wins_threshold=cfg.get("motivation", {}).get("playoff_wins_threshold", 42),
        )
        if not mot_df.empty:
            extra = [c for c in mot_df.columns if c not in (team_id_col, "as_of_date")]
            if extra:
                mot_sub = mot_df[[team_id_col, "as_of_date"] + extra]
                out = out.merge(mot_sub, on=[team_id_col, "as_of_date"], how="left")
                for c in extra:
                    out[c] = out[c].fillna(0)

    if cfg.get("injury", {}).get("enabled", False) and pgl is not None:
        from ..data.injury_loader import load_injury_reports
        from .injury_adjustment import proj_available_rating_per_team
        paths_cfg = cfg.get("paths", {})
        base = Path(paths_cfg.get("raw", "data/raw"))
        injury_path = cfg.get("injury", {}).get("data_path", "data/raw/injury_reports")
        inj_path = Path(injury_path) if str(injury_path).startswith("/") or (len(str(injury_path)) > 1 and str(injury_path)[1] == ":") else base.parent / injury_path
        injury_df = load_injury_reports(inj_path)
        inj_df = proj_available_rating_per_team(
            pgl, tgl, games, team_dates, injury_df, seasons_cfg,
            minutes_heuristic=cfg.get("injury", {}).get("minutes_heuristic", "proportional"),
        )
        if not inj_df.empty:
            out = out.merge(inj_df, on=[team_id_col, "as_of_date"], how="left")
            out["proj_available_rating"] = out["proj_available_rating"].fillna(1.0)

    if cfg.get("raptor", {}).get("enabled", False) and pgl is not None:
        from datetime import timedelta
        from ..data.raptor_loader import load_raptor_by_player
        from .build_roster_set import get_roster_as_of_date, latest_team_map_as_of
        paths_cfg = cfg.get("paths", {})
        raw_base = Path(paths_cfg.get("raw", "data/raw")).resolve()
        project_root = raw_base.parent.parent if raw_base.name == "raw" else raw_base.parent
        raptor_path = cfg.get("raptor", {}).get("data_path", "docs/modern_RAPTOR_by_team.csv")
        if str(raptor_path).startswith("/") or (len(str(raptor_path)) > 1 and str(raptor_path)[1] == ":"):
            raptor_full = Path(raptor_path)
        else:
            raptor_full = project_root / raptor_path
        if not raptor_full.exists():
            raptor_full = project_root / "data/modern_RAPTOR_by_team.csv"
        if raptor_full.exists():
            from ..data.db import get_connection
            db_path = project_root / paths_cfg.get("db", "data/processed/nba_build_run.duckdb")
            if db_path.exists():
                con = get_connection(db_path, read_only=True)
                players_df = con.execute("SELECT player_id, player_name FROM players").df()
                con.close()
                raptor_df = load_raptor_by_player(raptor_full, players_df)
                if not raptor_df.empty:
                    raptor_df = raptor_df.set_index(["player_id", "season"])

                    # RAPTOR data ends in 2022-23; for later seasons carry the player's
                    # most recent rating forward with decay instead of silently using 0
                    # (all-zero test seasons vs populated train seasons = train/test skew).
                    def _prior_season_str(s: str) -> str:
                        y = int(str(s)[:4])
                        return f"{y - 1}-{str(y % 100).zfill(2)}"

                    cf_decay = float(cfg.get("raptor", {}).get("carry_forward_decay", 0.7))
                    cf_max_back = int(cfg.get("raptor", {}).get("carry_forward_seasons", 3))

                    def _raptor_lookup(pid: int, season: str) -> tuple[float, float] | None:
                        s = season
                        for gap in range(cf_max_back + 1):
                            if (pid, s) in raptor_df.index:
                                r = raptor_df.loc[(pid, s)]
                                w = cf_decay ** gap
                                return (
                                    w * float(r.get("raptor_offense", 0) or 0),
                                    w * float(r.get("raptor_defense", 0) or 0),
                                )
                            s = _prior_season_str(s)
                        return None

                    date_to_season_r: dict[str, str] = {}
                    for seas, bounds in (seasons_cfg or {}).items():
                        if isinstance(bounds, dict) and "start" in bounds and "end" in bounds:
                            start = pd.to_datetime(bounds["start"]).date()
                            end = pd.to_datetime(bounds["end"]).date()
                            d = start
                            while d <= end:
                                date_to_season_r[str(d)] = seas
                                d = d + timedelta(days=1)
                    def _season_for_d(d):
                        ds = str(pd.to_datetime(d).date()) if d is not None else ""
                        return date_to_season_r.get(str(d), date_to_season_r.get(ds, None))
                    raptor_rows = []
                    for tid, as_of in team_dates:
                        tid = int(tid)
                        ad = pd.to_datetime(as_of).date() if isinstance(as_of, str) else as_of
                        season_start_d = _season_for_d(ad)
                        ss = pd.to_datetime(seasons_cfg.get(season_start_d, {}).get("start", "")).date() if season_start_d else None
                        latest_team_map = latest_team_map_as_of(pgl, as_of, season_start=ss)
                        roster = get_roster_as_of_date(pgl, tid, as_of, season_start=ss, latest_team_map=latest_team_map)
                        top5 = roster.head(5)
                        off_sum = 0.0
                        def_sum = 0.0
                        for _, row in top5.iterrows():
                            pid = int(row["player_id"])
                            res = _raptor_lookup(pid, season_start_d) if season_start_d else None
                            if res is not None:
                                off_sum += res[0]
                                def_sum += res[1]
                        raptor_rows.append({team_id_col: tid, "as_of_date": as_of, "raptor_offense_sum_top5": off_sum, "raptor_defense_sum_top5": def_sum})
                    raptor_out = pd.DataFrame(raptor_rows)
                    out = out.merge(raptor_out, on=[team_id_col, "as_of_date"], how="left")
                    for c in ("raptor_offense_sum_top5", "raptor_defense_sum_top5"):
                        if c in out.columns:
                            out[c] = out[c].fillna(0.0)

    if cfg.get("sos_srs", {}).get("enabled", False):
        # Causal to-date SRS/SOS computed from games before each as_of date.
        # Replaces end-of-season Kaggle values that leaked full-season results.
        from .sos_srs import get_sos_srs_as_of_dates
        sos_df = get_sos_srs_as_of_dates(games, tgl, team_dates, seasons_cfg, team_id_col=team_id_col, date_col=date_col)
        if not sos_df.empty:
            out = out.merge(sos_df, on=[team_id_col, "as_of_date"], how="left")
            for c in ("sos", "srs"):
                if c in out.columns:
                    out[c] = out[c].fillna(0.0)

    for c in list(out.columns):
        if any(f in str(c).lower() for f in FORBIDDEN):
            raise ValueError(f"Model B must not include net_rating; found: {c}")

    return out
