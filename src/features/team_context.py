"""Team-context features for Model B: Four Factors, pace, SOS/SRS, Elo, Massey, rolling, motivation, injury. FORBIDDEN: net_rating."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .four_factors import four_factors_from_team_logs

FORBIDDEN = {"net_rating", "NET_RATING", "net rating"}


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


# Model B feature column names (no net_rating)
TEAM_CONTEXT_FEATURE_COLS: list[str] = ["eFG", "TOV_pct", "FT_rate", "ORB_pct", "pace"]

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
    ctx = build_team_context(tgl, games)
    if "game_date" not in ctx.columns and "game_id" in games.columns and "game_date" in games.columns:
        ctx = ctx.merge(games[["game_id", "game_date"]].drop_duplicates(), on="game_id", how="left")
    ctx[date_col] = pd.to_datetime(ctx[date_col]).dt.date
    feat_cols = [c for c in TEAM_CONTEXT_FEATURE_COLS if c in ctx.columns]
    rows = []
    for tid, as_of in team_dates:
        ad = pd.to_datetime(as_of).date() if isinstance(as_of, str) else as_of
        past = ctx[(ctx[team_id_col] == tid) & (ctx[date_col] < ad)]
        if past.empty or not feat_cols:
            row = {team_id_col: tid, "as_of_date": as_of, **{c: 0.0 for c in TEAM_CONTEXT_FEATURE_COLS}}
        else:
            agg = past[feat_cols].mean()
            row = {team_id_col: tid, "as_of_date": as_of, **{c: 0.0 for c in TEAM_CONTEXT_FEATURE_COLS}}
            for c in feat_cols:
                row[c] = float(agg[c]) if pd.notna(agg[c]) else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in TEAM_CONTEXT_FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0.0

    cfg = config or {}
    seasons_cfg = cfg.get("seasons") or {}

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

    if cfg.get("sos_srs", {}).get("enabled", False) and teams is not None and not teams.empty:
        from ..data.kaggle_loader import load_team_records_srs
        project_root = Path(__file__).resolve().parents[2]
        srs_path = cfg.get("sos_srs", {}).get("data_path", "data/raw/Team_Records.csv")
        srs_full = Path(srs_path) if Path(srs_path).is_absolute() else project_root / srs_path
        sos_srs_df = load_team_records_srs(srs_full, teams)
        if not sos_srs_df.empty and "season" in sos_srs_df.columns:
            out["_season"] = out["as_of_date"].apply(_as_of_to_season)
            out = out.merge(sos_srs_df, left_on=[team_id_col, "_season"], right_on=["team_id", "season"], how="left", suffixes=("", "_srs"))
            out = out.drop(columns=["_season"], errors="ignore")
            for c in ["sos", "srs"]:
                if c in out.columns:
                    out[c] = out[c].fillna(0.0)

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
        from pathlib import Path
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
        from pathlib import Path
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
                            if season_start_d and (pid, season_start_d) in raptor_df.index:
                                r = raptor_df.loc[(pid, season_start_d)]
                                off_sum += float(r.get("raptor_offense", 0) or 0)
                                def_sum += float(r.get("raptor_defense", 0) or 0)
                        raptor_rows.append({team_id_col: tid, "as_of_date": as_of, "raptor_offense_sum_top5": off_sum, "raptor_defense_sum_top5": def_sum})
                    raptor_out = pd.DataFrame(raptor_rows)
                    out = out.merge(raptor_out, on=[team_id_col, "as_of_date"], how="left")
                    for c in ("raptor_offense_sum_top5", "raptor_defense_sum_top5"):
                        if c in out.columns:
                            out[c] = out[c].fillna(0.0)

    if cfg.get("sos_srs", {}).get("enabled", False):
        from datetime import timedelta
        from pathlib import Path
        from ..data.kaggle_loader import load_team_records_srs
        paths_cfg = cfg.get("paths", {})
        raw_base = Path(paths_cfg.get("raw", "data/raw")).resolve()
        project_root = raw_base.parent.parent if raw_base.name == "raw" else raw_base.parent
        sos_path = cfg.get("sos_srs", {}).get("data_path", "data/raw/Team_Records.csv")
        if str(sos_path).startswith("/") or (len(str(sos_path)) > 1 and str(sos_path)[1] == ":"):
            sos_full = Path(sos_path)
        else:
            sos_full = project_root / sos_path
        if sos_full.exists():
            teams_df = teams if teams is not None else pd.DataFrame()
            if not teams_df.empty:
                sos_df = load_team_records_srs(sos_full, teams_df)
                if not sos_df.empty and "team_id" in sos_df.columns and "season" in sos_df.columns:
                    date_to_season: dict[str, str] = {}
                    for seas, bounds in (seasons_cfg or {}).items():
                        if isinstance(bounds, dict) and "start" in bounds and "end" in bounds:
                            start = pd.to_datetime(bounds["start"]).date()
                            end = pd.to_datetime(bounds["end"]).date()
                            d = start
                            while d <= end:
                                date_to_season[str(d)] = seas
                                d = d + timedelta(days=1)
                    def _season_for_date(d):
                        ds = str(pd.to_datetime(d).date()) if d is not None else ""
                        return date_to_season.get(str(d), date_to_season.get(ds, None))
                    out["_season"] = out["as_of_date"].apply(_season_for_date)
                    sos_sub = sos_df[["team_id", "season", "srs", "sos"]].drop_duplicates(subset=["team_id", "season"])
                    out = out.merge(sos_sub, left_on=[team_id_col, "_season"], right_on=["team_id", "season"], how="left")
                    out = out.drop(columns=["_season", "season"], errors="ignore")
                    for c in ("sos", "srs"):
                        if c in out.columns:
                            out[c] = out[c].fillna(0.0)

    for c in list(out.columns):
        if any(f in str(c).lower() for f in FORBIDDEN):
            raise ValueError(f"Model B must not include net_rating; found: {c}")

    return out
