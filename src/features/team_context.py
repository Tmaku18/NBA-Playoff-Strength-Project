"""Team-context features for Model B: Four Factors, pace, SOS/SRS. FORBIDDEN: net_rating."""

from __future__ import annotations

import pandas as pd

from .four_factors import four_factors_from_team_logs

FORBIDDEN = {"net_rating", "NET_RATING", "net rating"}
TEAM_CONTEXT_FEATURE_COLS = ["eFG", "TOV_pct", "FT_rate", "ORB_pct", "pace", "sos", "srs"]


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


def build_team_context_as_of_dates(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    team_dates: list[tuple[int, str]],
) -> pd.DataFrame:
    """Build team-context features aggregated up to each as_of_date."""
    if not team_dates:
        return pd.DataFrame(columns=["team_id", "as_of_date"] + TEAM_CONTEXT_FEATURE_COLS)

    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    t = tgl.copy()
    t["game_date"] = pd.to_datetime(t["game_date"])

    out_rows: list[pd.DataFrame] = []
    for as_of in sorted({d for _, d in team_dates}):
        ad = pd.to_datetime(as_of)
        g_sub = g[g["game_date"] < ad]
        t_sub = t[t["game_date"] < ad]
        feats = build_team_context(t_sub, g_sub)
        if feats.empty:
            continue
        agg = feats.groupby("team_id", as_index=False).mean(numeric_only=True)
        agg["as_of_date"] = str(pd.to_datetime(as_of).date())
        out_rows.append(agg)

    if not out_rows:
        return pd.DataFrame(columns=["team_id", "as_of_date"] + TEAM_CONTEXT_FEATURE_COLS)

    df = pd.concat(out_rows, ignore_index=True)
    # ensure all expected columns exist
    for c in TEAM_CONTEXT_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0

    keep = ["team_id", "as_of_date"] + TEAM_CONTEXT_FEATURE_COLS
    return df[keep].copy()
