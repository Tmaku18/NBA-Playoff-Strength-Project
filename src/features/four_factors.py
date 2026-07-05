"""Four Factors: eFG%, TOV%, ORB% (Opp DRB from other row in same game), FT rate. No net_rating."""

from __future__ import annotations

import pandas as pd

# Raw counting columns emitted alongside per-game rates so callers can compute
# season aggregates as ratio-of-sums instead of mean-of-per-game-rates.
FF_COUNT_COLS: list[str] = [
    "_ff_fgm", "_ff_fga", "_ff_fg3m", "_ff_fta", "_ff_tov", "_ff_oreb", "_ff_opp_dreb",
]


def four_factors_from_team_logs(tgl: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    tgl: team_game_logs with game_id, team_id, fgm, fga, fg3m, ftm, fta, tov, oreb, pts.
    games: game_id, home_team_id, away_team_id.
    For ORB% we need Opp_DRB. Infer from the other team's row in the same game (DREB).
    eFG% = (FGM + 0.5*FG3M)/FGA; TOV% = TOV/(FGA+0.44*FTA+TOV); FT_rate = FTA/FGA;
    ORB% = OREB/(OREB+Opp_DRB).
    Rates are NaN when their denominator is missing/zero (early-era logs often lack FGA;
    substituting a placeholder denominator previously made eFG equal raw FGM).
    Also returns FF_COUNT_COLS raw counts for ratio-of-sums season aggregation.
    """
    tgl = tgl.copy()
    for c in ("fgm", "fga", "fg3m", "fta", "tov", "oreb", "dreb"):
        if c in tgl.columns:
            tgl[c] = pd.to_numeric(tgl[c], errors="coerce")
    fga_valid = tgl["fga"].where(tgl["fga"] > 0)
    tgl["eFG"] = (tgl["fgm"].fillna(0) + 0.5 * tgl["fg3m"].fillna(0)) / fga_valid
    tov_denom = tgl["fga"].fillna(0) + 0.44 * tgl["fta"].fillna(0) + tgl["tov"].fillna(0)
    tgl["TOV_pct"] = tgl["tov"].fillna(0) / tov_denom.where(tov_denom > 0)
    tgl["FT_rate"] = tgl["fta"].fillna(0) / fga_valid

    # Opp DRB: for each (game_id, team_id) the opponent is the other team; get their dreb
    g = games[["game_id", "home_team_id", "away_team_id"]].drop_duplicates()
    tgl = tgl.merge(g, on="game_id", how="left")
    tgl["opp_team_id"] = tgl.apply(
        lambda r: r["away_team_id"] if r["team_id"] == r["home_team_id"] else r["home_team_id"],
        axis=1,
    )
    opp = tgl[["game_id", "team_id", "dreb"]].rename(columns={"team_id": "opp_team_id", "dreb": "opp_dreb"})
    tgl = tgl.merge(opp, on=["game_id", "opp_team_id"], how="left")
    orb_denom = tgl["oreb"].fillna(0) + tgl["opp_dreb"].fillna(0)
    tgl["ORB_pct"] = tgl["oreb"].fillna(0) / orb_denom.where(orb_denom > 0)

    tgl["_ff_fgm"] = tgl["fgm"].fillna(0.0)
    tgl["_ff_fga"] = tgl["fga"].fillna(0.0)
    tgl["_ff_fg3m"] = tgl["fg3m"].fillna(0.0)
    tgl["_ff_fta"] = tgl["fta"].fillna(0.0)
    tgl["_ff_tov"] = tgl["tov"].fillna(0.0)
    tgl["_ff_oreb"] = tgl["oreb"].fillna(0.0)
    tgl["_ff_opp_dreb"] = tgl["opp_dreb"].fillna(0.0)

    return tgl[["game_id", "team_id", "eFG", "TOV_pct", "FT_rate", "ORB_pct"] + FF_COUNT_COLS].copy()
