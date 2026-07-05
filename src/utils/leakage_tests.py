"""Lightweight leakage tests: run before training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.rolling import compute_rolling_stats
from src.features.team_context import (
    FORBIDDEN,
    build_team_context,
    build_team_context_as_of_dates,
)


def test_no_future_leakage_in_features():
    """Assert no feature computation uses game_date >= as_of_date."""
    df = pd.DataFrame({
        "player_id": [1, 1, 1],
        "game_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "pts": [10, 12, 14],
        "reb": [3, 4, 5],
        "min": [30, 32, 28],
    })
    out = compute_rolling_stats(df, as_of_date="2024-01-02", stat_cols=["pts", "reb"], windows=[2])
    # Output must only include rows with game_date < 2024-01-02
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    bad = out[out["game_date"] >= pd.to_datetime("2024-01-02").date()]
    assert len(bad) == 0, "rolling must not include rows with game_date >= as_of_date"


def test_model_b_excludes_net_rating():
    """Assert Model B feature set and FORBIDDEN include no net_rating."""
    assert "net_rating" in FORBIDDEN or any("net_rating" in str(f).lower() for f in FORBIDDEN)

    # build_team_context must not produce net_rating
    games = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "game_date": ["2024-01-01", "2024-01-02"],
        "home_team_id": [1, 2],
        "away_team_id": [2, 1],
    })
    tgl = pd.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"],
        "team_id": [1, 2, 2, 1],
        "fgm": [40, 38, 42, 39],
        "fga": [85, 88, 86, 84],
        "fg3m": [10, 9, 11, 10],
        "ftm": [8, 12, 7, 11],
        "fta": [10, 14, 9, 13],
        "tov": [12, 10, 11, 13],
        "oreb": [8, 7, 9, 8],
        "dreb": [32, 34, 33, 31],
    })
    ctx = build_team_context(tgl, games)
    for c in ctx.columns:
        assert not any(f in str(c).lower() for f in FORBIDDEN), f"Model B must not include {c}"


def _two_season_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two games in 1985-86 with extreme stats, two games in 2023-24 with normal stats."""
    games = pd.DataFrame({
        "game_id": ["o1", "o2", "n1", "n2"],
        "game_date": ["1985-11-01", "1985-11-05", "2023-11-01", "2023-11-05"],
        "home_team_id": [1, 2, 1, 2],
        "away_team_id": [2, 1, 2, 1],
    })
    tgl = pd.DataFrame({
        "game_id": ["o1", "o1", "o2", "o2", "n1", "n1", "n2", "n2"],
        "team_id": [1, 2, 2, 1, 1, 2, 2, 1],
        "game_date": ["1985-11-01"] * 2 + ["1985-11-05"] * 2 + ["2023-11-01"] * 2 + ["2023-11-05"] * 2,
        # Old-era rows: fga missing (None) — the pattern that produced eFG ~= raw FGM
        "fgm": [45, 44, 46, 43, 40, 38, 42, 39],
        "fga": [None, None, None, None, 85, 88, 86, 84],
        "fg3m": [0, 0, 0, 0, 10, 9, 11, 10],
        "ftm": [20, 18, 19, 21, 8, 12, 7, 11],
        "fta": [25, 22, 24, 26, 10, 14, 9, 13],
        "tov": [15, 14, 16, 13, 12, 10, 11, 13],
        "oreb": [12, 11, 13, 10, 8, 7, 9, 8],
        "dreb": [30, 29, 31, 28, 32, 34, 33, 31],
        "pts": [110, 108, 112, 106, 100, 96, 104, 98],
        "wl": ["W", "L", "W", "L", "W", "L", "W", "L"],
    })
    return tgl, games


def test_four_factors_nan_when_fga_missing():
    """eFG/FT_rate must be NaN (not raw FGM) when FGA is missing or zero."""
    tgl, games = _two_season_fixture()
    ctx = build_team_context(tgl, games)
    old_rows = ctx[ctx["game_id"].isin(["o1", "o2"])]
    assert old_rows["eFG"].isna().all(), "eFG must be NaN when FGA missing, not raw FGM"
    assert old_rows["FT_rate"].isna().all(), "FT_rate must be NaN when FGA missing"
    new_rows = ctx[ctx["game_id"].isin(["n1", "n2"])]
    assert ((new_rows["eFG"] >= 0) & (new_rows["eFG"] <= 1.5)).all(), "eFG out of plausible bounds"


def test_team_context_is_season_scoped():
    """As-of-date features must only aggregate the current season, not franchise history."""
    tgl, games = _two_season_fixture()
    out = build_team_context_as_of_dates(tgl, games, [(1, "2023-11-10"), (2, "2023-11-10")])
    assert not out.empty
    # eFG for modern games of team 1: (40+0.5*10)/85 and (39+0.5*10)/84 -> ratio of sums
    expected = (40 + 39 + 0.5 * (10 + 10)) / (85 + 84)
    got = float(out[out["team_id"] == 1]["eFG"].iloc[0])
    assert abs(got - expected) < 1e-9, f"eFG {got} != season-scoped ratio-of-sums {expected}"
    assert (out["eFG"] <= 1.5).all(), "eFG contaminated by old-era rows with missing FGA"


def test_massey_is_causal():
    """Massey rating as of a date must ignore games on/after that date."""
    from src.features.massey import get_massey_as_of_dates
    games = pd.DataFrame({
        "game_id": ["g1", "g2"],
        "game_date": ["2024-01-01", "2024-02-01"],
        "season": ["2023-24", "2023-24"],
        "home_team_id": [1, 1],
        "away_team_id": [2, 2],
    })
    tgl = pd.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"],
        "team_id": [1, 2, 1, 2],
        "pts": [100, 90, 80, 120],  # g1: team1 +10; g2 (future): team1 -40
    })
    out = get_massey_as_of_dates(games, tgl, [(1, "2024-01-15")], seasons_cfg={})
    r = float(out["massey_rating"].iloc[0])
    assert r > 0, f"Massey as of 2024-01-15 must only see g1 (team1 +10); got {r}"


def test_build_lists_season_scoped():
    """Win rates in lists must be season-to-date, not all-history."""
    from src.training.build_lists import build_lists_for_conference_date
    tgl, games = _two_season_fixture()
    teams = pd.DataFrame({"team_id": [1, 2], "conference": ["E", "E"], "abbreviation": ["AAA", "BBB"]})
    lst = build_lists_for_conference_date(tgl, games, teams, "2023-11-10", "E")
    rates = dict(lst)
    # In 2023-24 to date: team 1 won both (W on n1 home... wl says team1 W in n1, L in n2 -> 0.5 each)
    assert rates[1] == 0.5 and rates[2] == 0.5, f"expected season-scoped 0.5/0.5, got {rates}"


def run_all():
    test_no_future_leakage_in_features()
    test_model_b_excludes_net_rating()
    test_four_factors_nan_when_fga_missing()
    test_team_context_is_season_scoped()
    test_massey_is_causal()
    test_build_lists_season_scoped()
