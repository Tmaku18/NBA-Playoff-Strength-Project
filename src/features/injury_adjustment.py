"""Injury adjustments: Projected Available Rating. Downgrade team when star is out."""
from __future__ import annotations

import pandas as pd

from src.features.on_off import get_on_court_pm_as_of_date
from src.features.build_roster_set import get_roster_as_of_date, latest_team_map_as_of
from src.data.injury_loader import load_injury_reports, get_injured_players_as_of


TOTAL_TEAM_MINUTES = 48.0 * 5.0  # 240


def proj_available_rating_per_team(
    pgl: pd.DataFrame,
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    team_dates: list[tuple[int, str]],
    injury_df: pd.DataFrame,
    seasons_cfg: dict,
    *,
    minutes_heuristic: str = "proportional",
    season_start: str | None = None,
) -> pd.DataFrame:
    """
    Compute proj_available_rating per (team_id, as_of_date).
    Formula: ProjRating = sum(metric_p * proj_minutes_p for p in Active) / 240.
    When injury data is empty, returns 1.0 (full strength) or uses full roster.
    """
    if not team_dates:
        return pd.DataFrame(columns=["team_id", "as_of_date", "proj_available_rating"])

    injured_by_team: dict[tuple[int, str], dict[int, set[int]]] = {}
    if not injury_df.empty:
        for tid, as_of in team_dates:
            by_team = get_injured_players_as_of(injury_df, as_of)
            injured_by_team[(int(tid), as_of)] = by_team

    on_court_pm_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for tid, as_of in team_dates:
        tid = int(tid)
        ad = pd.to_datetime(as_of).date() if isinstance(as_of, str) else as_of
        season_start_d = _season_for_date(ad, seasons_cfg)
        ss = pd.to_datetime(seasons_cfg.get(season_start_d, {}).get("start", "")).date() if season_start_d else None

        latest_team_map = latest_team_map_as_of(pgl, as_of, season_start=ss)
        roster = get_roster_as_of_date(
            pgl, tid, as_of,
            season_start=ss,
            latest_team_map=latest_team_map,
        )
        if roster.empty:
            rows.append({"team_id": tid, "as_of_date": as_of, "proj_available_rating": 1.0})
            continue

        injured = injured_by_team.get((tid, as_of), {}).get(tid, set())
        active_roster = roster[~roster["player_id"].isin(injured)]

        if active_roster.empty:
            rows.append({"team_id": tid, "as_of_date": as_of, "proj_available_rating": 0.5})
            continue

        if as_of not in on_court_pm_cache:
            on_court_pm_cache[as_of] = get_on_court_pm_as_of_date(pgl, tgl, games, as_of)
        on_court_pm = on_court_pm_cache[as_of]
        metric_col = "on_court_pm_approx_L10"
        if metric_col not in on_court_pm.columns:
            metric_col = "on_court_pm_approx_L30" if "on_court_pm_approx_L30" in on_court_pm.columns else None

        total_min = roster["total_min"].sum()
        if total_min <= 0:
            total_min = 1.0
        injured_min = roster[roster["player_id"].isin(injured)]["total_min"].sum()
        active_min_pool = total_min - injured_min
        if active_min_pool <= 0:
            active_min_pool = total_min

        proj_minutes = _project_minutes(
            roster, active_roster, injured_min,
            minutes_heuristic=minutes_heuristic,
            total_min=total_min,
        )

        player_to_metric = {}
        if metric_col and not on_court_pm.empty:
            pm_df = on_court_pm.set_index("player_id")
            for pid in proj_minutes:
                if pid in pm_df.index:
                    player_to_metric[pid] = float(pm_df.loc[pid, metric_col])
                else:
                    player_to_metric[pid] = 0.0

        weighted = sum(
            player_to_metric.get(pid, 0.0) * proj_minutes.get(pid, 0.0)
            for pid in proj_minutes
        )
        raw_rating = weighted / TOTAL_TEAM_MINUTES
        full_strength = sum(
            player_to_metric.get(pid, 0.0) * (roster.loc[roster["player_id"] == pid, "total_min"].iloc[0] if len(roster[roster["player_id"] == pid]) else 0)
            for pid in roster["player_id"]
        ) / max(total_min, 1) * (TOTAL_TEAM_MINUTES / TOTAL_TEAM_MINUTES)
        full_weighted = sum(
            player_to_metric.get(pid, 0.0) * roster.loc[roster["player_id"] == pid, "total_min"].values[0]
            for pid in roster["player_id"]
            if len(roster[roster["player_id"] == pid]) and pid in player_to_metric
        )
        full_rating = full_weighted / max(total_min, 1) * (TOTAL_TEAM_MINUTES / total_min) if total_min else 0
        scale = full_rating if full_rating != 0 else 1.0
        proj_available = raw_rating / scale if scale else 1.0
        proj_available = max(0.0, min(2.0, proj_available))

        rows.append({"team_id": tid, "as_of_date": as_of, "proj_available_rating": float(proj_available)})

    return pd.DataFrame(rows)


def _project_minutes(
    roster: pd.DataFrame,
    active_roster: pd.DataFrame,
    injured_minutes: float,
    *,
    total_min: float = 1.0,
) -> dict[int, float]:
    """Project minutes for active players. Proportional: redistribute 240 by minute share."""
    active_min = active_roster["total_min"].sum()
    if active_min <= 0:
        n = max(len(active_roster), 1)
        return {int(pid): TOTAL_TEAM_MINUTES / n for pid in active_roster["player_id"]}
    share = active_roster.set_index("player_id")["total_min"] / active_min
    return {int(pid): TOTAL_TEAM_MINUTES * float(share.loc[pid]) for pid in active_roster["player_id"]}


def _season_for_date(d: object, seasons_cfg: dict) -> str | None:
    d = pd.to_datetime(d).date() if d is not None else None
    if d is None or not seasons_cfg:
        return None
    for season, rng in seasons_cfg.items():
        start = pd.to_datetime(rng.get("start")).date()
        end = pd.to_datetime(rng.get("end")).date()
        if start <= d <= end:
            return season
    return None
