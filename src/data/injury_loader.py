"""Load injury reports from JSON. Expected schema: date, injuries list with player_id, team_id, status."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_injury_reports(data_path: str | Path) -> pd.DataFrame:
    """
    Load injury reports from directory or single JSON.
    Expected file schema: { "date": "YYYY-MM-DD", "injuries": [ { "player_id", "team_id", "status": "Out"|"Day-To-Day"|... } ] }
    Or directory of JSON files, each with that schema.
    Returns DataFrame with columns: date, player_id, team_id, status.
    """
    path = Path(data_path)
    if not path.exists():
        return pd.DataFrame(columns=["date", "player_id", "team_id", "status"])

    rows: list[dict[str, Any]] = []
    if path.is_file() and path.suffix.lower() in (".json",):
        rows.extend(_parse_injury_file(path))
    elif path.is_dir():
        for f in path.glob("**/*.json"):
            rows.extend(_parse_injury_file(f))

    if not rows:
        return pd.DataFrame(columns=["date", "player_id", "team_id", "status"])
    return pd.DataFrame(rows)


def _parse_injury_file(p: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return out
    if not isinstance(data, dict):
        return out
    date_str = data.get("date") or data.get("game_date")
    if not date_str:
        return out
    injuries = data.get("injuries") or data.get("players") or []
    if not isinstance(injuries, list):
        return out
    for item in injuries:
        if not isinstance(item, dict):
            continue
        pid = item.get("player_id") or item.get("PLAYER_ID")
        tid = item.get("team_id") or item.get("TEAM_ID")
        status = item.get("status") or item.get("STATUS") or "Out"
        if pid is not None and tid is not None:
            out.append({
                "date": str(date_str),
                "player_id": int(pid),
                "team_id": int(tid),
                "status": str(status),
            })
    return out


def get_injured_players_as_of(
    injury_df: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
) -> dict[int, set[int]]:
    """
    Return team_id -> set of player_ids who are injured (status Out) as of date.
    Uses most recent report before or on as_of_date per (team_id, player_id).
    """
    if injury_df.empty:
        return {}
    df = injury_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    ad = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    past = df[df["date"] <= ad]
    if past.empty:
        return {}
    past = past.sort_values("date")
    out_status = past.groupby(["team_id", "player_id"], as_index=False).last()
    out_players = out_status[
        out_status["status"].astype(str).str.upper().str.contains("OUT", na=False)
    ]
    result: dict[int, set[int]] = {}
    for _, r in out_players.iterrows():
        tid = int(r["team_id"])
        pid = int(r["player_id"])
        if tid not in result:
            result[tid] = set()
        result[tid].add(pid)
    return result
