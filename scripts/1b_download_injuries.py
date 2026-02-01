"""Download injury reports via nbainjuries. Maps Team/Player to IDs, writes JSON to data/raw/injury_reports/.
Requires: nbainjuries, Java 8+ (for tabula-py). Data available since 2021-22 only."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _last_first_to_first_last(name: str) -> str:
    """'Last, First' or 'Last, Jr., First' -> 'First Last' or 'First Last Jr.'."""
    s = str(name).strip()
    if "," not in s:
        return s
    parts = [p.strip() for p in s.split(",", 1)]
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return s


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
    injury_path = config.get("injury", {}).get("data_path", "data/raw/injury_reports")
    out_dir = ROOT / injury_path
    out_dir.mkdir(parents=True, exist_ok=True)

    source = config.get("injury", {}).get("source", "nbainjuries")
    if source != "nbainjuries":
        readme = out_dir / "README.txt"
        readme.write_text(
            "Injury reports: place JSON files here (source != nbainjuries).\n"
            "Schema: { \"date\": \"YYYY-MM-DD\", \"injuries\": [ { \"player_id\": int, \"team_id\": int, \"status\": \"Out\" } ] }\n"
        )
        print(f"source={source}: add injury JSON files manually.")
        return 0

    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        return 1

    from src.data.db import get_connection
    con = get_connection(db_path, read_only=True)
    teams_df = con.execute("SELECT team_id, name, abbreviation FROM teams").df()
    players_df = con.execute("SELECT player_id, player_name FROM players").df()
    con.close()

    team_to_id: dict[str, int] = {}
    for _, r in teams_df.iterrows():
        n = str(r.get("name", "")).strip()
        a = str(r.get("abbreviation", "")).strip()
        if n:
            team_to_id[n] = int(r["team_id"])
        if a:
            team_to_id[a] = int(r["team_id"])

    player_to_id: dict[str, int] = {}
    for _, r in players_df.iterrows():
        nm = str(r.get("player_name", "")).strip()
        if nm:
            player_to_id[nm] = int(r["player_id"])
    first_last_map = {_last_first_to_first_last(k): v for k, v in player_to_id.items()}
    for k, v in player_to_id.items():
        first_last_map[k] = v

    seasons_cfg = config.get("seasons", {})
    if not seasons_cfg:
        print("No seasons in config.", file=sys.stderr)
        return 1

    status_filter = {"Out", "OUT"}
    try:
        from nbainjuries import injury
    except ImportError:
        print("nbainjuries not installed. Run: pip install nbainjuries", file=sys.stderr)
        return 1

    # nbainjuries data available since 2021-22 only
    INJURY_DATA_START = "2021-22"
    dates_to_fetch: list[tuple[str, datetime]] = []
    for seas, bounds in seasons_cfg.items():
        if str(seas) < INJURY_DATA_START:
            continue
        if not isinstance(bounds, dict) or "start" not in bounds or "end" not in bounds:
            continue
        start = datetime.strptime(bounds["start"], "%Y-%m-%d")
        end = datetime.strptime(bounds["end"], "%Y-%m-%d")
        d = start
        while d <= end:
            dates_to_fetch.append((d.strftime("%Y-%m-%d"), d))
            d = d + timedelta(days=1)

    print(f"Fetching {len(dates_to_fetch)} dates (nbainjuries, 5pm ET)...")
    written = 0
    errors = 0
    for date_str, dt in dates_to_fetch:
        try:
            df = injury.get_reportdata(
                datetime(dt.year, dt.month, dt.day, 17, 30),
                return_df=True,
            )
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  {date_str}: {e}", file=sys.stderr)
            continue
        if df is None or df.empty:
            continue
        team_col = "Team" if "Team" in df.columns else "team"
        player_col = "Player Name" if "Player Name" in df.columns else "player_name"
        status_col = "Current Status" if "Current Status" in df.columns else "status"
        if team_col not in df.columns or player_col not in df.columns:
            continue
        injuries = []
        for _, row in df.iterrows():
            status = str(row.get(status_col, "")).strip()
            if status not in status_filter:
                continue
            team_name = str(row.get(team_col, "")).strip()
            player_name_raw = str(row.get(player_col, "")).strip()
            tid = team_to_id.get(team_name)
            pid = player_to_id.get(player_name_raw) or first_last_map.get(_last_first_to_first_last(player_name_raw))
            if tid is not None and pid is not None:
                injuries.append({"player_id": int(pid), "team_id": int(tid), "status": status})
        if injuries:
            out_file = out_dir / f"{date_str}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump({"date": date_str, "injuries": injuries}, f, indent=0)
            written += 1
    print(f"Wrote {written} JSON files to {out_dir}.")
    if errors:
        print(f"({errors} dates had errors)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
