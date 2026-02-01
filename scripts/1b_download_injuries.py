"""Optional: Download injury reports. Stub when no API available.
Place JSON files in data/raw/injury_reports/ with schema:
{ "date": "YYYY-MM-DD", "injuries": [ { "player_id", "team_id", "status": "Out"|"Day-To-Day"|... } ] }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
    injury_path = config.get("injury", {}).get("data_path", "data/raw/injury_reports")
    path = ROOT / injury_path
    path.mkdir(parents=True, exist_ok=True)
    # Stub: no public NBA injury API in nba_api. Document manual process.
    readme = path / "README.txt"
    readme.write_text(
        "Injury reports: place JSON files here.\n"
        "Schema: { \"date\": \"YYYY-MM-DD\", \"injuries\": [ { \"player_id\": int, \"team_id\": int, \"status\": \"Out\" } ] }\n"
        "Sources: Sportradar (API key), ESPN, or manual scraping."
    )
    print(f"Created {readme}. Add injury JSON files manually.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
