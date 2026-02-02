"""Download RAPTOR CSV from FiveThirtyEight GitHub. Saves to data/raw/raptor/ or config path."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

URL = "https://raw.githubusercontent.com/fivethirtyeight/data/master/nba-raptor/historical_RAPTOR_by_player.csv"


def main():
    import urllib.request

    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        import yaml
        config = yaml.safe_load(f)
    path_cfg = config.get("raptor", {}).get("data_path", "data/raw/raptor/historical_RAPTOR_by_player.csv")
    out_path = ROOT / path_cfg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(URL, out_path)
        print(f"Downloaded RAPTOR to {out_path}")
        return 0
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
