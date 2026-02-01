"""Stub: load playoff player logs, compute per-player playoff vs regular-season stat delta.

Writes outputs2/playoff_adjustment_stub.json with placeholder structure.
No training yet. Expand per PlayoffPerformanceLearning.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import yaml

    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out_name = config.get("paths", {}).get("outputs", "outputs")
    out_dir = Path(out_name) if Path(out_name).is_absolute() else ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1 and 2 first.", file=sys.stderr)
        return 1

    from src.data.db_loader import load_playoff_data
    from src.evaluation.playoffs import compute_playoff_contribution_per_player

    try:
        pg, _, ppgl = load_playoff_data(db_path)
    except FileNotFoundError:
        print("Playoff data not loaded.", file=sys.stderr)
        return 1

    if pg.empty or ppgl.empty:
        stub = {"note": "No playoff data", "structure": "placeholder"}
    else:
        contrib = compute_playoff_contribution_per_player(pg, ppgl)
        stub = {
            "note": "Stub: playoff contribution computed; no training yet",
            "n_players": len(contrib) if not contrib.empty else 0,
            "sample": contrib.head(3).to_dict(orient="records") if not contrib.empty else [],
        }

    out_path = out_dir / "playoff_adjustment_stub.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stub, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
