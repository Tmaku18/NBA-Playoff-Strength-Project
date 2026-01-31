"""Train Model A (DeepSet + ListMLE)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.data.db_loader import load_training_data
from src.training.model_a_data import build_model_a_batches
from src.training.train_model_a import train_model_a


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        sys.exit(1)
    games, tgl, teams, pgl = load_training_data(db_path)
    train_batches, val_batches = build_model_a_batches(config, games, tgl, teams, pgl)
    out = Path(config["paths"]["outputs"])
    path = train_model_a(config, out, train_batches=train_batches, val_batches=val_batches)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
