"""Train Model A (DeepSet + ListMLE)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.training.train_model_a import train_model_a


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config["paths"]["outputs"])
    path = train_model_a(config, out)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
