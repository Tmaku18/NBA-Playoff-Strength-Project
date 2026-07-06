"""Script 6: Run inference and write predictions.

What this does:
- Loads trained Model A (best_deep_set.pt), Model B (xgb/rf), and stacking meta.
- Runs inference on test dates to produce team strength scores.
- Writes predictions.json (and per-season predictions_YYYY.json) to outputs/run_NNN/.
- Uses run_id from .current_run (set by script 3) so one pipeline = one run folder.

Run after scripts 3, 4, 4b. Required before evaluation (script 5)."""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict import run_inference
from src.utils.run_id import RUN_ID_PATTERN, next_run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    args = parser.parse_args()
    config_path = Path(args.config) if args.config else ROOT / "config" / "defaults.yaml"
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    run_id = config.get("inference", {}).get("run_id")
    run_id_base = config.get("inference", {}).get("run_id_base")
    # Prefer config run_id; else use the run_id that script 3 wrote to .current_run so pipeline uses one folder.
    if run_id is None or (isinstance(run_id, str) and run_id.strip().lower() in ("null", "")):
        current_run_file = out / ".current_run"
        if current_run_file.exists():
            run_id = current_run_file.read_text(encoding="utf-8").strip()
            if run_id and RUN_ID_PATTERN.match(run_id):
                pass  # use reserved run_id
            else:
                run_id = next_run_id(out, run_id_base=run_id_base)
        else:
            run_id = next_run_id(out, run_id_base=run_id_base)
    else:
        run_id = str(run_id).strip()
    try:
        p = run_inference(out, config, run_id=run_id)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out / run_id} (run_id={run_id})")


if __name__ == "__main__":
    main()
