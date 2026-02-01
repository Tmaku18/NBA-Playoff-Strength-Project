"""Run inference pipeline: predictions.json and figures."""
import argparse
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict import run_inference


def _next_run_id(outputs_dir: Path, run_id_base: int | None = None) -> str:
    """Auto-increment: find existing run_NNN dirs, return run_{max+1:03d}.
    If outputs dir has no run_* subdirs and run_id_base N is set, return run_{N:03d} (e.g. run_019)."""
    outputs_dir = Path(outputs_dir)
    pattern = re.compile(r"^run_(\d+)$", re.I)
    numbers = []
    if outputs_dir.exists():
        for p in outputs_dir.iterdir():
            if p.is_dir() and pattern.match(p.name):
                numbers.append(int(pattern.match(p.name).group(1)))
    if not numbers and run_id_base is not None:
        return f"run_{run_id_base:03d}"
    next_n = max(numbers, default=0) + 1
    return f"run_{next_n:03d}"


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
    if run_id is None or (isinstance(run_id, str) and run_id.strip().lower() in ("null", "")):
        # Reuse run_id reserved by script 3 so one pipeline run = one run folder
        current_run_file = out / ".current_run"
        if current_run_file.exists():
            run_id = current_run_file.read_text(encoding="utf-8").strip()
            if run_id and re.match(r"^run_\d+$", run_id, re.I):
                pass  # use reserved run_id
            else:
                run_id = _next_run_id(out, run_id_base=run_id_base)
        else:
            run_id = _next_run_id(out, run_id_base=run_id_base)
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
