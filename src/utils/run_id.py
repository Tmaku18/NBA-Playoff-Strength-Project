"""Resolve run directories (run_NNN or run_NNN_MM-DD) under an outputs dir."""
import re
from pathlib import Path


def latest_run_id(outputs_dir: Path) -> str | None:
    """Return the latest run dir name (run_NNN or run_NNN_MM-DD) that has predictions.json.
    Prefers run_\\d+ then run_\\d+_*; latest by numeric part then by name."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return None
    # run_NNN or run_NNN_MM-DD
    pattern = re.compile(r"^run_(\d+)(?:_.*)?$", re.I)
    candidates: list[tuple[int, str]] = []
    for p in outputs_dir.iterdir():
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if m and (p / "predictions.json").exists():
            candidates.append((int(m.group(1)), p.name))
    if not candidates:
        return None
    # Sort by run number desc, then by name desc (so run_025_02-15 beats run_025)
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][1]


def resolve_run_dir(out_dir: Path, run_id: str) -> Path:
    """Return the Path to the run directory for run_id under out_dir.
    If out_dir/run_id exists, use it. Else find a subdir whose name starts with run_id (e.g. run_025_02-15).
    Otherwise return out_dir/run_id so callers get a consistent path."""
    out_dir = Path(out_dir)
    run_id = run_id.strip()
    direct = out_dir / run_id
    if direct.is_dir():
        return direct
    # Match run_025 -> run_025_02-15
    prefix = run_id.lower()
    for p in out_dir.iterdir():
        if p.is_dir() and p.name.lower().startswith(prefix):
            return p
    return direct
