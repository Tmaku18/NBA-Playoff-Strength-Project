"""Resolve run directories (run_NNN or run_NNN_MM-DD_HHMM) under an outputs dir."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

RUN_ID_PATTERN = re.compile(r"^run_(\d+)(?:_.+)?$", re.I)


def next_run_id(outputs_dir: Path, run_id_base: int | None = None) -> str:
    """Return the next run folder name: ``run_NNN_MM-DD_HHMM``.

    The numeric part increments from existing ``run_*`` dirs; when none exist and
    ``run_id_base`` is set, that base is used (e.g. 25 -> run_025). The timestamp
    suffix makes each pipeline invocation unique even in fresh output roots.
    """
    outputs_dir = Path(outputs_dir)
    numbers: list[int] = []
    if outputs_dir.exists():
        for p in outputs_dir.iterdir():
            if not p.is_dir():
                continue
            m = RUN_ID_PATTERN.match(p.name)
            if m:
                numbers.append(int(m.group(1)))
    if not numbers and run_id_base is not None:
        n = run_id_base
    else:
        n = max(numbers, default=0) + 1
    stamp = datetime.now().strftime("%m-%d_%H%M")
    return f"run_{n:03d}_{stamp}"


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
