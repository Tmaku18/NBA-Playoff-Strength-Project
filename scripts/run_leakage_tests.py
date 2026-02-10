"""Run leakage tests to ensure no future info leaks into features.

What this does:
- Calls src.utils.leakage_tests.run_all() to verify t-1 and roster rules.
- Ensures features use only data available before prediction date.
- Run before training (part of full pipeline). Exits with error if tests fail."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.leakage_tests import run_all

if __name__ == "__main__":
    run_all()
    print("Leakage tests passed.")
