"""Shim: delegates to tests/e2e/run_e2e_eval.py (Prompt Diet controller)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "e2e" / "run_e2e_eval.py"

if __name__ == "__main__":
    sys.argv[0] = str(_TARGET)
    runpy.run_path(str(_TARGET), run_name="__main__")
