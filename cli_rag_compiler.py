"""Repo-root launcher for the GraphDistiller compiler (adds src/ to sys.path)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_distiller.cli_compiler import main

if __name__ == "__main__":
    raise SystemExit(main())
