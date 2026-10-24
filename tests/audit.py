"""GraphDistiller audit harness entrypoint (ADR-061).

    python tests/audit.py audit                 # run every $0 collector, write json/csv/md
    python tests/audit.py validate               # control-room only, fail fast before any LLM call
    python tests/audit.py compare --baseline ...  # diff current state against a saved baseline

$0 for all three commands above (no LLM calls). See tests/harness/cli.py for
the implementation; this file only wires up sys.path the same way
tests/e2e/run_e2e_eval.py does.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for p in (SRC, TESTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from harness.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
