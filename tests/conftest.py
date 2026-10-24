"""Shared pytest fixtures. Does not modify src/."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def repo_root() -> Path:
    return ROOT


def _strip_file_handlers() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            root.removeHandler(handler)
            handler.close()


@pytest.fixture(autouse=True)
def _no_file_logging_leaks_into_tracked_errors_log():
    """Importing graph_distiller.local_pipeline (directly, or via e2e.run_e2e_eval /
    harness.collectors) runs configure_pipeline_logging() as an import-time side effect and
    attaches a FileHandler onto the tracked output/errors.log; cli_compiler.main() attaches
    one explicitly on every call. Either can happen during test *collection*, before any
    fixture's setup runs, so a plain "snapshot handlers, restore them later" fixture would
    just adopt the already-leaked handler as its baseline. Instead this strips every
    FileHandler both before and after each test, unconditionally — a real bug hit while
    building the audit harness (ADR-061); do not weaken this to a snapshot/restore."""
    _strip_file_handlers()
    yield
    _strip_file_handlers()
