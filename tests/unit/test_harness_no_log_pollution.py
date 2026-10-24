"""Unit: running an audit must not write to the errors.log it reads (observability placement).

Regression for a real bug found while building the harness: importing
e2e.run_e2e_eval (for the repomap collector) transitively imports
graph_distiller.local_pipeline, whose module body calls configure_pipeline_logging()
as an import-time side effect and attaches a FileHandler to errors.log. Left
unfixed, every `python tests/audit.py audit` run appended its own INFO/DEBUG
lines to the product's error log, corrupting the very artifact being audited
and making `compare` report false drift between two otherwise-identical runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import cli

pytestmark = pytest.mark.unit


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "data" / "prompts").mkdir(parents=True)
    (root / "data" / "lexicon").mkdir(parents=True)
    (root / "data" / "checkpoints").mkdir(parents=True)
    (root / "output" / "codex").mkdir(parents=True)
    (root / "output" / "runs").mkdir(parents=True)
    log = root / "output" / "errors.log"
    log.write_text("2026-09-17 10:00:00,000 [INFO] x: pre-existing line\n", encoding="utf-8")
    return root


def test_build_report_does_not_grow_errors_log(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    log = root / "output" / "errors.log"
    before = log.read_text(encoding="utf-8")

    cli.build_report(root)
    cli.build_report(root)  # twice: any leaked handler would append on the second call too

    assert log.read_text(encoding="utf-8") == before


def test_repeated_audits_produce_identical_log_metrics(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    first = cli.build_report(root)
    second = cli.build_report(root)

    def log_metrics(report):
        return sorted((m.name, m.value) for m in report.metrics if m.scope == "logs")

    assert log_metrics(first) == log_metrics(second)
