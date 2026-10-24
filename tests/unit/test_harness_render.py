"""Unit: audit harness schema + JSON/CSV/Markdown renderers ($0)."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.model import AuditReport, CheckResult, CollectorError, MetricRecord
from harness.render import csv_out, json_out, markdown

pytestmark = pytest.mark.unit


def _report() -> AuditReport:
    return AuditReport(
        audit_id="audit_test",
        generated_at="2026-09-17T00:00:00",
        inputs={"codex": "output/codex"},
        metrics=[
            MetricRecord(scope="codex", subject="chapter_01", name="citations", value=7, unit="count"),
            MetricRecord(scope="codex", subject="chapter_01", name="density", value=0.25, threshold=0.25),
            MetricRecord(scope="config", name="rules_valid", value=True),
            MetricRecord(scope="ledger", run_id="codex_1", name="model", value="a|b, \"c\"\n한글"),
            MetricRecord(scope="ledger", run_id="codex_1", name="cost_usd", value=None, status="unknown"),
        ],
        checks=[
            CheckResult(check_id="verifier.pass", scope="codex", subject="chapter_01", passed=True),
            CheckResult(
                check_id="verifier.pass",
                scope="codex",
                subject="chapter_02",
                passed=False,
                message="Unknown DOC_IDs cited: ['X']",
                evidence="X",
            ),
            CheckResult(check_id="logs.no_dlq", scope="logs", passed=False, severity="warn"),
        ],
    )


def test_json_roundtrip_is_lossless(tmp_path: Path) -> None:
    report = _report()
    path = json_out.write_json(report, tmp_path / "metrics.json")
    assert json_out.read_json(path) == report


def test_metrics_csv_roundtrip_preserves_types(tmp_path: Path) -> None:
    report = _report()
    path = csv_out.write_csv(csv_out.metrics_to_csv(report.metrics), tmp_path / "metrics.csv")
    back = csv_out.parse_metrics_csv(csv_out.read_csv(path))
    assert back == report.metrics
    assert isinstance(back[0].value, int)
    assert isinstance(back[1].value, float)
    assert back[2].value is True
    assert back[4].value is None


def test_checks_csv_roundtrip(tmp_path: Path) -> None:
    report = _report()
    text = csv_out.checks_to_csv(report.checks)
    assert csv_out.parse_checks_csv(text) == report.checks


def test_csv_column_order_is_stable() -> None:
    header = csv_out.metrics_to_csv([]).splitlines()[0]
    assert header == ",".join(csv_out.METRIC_COLUMNS)
    assert csv_out.checks_to_csv([]).splitlines()[0] == ",".join(csv_out.CHECK_COLUMNS)


def test_exit_code_levels() -> None:
    report = _report()
    assert report.exit_code() == 1
    report.checks = [c for c in report.checks if c.severity == "warn"]
    assert report.exit_code() == 0
    report.collector_errors = [CollectorError(collector="codex", message="missing")]
    assert report.exit_code() == 2


def test_record_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        MetricRecord(scope="x", name="y", value=1, typo_field=2)


def test_markdown_lists_failures_and_metric_tables() -> None:
    text = markdown.to_markdown(_report())
    assert "판정: FAIL" in text
    assert "verifier.pass" in text and "chapter_02" in text
    assert "### codex" in text and "### ledger" in text
    assert "a\\|b" in text  # pipe escaped so the table does not break
