"""Tidy CSV: one metric (or check) per row, fixed column order.

`value_type` makes the file round-trippable without guessing, so spreadsheets,
pandas or another program can read values back with their original types.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable, List, Sequence

from harness.model import CheckResult, MetricRecord, Scalar

METRIC_COLUMNS: Sequence[str] = (
    "run_id",
    "scope",
    "subject",
    "name",
    "value",
    "value_type",
    "unit",
    "status",
    "threshold",
    "threshold_type",
    "source",
    "note",
)

CHECK_COLUMNS: Sequence[str] = (
    "run_id",
    "scope",
    "subject",
    "check_id",
    "passed",
    "severity",
    "message",
    "evidence",
)


def _type_name(value: Scalar) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "str"


def _encode(value: Scalar) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _decode(text: str, type_name: str) -> Scalar:
    if type_name == "null":
        return None
    if type_name == "bool":
        return text == "true"
    if type_name == "int":
        return int(text)
    if type_name == "float":
        return float(text)
    return text


def metrics_to_csv(records: Iterable[MetricRecord]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(METRIC_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for r in records:
        writer.writerow(
            {
                "run_id": r.run_id,
                "scope": r.scope,
                "subject": r.subject,
                "name": r.name,
                "value": _encode(r.value),
                "value_type": _type_name(r.value),
                "unit": r.unit,
                "status": r.status,
                "threshold": _encode(r.threshold),
                "threshold_type": _type_name(r.threshold),
                "source": r.source,
                "note": r.note,
            }
        )
    return buf.getvalue()


def checks_to_csv(checks: Iterable[CheckResult]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(CHECK_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for c in checks:
        writer.writerow(
            {
                "run_id": c.run_id,
                "scope": c.scope,
                "subject": c.subject,
                "check_id": c.check_id,
                "passed": "true" if c.passed else "false",
                "severity": c.severity,
                "message": c.message,
                "evidence": c.evidence,
            }
        )
    return buf.getvalue()


def parse_metrics_csv(text: str) -> List[MetricRecord]:
    rows = csv.DictReader(io.StringIO(text))
    return [
        MetricRecord(
            run_id=row["run_id"],
            scope=row["scope"],
            subject=row["subject"],
            name=row["name"],
            value=_decode(row["value"], row["value_type"]),
            unit=row["unit"],
            status=row["status"],
            threshold=_decode(row["threshold"], row["threshold_type"]),
            source=row["source"],
            note=row["note"],
        )
        for row in rows
    ]


def parse_checks_csv(text: str) -> List[CheckResult]:
    rows = csv.DictReader(io.StringIO(text))
    return [
        CheckResult(
            run_id=row["run_id"],
            scope=row["scope"],
            subject=row["subject"],
            check_id=row["check_id"],
            passed=row["passed"] == "true",
            severity=row["severity"],
            message=row["message"],
            evidence=row["evidence"],
        )
        for row in rows
    ]


def write_csv(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel on Windows opens Korean text correctly.
    path.write_text(text, encoding="utf-8-sig", newline="")
    return path


def read_csv(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8-sig")
