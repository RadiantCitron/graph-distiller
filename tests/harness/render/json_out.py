"""Lossless JSON form of AuditReport (the canonical machine format)."""

from __future__ import annotations

from pathlib import Path

from harness.model import AuditReport


def to_json(report: AuditReport) -> str:
    return report.model_dump_json(indent=2) + "\n"


def write_json(report: AuditReport, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(report), encoding="utf-8")
    return path


def read_json(path: Path) -> AuditReport:
    return AuditReport.model_validate_json(Path(path).read_text(encoding="utf-8"))
