"""Serialization schema for audit output (ADR-061).

One MetricRecord = one measured number/label; one CheckResult = one pass/fail
judgement. Field order here is the CSV column order — append new fields at the
end and bump SCHEMA_VERSION when renaming or removing one.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

Scalar = Union[bool, int, float, str, None]
Status = Literal["ok", "warn", "fail", "unknown", "stale"]
Severity = Literal["error", "warn", "info"]


class MetricRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: str
    name: str
    value: Scalar
    subject: str = "global"
    run_id: str = ""
    unit: str = ""
    status: Status = "ok"
    threshold: Scalar = None
    source: str = ""
    note: str = ""

    def key(self) -> Tuple[str, str, str, str]:
        return (self.scope, self.subject, self.run_id, self.name)


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    check_id: str
    scope: str
    passed: bool
    subject: str = "global"
    run_id: str = ""
    severity: Severity = "error"
    message: str = ""
    evidence: str = ""

    def key(self) -> Tuple[str, str, str, str]:
        return (self.scope, self.subject, self.run_id, self.check_id)


class CollectorError(BaseModel):
    """A collector that could not run (missing input, parse failure)."""

    model_config = ConfigDict(extra="forbid")

    collector: str
    message: str


class AuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    audit_id: str
    generated_at: str
    inputs: Dict[str, str] = Field(default_factory=dict)
    metrics: List[MetricRecord] = Field(default_factory=list)
    checks: List[CheckResult] = Field(default_factory=list)
    collector_errors: List[CollectorError] = Field(default_factory=list)

    def failed_checks(self, severity: Severity | None = "error") -> List[CheckResult]:
        return [
            c
            for c in self.checks
            if not c.passed and (severity is None or c.severity == severity)
        ]

    def exit_code(self) -> int:
        """0 = clean, 1 = error-severity check failed, 2 = a collector could not run."""
        if self.collector_errors:
            return 2
        if self.failed_checks("error"):
            return 1
        return 0

    def summary(self) -> Dict[str, int]:
        return {
            "metrics": len(self.metrics),
            "checks": len(self.checks),
            "checks_passed": sum(1 for c in self.checks if c.passed),
            "checks_failed_error": len(self.failed_checks("error")),
            "checks_failed_warn": len(self.failed_checks("warn")),
            "collector_errors": len(self.collector_errors),
        }
