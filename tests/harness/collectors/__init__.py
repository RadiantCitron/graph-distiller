"""Collectors: pure readers of product artifacts → metrics + checks.

Contract for every collector module:
- `collect(...) -> Collected` with explicit input paths (no hidden globals).
- A missing optional artifact is reported as a failed `*.input_present` check
  (warn), not raised — an audit on a fresh checkout still completes.
- Raise `CollectorUnavailable` only when the collector genuinely cannot run
  (corrupt input, import failure); the runner records it as exit code 2.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from harness.model import CheckResult, MetricRecord, Scalar

ROOT = Path(__file__).resolve().parents[3]


class CollectorUnavailable(RuntimeError):
    pass


@dataclass
class Collected:
    metrics: List[MetricRecord] = field(default_factory=list)
    checks: List[CheckResult] = field(default_factory=list)

    def metric(self, scope: str, name: str, value: Scalar, **kw) -> MetricRecord:
        rec = MetricRecord(scope=scope, name=name, value=value, **kw)
        self.metrics.append(rec)
        return rec

    def check(self, check_id: str, scope: str, passed: bool, **kw) -> CheckResult:
        res = CheckResult(check_id=check_id, scope=scope, passed=bool(passed), **kw)
        self.checks.append(res)
        return res

    def extend(self, other: "Collected") -> "Collected":
        self.metrics.extend(other.metrics)
        self.checks.extend(other.checks)
        return self

    def require_input(self, scope: str, path: Path, *, severity: str = "warn") -> bool:
        present = Path(path).exists()
        self.check(
            f"{scope}.input_present",
            scope,
            present,
            severity=severity,
            message="" if present else f"input not found: {rel(path)}",
            evidence=rel(path),
        )
        return present


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
