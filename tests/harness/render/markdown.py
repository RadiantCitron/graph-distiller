"""Human-readable audit report. Same data as JSON/CSV, grouped for reading."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

from harness.model import AuditReport, CheckResult, MetricRecord, Scalar


def _cell(value: Scalar) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _group(records: List[MetricRecord]) -> Dict[str, List[MetricRecord]]:
    groups: Dict[str, List[MetricRecord]] = OrderedDict()
    for r in records:
        groups.setdefault(r.scope, []).append(r)
    return groups


def _check_line(c: CheckResult) -> str:
    where = c.subject if not c.run_id else f"{c.run_id} / {c.subject}"
    line = f"- **[{c.severity}] {c.check_id}** ({c.scope} · {where}): {_cell(c.message)}"
    if c.evidence:
        line += f"\n  - 근거: `{_cell(c.evidence)}`"
    return line


def to_markdown(report: AuditReport) -> str:
    s = report.summary()
    verdict = {0: "PASS", 1: "FAIL (error 체크 실패)", 2: "ERROR (수집기 실행 불가)"}[
        report.exit_code()
    ]
    lines: List[str] = [
        f"# GraphDistiller 전수검사 보고서 — {report.audit_id}",
        "",
        f"- 생성 시각: {report.generated_at}",
        f"- 스키마 버전: {report.schema_version}",
        f"- **판정: {verdict}** (종료 코드 {report.exit_code()})",
        "",
        "## 요약",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 지표 수 | {s['metrics']} |",
        f"| 체크 수 | {s['checks']} |",
        f"| 통과 | {s['checks_passed']} |",
        f"| 실패 (error) | {s['checks_failed_error']} |",
        f"| 실패 (warn) | {s['checks_failed_warn']} |",
        f"| 수집기 오류 | {s['collector_errors']} |",
        "",
    ]
    if report.inputs:
        lines += ["## 입력", "", "| 이름 | 경로 |", "|---|---|"]
        lines += [f"| {k} | `{v}` |" for k, v in report.inputs.items()]
        lines.append("")

    if report.collector_errors:
        lines += ["## 수집기 오류", ""]
        lines += [f"- **{e.collector}**: {_cell(e.message)}" for e in report.collector_errors]
        lines.append("")

    failed = report.failed_checks(None)
    lines += ["## 실패한 체크", ""]
    lines += [_check_line(c) for c in failed] if failed else ["없음."]
    lines.append("")

    lines += ["## 지표", ""]
    for scope, records in _group(report.metrics).items():
        lines += [
            f"### {scope}",
            "",
            "| run | subject | name | value | unit | status | threshold |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in records:
            lines.append(
                f"| {_cell(r.run_id or None)} | {_cell(r.subject)} | {_cell(r.name)} | "
                f"{_cell(r.value)} | {_cell(r.unit or None)} | {r.status} | {_cell(r.threshold)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
