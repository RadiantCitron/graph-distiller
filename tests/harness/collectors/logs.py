"""errors.log / state["errors"] tag counts (tags come from graph_distiller.contracts)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from graph_distiller.contracts import TAG_DLQ, TAG_FATAL, TAG_VERIFY_FAIL

from harness.collectors import Collected, rel

SCOPE = "logs"
_LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[(\w+)\] ([\w.]+): ")
_LLM_REQUEST_MARK = "HTTP Request: POST https://openrouter.ai/"


def count_error_tags(errors: Iterable[str]) -> Dict[str, int]:
    """Tag counts over error lines. `poison` keeps the E2E harness's historic definition."""
    lines = list(errors)
    return {
        "dlq": sum(1 for e in lines if TAG_DLQ in e),
        "verify_fail": sum(1 for e in lines if TAG_VERIFY_FAIL in e),
        "fatal": sum(1 for e in lines if TAG_FATAL in e),
        "poison": sum(
            1
            for e in lines
            if TAG_FATAL in e or "failed after" in e.lower() or "poison" in e.lower()
        ),
    }


def filter_since(lines: Iterable[str], since: Optional[str]) -> List[str]:
    """Keep log records stamped at/after `since` ("YYYY-MM-DD HH:MM:SS"); continuation lines follow their record."""
    if not since:
        return list(lines)
    kept: List[str] = []
    keep = False
    for line in lines:
        m = _LINE_RE.match(line)
        if m:
            keep = m.group(1) >= since
        if keep:
            kept.append(line)
    return kept


def collect(errors_log: Path, *, since: Optional[str] = None) -> Collected:
    out = Collected()
    if not out.require_input(SCOPE, errors_log):
        return out
    src = rel(errors_log)
    lines = filter_since(
        Path(errors_log).read_text(encoding="utf-8", errors="replace").splitlines(), since
    )
    note = f"since {since}" if since else "whole file"

    levels: Counter[str] = Counter()
    for line in lines:
        m = _LINE_RE.match(line)
        if m:
            levels[m.group(2)] += 1

    out.metric(SCOPE, "lines", len(lines), unit="count", source=src, note=note)
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        out.metric(SCOPE, f"level_{level.lower()}", levels.get(level, 0), unit="count", source=src, note=note)
    tags = count_error_tags(lines)
    for name, value in tags.items():
        out.metric(SCOPE, f"tag_{name}", value, unit="count", source=src, note=note)
    llm_requests = sum(1 for line in lines if _LLM_REQUEST_MARK in line)
    out.metric(SCOPE, "llm_http_requests", llm_requests, unit="count", source=src, note=note)

    out.check("logs.no_fatal", SCOPE, tags["fatal"] == 0, severity="warn",
              message=f"{tags['fatal']} FATAL line(s) ({note})", evidence=src)
    out.check("logs.no_dlq", SCOPE, tags["dlq"] == 0, severity="warn",
              message=f"{tags['dlq']} DLQ line(s) ({note})", evidence=src)
    return out
