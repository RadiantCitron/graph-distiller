"""output/runs/*/ledger.json — every run's cost, tokens, graph counts, chapters, plus latest-vs-previous deltas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from harness.collectors import Collected, CollectorUnavailable, rel

SCOPE = "ledger"
_IMPROVEMENT_SCALARS = (
    "orphans_dropped",
    "extracted_nodes",
    "nodes_after_orphan",
    "communities_detected",
    "communities_kept",
    "chapters_in_toc",
    "drafts",
    "poisoned_count",
)
_TREND_KEYS = ("poisoned_count", "drafts", "chapters_in_toc", "cost_usd_total", "tokens_total")


def _num(value: Any) -> Optional[float | int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def load_ledgers(runs_dir: Path) -> List[Dict[str, Any]]:
    ledgers: List[Dict[str, Any]] = []
    for path in sorted(Path(runs_dir).glob("*/ledger.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CollectorUnavailable(f"unreadable ledger {rel(path)}: {exc}") from exc
        data["_path"] = rel(path)
        ledgers.append(data)
    return ledgers


def _run_totals(ledger: Mapping[str, Any]) -> Dict[str, Optional[float | int]]:
    imp = ledger.get("improvement") or {}
    cost = imp.get("cost_usd") or {}
    tokens = imp.get("tokens") or {}
    totals: Dict[str, Optional[float | int]] = {k: _num(imp.get(k)) for k in _IMPROVEMENT_SCALARS}
    totals["cost_usd_total"] = _num(cost.get("phase2_plus_4"))
    totals["tokens_total"] = _num(tokens.get("phase2_plus_4"))
    return totals


def _collect_run(out: Collected, ledger: Mapping[str, Any]) -> None:
    run_id = str(ledger.get("run_id") or Path(ledger["_path"]).parent.name)
    src = ledger["_path"]

    def m(name: str, value: Any, **kw: Any) -> None:
        out.metric(SCOPE, name, value, run_id=run_id, source=src, **kw)

    m("duration_s", _num(ledger.get("duration_s")), unit="s")
    m("exit_code", _num(ledger.get("exit_code")))
    m("finished", bool(ledger.get("finished_at")))

    for name, value in _run_totals(ledger).items():
        unit = "usd" if name.startswith("cost") else ("tokens" if name.startswith("tokens") else "count")
        m(name, value, unit=unit, status="ok" if value is not None else "unknown")

    for phase, rec in sorted((ledger.get("phases") or {}).items()):
        rec = rec or {}
        m(f"{phase}.duration_s", _num(rec.get("duration_s")), unit="s")
        for key, value in sorted((rec.get("counts") or {}).items()):
            if _num(value) is not None:
                m(f"{phase}.count.{key}", value, unit="count")
        usage = rec.get("usage") or {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "total_cost_usd"):
            if key in usage:
                m(f"{phase}.usage.{key}", _num(usage[key]), unit="usd" if "cost" in key else "tokens")
        estimate = rec.get("estimate") or {}
        for key in ("total_tokens", "cost_usd"):
            if key in estimate:
                m(f"{phase}.estimate.{key}", _num(estimate[key]), unit="usd" if "cost" in key else "tokens")

    chapters = ledger.get("chapters") or []
    retries_total = 0
    for ch in chapters:
        subject = f"chapter_{int(ch.get('chapter_id', -1)):02d}"
        retries = int(ch.get("retries") or 0)
        retries_total += retries
        out.metric(SCOPE, "chapter.chars", _num(ch.get("chars")), run_id=run_id, subject=subject, unit="chars", source=src)
        out.metric(SCOPE, "chapter.cited_doc_ids", len(ch.get("cited_doc_ids") or []), run_id=run_id, subject=subject, unit="count", source=src)
        out.metric(SCOPE, "chapter.retries", retries, run_id=run_id, subject=subject, unit="count", source=src)
        out.metric(SCOPE, "chapter.poisoned", bool(ch.get("poisoned")), run_id=run_id, subject=subject, source=src)
        out.metric(SCOPE, "chapter.drafted", bool(ch.get("drafted")), run_id=run_id, subject=subject, source=src)
    m("chapters.retries_total", retries_total, unit="count")
    m("errors", len(ledger.get("errors") or []), unit="count")

    totals = _run_totals(ledger)
    out.check("ledger.exit_code_zero", SCOPE, ledger.get("exit_code") == 0, run_id=run_id,
              severity="warn", message=f"exit_code={ledger.get('exit_code')}", evidence=src)
    out.check("ledger.no_poison", SCOPE, (totals["poisoned_count"] or 0) == 0, run_id=run_id,
              severity="warn", message=f"poisoned_count={totals['poisoned_count']}", evidence=src)
    toc, drafts = totals["chapters_in_toc"], totals["drafts"]
    if toc is not None and drafts is not None:
        out.check("ledger.all_toc_drafted", SCOPE, drafts >= toc, run_id=run_id, severity="warn",
                  message=f"drafts={drafts} chapters_in_toc={toc}", evidence=src)


def collect(runs_dir: Path) -> Collected:
    out = Collected()
    if not out.require_input(SCOPE, runs_dir):
        return out
    ledgers = load_ledgers(runs_dir)
    out.metric(SCOPE, "runs", len(ledgers), unit="count", source=rel(runs_dir))
    for ledger in ledgers:
        _collect_run(out, ledger)

    complete = [l for l in ledgers if l.get("improvement")]
    if len(complete) >= 2:
        prev, last = complete[-2], complete[-1]
        prev_t, last_t = _run_totals(prev), _run_totals(last)
        last_id = str(last.get("run_id"))
        for key in _TREND_KEYS:
            a, b = prev_t.get(key), last_t.get(key)
            out.metric(
                "ledger_trend",
                f"delta.{key}",
                None if a is None or b is None else b - a,
                run_id=last_id,
                status="ok" if a is not None and b is not None else "unknown",
                note=f"vs {prev.get('run_id')}",
                source=last["_path"],
            )
    return out
