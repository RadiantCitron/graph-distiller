"""Audit harness CLI (ADR-061): validate | audit | compare.

$0 by default — the only paid path is a future `--judge` step (not wired
here yet; scoring/judge lands in a later plan step). Exit codes:
  0 = clean, 1 = an error-severity check failed, 2 = a collector could not run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from harness.collectors import Collected, CollectorUnavailable
from harness.collectors import chunks as chunks_collector
from harness.collectors import codex as codex_collector
from harness.collectors import config as config_collector
from harness.collectors import graph as graph_collector
from harness.collectors import judge as judge_collector
from harness.collectors import ledger as ledger_collector
from harness.collectors import logs as logs_collector
from harness.collectors import repomap as repomap_collector
from harness.model import AuditReport, CollectorError, MetricRecord
from harness.render import csv_out, json_out, markdown

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def _detach_product_log_handlers() -> None:
    """Importing product modules (e2e.run_e2e_eval -> graph_distiller.local_pipeline) attaches a
    FileHandler onto errors.log as an import-time side effect (observability.py
    configure_pipeline_logging). The audit reads that log; it must not write to it, or every
    audit run pollutes the very file the next audit compares against."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()


def _default_inputs(root: Path) -> dict:
    return {
        "checkpoint_dir": str(root / "data" / "checkpoints"),
        "prompts_dir": str(root / "data" / "prompts"),
        "lexicon_path": str(root / "data" / "lexicon" / "entity_aliases.json"),
        "codex_dir": str(root / "output" / "codex"),
        "runs_dir": str(root / "output" / "runs"),
        "errors_log": str(root / "output" / "errors.log"),
    }


def build_report(root: Path, *, since: Optional[str] = None, judge: bool = False) -> AuditReport:
    _detach_product_log_handlers()
    inputs = _default_inputs(root)
    audit_id = "audit_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    report = AuditReport(
        audit_id=audit_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        inputs=inputs,
    )

    def run(name: str, fn) -> None:
        try:
            collected = fn()
        except CollectorUnavailable as exc:
            report.collector_errors.append(CollectorError(collector=name, message=str(exc)))
            return
        except Exception as exc:  # noqa: BLE001 — a broken collector must not kill the whole audit
            report.collector_errors.append(CollectorError(collector=name, message=f"{type(exc).__name__}: {exc}"))
            return
        report.metrics.extend(collected.metrics)
        report.checks.extend(collected.checks)

    checkpoint_dir = Path(inputs["checkpoint_dir"])
    runs_dir = Path(inputs["runs_dir"])
    run("config", lambda: config_collector.collect(Path(inputs["prompts_dir"]), Path(inputs["lexicon_path"])))
    run("logs", lambda: logs_collector.collect(Path(inputs["errors_log"]), since=since))
    run("ledger", lambda: ledger_collector.collect(runs_dir))
    run("chunks", lambda: chunks_collector.collect(checkpoint_dir))
    run("graph", lambda: graph_collector.collect(checkpoint_dir))
    run("repomap", lambda: repomap_collector.collect(checkpoint_dir))
    run("codex", lambda: codex_collector.collect(Path(inputs["codex_dir"]), runs_dir))

    if judge:
        def run_judge_collector() -> Collected:
            diet = repomap_collector.compute_diet(checkpoint_dir)
            if diet is None:
                out = Collected()
                out.check("judge.input_present", "judge", False, severity="warn",
                          message="no chapters to judge (missing checkpoints or empty TOC)")
                return out
            return judge_collector.collect(diet)

        run("judge", run_judge_collector)
    return report


def write_report(report: AuditReport, out_dir: Path) -> Path:
    out_dir = Path(out_dir) / report.audit_id
    json_out.write_json(report, out_dir / "metrics.json")
    csv_out.write_csv(csv_out.metrics_to_csv(report.metrics), out_dir / "metrics.csv")
    csv_out.write_csv(csv_out.checks_to_csv(report.checks), out_dir / "checks.csv")
    (out_dir / "report.md").write_text(markdown.to_markdown(report), encoding="utf-8")
    json_out.write_json(report, REPORTS_DIR / "latest_audit.json")
    return out_dir


def _load_report(path: Path) -> AuditReport:
    path = Path(path)
    candidate = path / "metrics.json" if path.is_dir() else path
    return json_out.read_json(candidate)


def _diff_metrics(base: List[MetricRecord], head: List[MetricRecord]) -> List[str]:
    base_map = {m.key(): m.value for m in base}
    head_map = {m.key(): m.value for m in head}
    lines: List[str] = []
    for key in sorted(set(base_map) | set(head_map)):
        b, h = base_map.get(key, "<missing>"), head_map.get(key, "<missing>")
        if b != h:
            scope, subject, run_id, name = key
            where = f"{scope}/{subject}" + (f"@{run_id}" if run_id else "")
            lines.append(f"  {where} :: {name}: {b!r} -> {h!r}")
    return lines


def cmd_audit(args: argparse.Namespace) -> int:
    report = build_report(Path(args.root), since=args.since, judge=bool(args.judge))
    out_dir = write_report(report, Path(args.out_dir))
    sys.stdout.write(markdown.to_markdown(report))
    print(f"\n[audit] written to {out_dir}")
    return report.exit_code()


def cmd_validate(args: argparse.Namespace) -> int:
    """Control-room only (prompts + lexicon today; rules once ADR-060 lands). No checkpoints/ledgers needed."""
    root = Path(args.root)
    inputs = _default_inputs(root)
    try:
        collected = config_collector.collect(Path(inputs["prompts_dir"]), Path(inputs["lexicon_path"]))
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] control-room collector crashed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    errors = [c for c in collected.checks if not c.passed and c.severity == "error"]
    if errors:
        for c in errors:
            print(f"[FATAL] {c.check_id} ({c.subject}): {c.message}", file=sys.stderr)
        return 2
    print("[validate] control room OK (prompts + lexicon parse and pass schema checks)")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    baseline = _load_report(Path(args.baseline))
    current = _load_report(Path(args.current)) if args.current else build_report(Path(args.root))
    diffs = _diff_metrics(baseline.metrics, current.metrics)
    if diffs:
        print(f"[compare] {len(diffs)} metric(s) differ from baseline {baseline.audit_id}:")
        for line in diffs:
            print(line)
        return 1
    print(f"[compare] no metric differences vs baseline {baseline.audit_id}")
    return 0


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="GraphDistiller audit harness ($0 by default)")
    parser.add_argument("--root", type=Path, default=root_default, help="repo root")
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="run every $0 collector and write json/csv/md")
    p_audit.add_argument("--out-dir", type=Path, default=REPORTS_DIR)
    p_audit.add_argument("--since", type=str, default=None, help="only count errors.log lines at/after this timestamp")
    p_audit.add_argument("--judge", action="store_true", help="also run the paid LLM Prompt Diet judge (real $ cost)")
    p_audit.set_defaults(func=cmd_audit)

    p_validate = sub.add_parser("validate", help="control-room only: fail fast before any LLM call")
    p_validate.set_defaults(func=cmd_validate)

    p_compare = sub.add_parser("compare", help="diff a metrics.json/audit dir against a baseline")
    p_compare.add_argument("--baseline", type=Path, required=True)
    p_compare.add_argument("--current", type=Path, default=None, help="default: run a fresh audit")
    p_compare.set_defaults(func=cmd_compare)

    return parser.parse_args(list(argv) if argv is not None else None)


def _force_utf8_stdio() -> None:
    """Windows consoles default to cp949/cp1252; report text is Korean + em dashes."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    _force_utf8_stdio()
    args = _parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
