"""CLI: Phase E4 leave-no-trace tent console.

Interactive:
    python cli_rag_tent.py

Batch (PLAN printed, -y skips confirm):
    python cli_rag_tent.py --option S
    python cli_rag_tent.py --option L --from output/rag_rebuild/rag_... -y
    python cli_rag_tent.py --option C --schema -y
    python cli_rag_tent.py --option C --run-id rag_... -y
    python cli_rag_tent.py --option R --schema --from output/rag_rebuild/rag_... -y
    python cli_rag_tent.py --option X

Never writes public.*. Schema wipe target is always ``graph_distiller``.
E5: ``--option I --from … --resolve resume|wipe-run|wipe-schema|abort``
     ``--assess`` prints health only (no mutate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_distiller.config import settings as default_settings  # noqa: E402
from graph_distiller.rag_resume import (  # noqa: E402
    RESOLVE_CHOICES,
    assessment_to_dict,
    assessments_from_status,
    assess_export_vs_db,
    format_assessments,
    interrupted_only,
    plan_from_resolution,
    skeleton_rework_hint,
)
from graph_distiller.rag_status import collect_status, scan_rebuild_root  # noqa: E402
from graph_distiller.rag_tent import (  # noqa: E402
    OPTION_ABORT,
    OPTION_CLEANUP,
    OPTION_INTERRUPT,
    OPTION_LOAD,
    OPTION_REWORK,
    OPTION_STATUS,
    TentPlan,
    default_confirm,
    execute_plan,
    format_menu,
    plan_abort,
    plan_cleanup_run,
    plan_cleanup_schema,
    plan_load,
    plan_rework,
    plan_status,
    render_opening_status,
)


def _uri(cli: str | None) -> str:
    return (cli or default_settings.rag_db_uri or "").strip()


def _pick_export(rebuild: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    exports = [e for e in scan_rebuild_root(rebuild) if e.status == "complete"]
    if not exports:
        raise SystemExit("[FATAL] no complete export under rebuild dir; pass --from")
    print("Complete exports:")
    for i, e in enumerate(exports, 1):
        print(f"  {i}. {e.run_id}")
    raw = input("Pick number (or path): ").strip()
    if raw.isdigit():
        idx = int(raw) - 1
        if idx < 0 or idx >= len(exports):
            raise SystemExit("[FATAL] invalid pick")
        return Path(exports[idx].path)
    return Path(raw)


def _assessment_for_args(args: argparse.Namespace, rebuild: Path, uri: str):
    if args.export_dir:
        report = collect_status(uri=uri or None, rebuild_dir=rebuild, skip_db=not uri)
        return assess_export_vs_db(
            args.export_dir,
            inventory=report.graph_distiller,
            db_counts=None,
        )
    report = collect_status(uri=uri or None, rebuild_dir=rebuild, skip_db=not uri)
    assessments = assessments_from_status(report, uri=uri if report.db_queried else None)
    bad = interrupted_only(assessments)
    if not bad:
        # fall back to first export assessment
        if not assessments:
            raise SystemExit("[FATAL] no exports to assess; pass --from")
        return assessments[0]
    return bad[0]


def _build_plan_from_args(args: argparse.Namespace, rebuild: Path, uri: str) -> TentPlan:
    opt = (args.option or "").strip().upper()
    if opt == OPTION_STATUS:
        return plan_status()
    if opt == OPTION_ABORT:
        return plan_abort()
    if opt == OPTION_LOAD:
        if not args.export_dir:
            raise SystemExit("[FATAL] --from required for L")
        return plan_load(args.export_dir, replace=not args.no_replace)
    if opt == OPTION_CLEANUP:
        if args.schema:
            return plan_cleanup_schema(also_export_dir=args.also_export_dir)
        if args.run_id:
            return plan_cleanup_run(
                args.run_id, also_export_dir=args.also_export_dir
            )
        raise SystemExit("[FATAL] C requires --schema or --run-id")
    if opt == OPTION_REWORK:
        if not args.export_dir:
            raise SystemExit("[FATAL] --from required for R")
        mode = "schema" if args.schema else "run"
        rid = args.run_id or Path(args.export_dir).name
        if args.schema:
            mode = "schema"
        return plan_rework(
            cleanup_mode=mode,
            export_dir=args.export_dir,
            run_id=rid,
            also_export_dir=args.also_export_dir,
        )
    if opt == OPTION_INTERRUPT:
        if not args.resolve:
            raise SystemExit(
                "[FATAL] I requires --resolve "
                f"{'/'.join(RESOLVE_CHOICES)}"
            )
        assessment = _assessment_for_args(args, rebuild, uri)
        print(format_assessments([assessment]))
        print(f"  {skeleton_rework_hint()}")
        return plan_from_resolution(
            assessment,
            args.resolve,
            also_export_dir=args.also_export_dir,
        )
    raise SystemExit(f"[FATAL] unknown --option {args.option!r}")


def _interactive_interrupt(rebuild: Path, uri: str) -> TentPlan:
    report = collect_status(uri=uri, rebuild_dir=rebuild, skip_db=False)
    assessments = assessments_from_status(report, uri=uri)
    bad = interrupted_only(assessments)
    print(format_assessments(bad or assessments))
    print(skeleton_rework_hint())
    if not assessments:
        print("No exports — Abort")
        return plan_abort()
    target = bad[0] if bad else assessments[0]
    print(f"Focus run_id={target.run_id} health={target.health}")
    print(f"Suggested: {list(target.suggested)}")
    resolve = input(
        "Resolve [resume/wipe-run/wipe-schema/abort/none]: "
    ).strip().lower()
    also = input("Also delete local export dirs? [y/N]: ").strip().lower() in (
        "y",
        "yes",
    )
    return plan_from_resolution(target, resolve or "abort", also_export_dir=also)


def _interactive_plan(rebuild: Path, uri: str) -> TentPlan:
    print(format_menu())
    raw = input("Choose [S/L/R/C/I/X]: ").strip().upper()
    if raw == OPTION_STATUS:
        return plan_status()
    if raw == OPTION_ABORT or raw == "":
        return plan_abort()
    if raw == OPTION_INTERRUPT:
        return _interactive_interrupt(rebuild, uri)
    if raw == OPTION_LOAD:
        export = _pick_export(rebuild, None)
        return plan_load(export)
    if raw == OPTION_CLEANUP:
        mode = input("Cleanup mode [schema/run]: ").strip().lower()
        also = input("Also delete local export dirs? [y/N]: ").strip().lower() in (
            "y",
            "yes",
        )
        if mode == "schema":
            return plan_cleanup_schema(also_export_dir=also)
        if mode == "run":
            rid = input("run_id: ").strip()
            return plan_cleanup_run(rid, also_export_dir=also)
        print("Invalid cleanup mode → Abort")
        return plan_abort()
    if raw == OPTION_REWORK:
        mode = input("Wipe mode [schema/run]: ").strip().lower() or "run"
        export = _pick_export(rebuild, None)
        also = input("Also delete local export dirs before reload? [y/N]: ").strip().lower() in (
            "y",
            "yes",
        )
        return plan_rework(
            cleanup_mode=mode if mode in ("schema", "run") else "run",
            export_dir=export,
            run_id=export.name,
            also_export_dir=also,
        )
    print("Unknown choice → Abort")
    return plan_abort()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Leave-no-trace tent: STATUS → OPTIONS → PLAN → confirm"
    )
    p.add_argument(
        "--option",
        choices=["S", "L", "R", "C", "I", "X", "s", "l", "r", "c", "i", "x"],
        default=None,
        help="Batch option (skip menu). Omit for interactive.",
    )
    p.add_argument("--from", dest="export_dir", type=Path, default=None)
    p.add_argument("--schema", action="store_true", help="Cleanup/Rework: DROP SCHEMA graph_distiller")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--also-export-dir", action="store_true")
    p.add_argument("--no-replace", action="store_true")
    p.add_argument(
        "--resolve",
        choices=list(RESOLVE_CHOICES),
        default=None,
        help="E5 resolution for --option I",
    )
    p.add_argument(
        "--assess",
        action="store_true",
        help="E5: print health assessments only (no PLAN execute).",
    )
    p.add_argument("--uri", default=None)
    p.add_argument("--rebuild-dir", type=Path, default=None)
    p.add_argument(
        "--offline",
        action="store_true",
        help="Opening STATUS skips DB (interactive/batch S still needs URI for mutate).",
    )
    p.add_argument(
        "--skip-status",
        action="store_true",
        help="Do not print opening STATUS.",
    )
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirm after PLAN.")
    args = p.parse_args(argv)

    rebuild = args.rebuild_dir or default_settings.rag_rebuild_dir
    uri = _uri(args.uri)

    if args.assess:
        if not args.offline and not uri:
            print("[FATAL] URI required for --assess (or --offline).", file=sys.stderr)
            return 2
        report = collect_status(
            uri=uri or None, rebuild_dir=rebuild, skip_db=args.offline
        )
        assessments = assessments_from_status(
            report, uri=uri if report.db_queried else None
        )
        print(format_assessments(assessments))
        print(skeleton_rework_hint())
        print(
            json.dumps(
                [assessment_to_dict(a) for a in assessments],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not args.skip_status:
        if not args.offline and not uri:
            print(
                "[FATAL] RAG_DB_URI / DB_URI required (or --offline / --skip-status).",
                file=sys.stderr,
            )
            return 2
        _, text = render_opening_status(
            uri=uri or None, rebuild_dir=rebuild, offline=args.offline
        )
        print(text)
        print()

    if args.option:
        plan = _build_plan_from_args(args, rebuild, uri)
    else:
        if not uri and not args.offline:
            print("[FATAL] URI required for interactive tent.", file=sys.stderr)
            return 2
        plan = _interactive_plan(rebuild, uri)

    print(plan.format_text())

    if plan.mutates_db and not uri:
        print("[FATAL] URI required for mutating options.", file=sys.stderr)
        return 2

    if not args.yes:
        if not default_confirm("Proceed? [y/N] "):
            print("Aborted (no execute).")
            return 1
    else:
        print("[PLAN] confirmed via -y")

    if plan.option == OPTION_ABORT:
        result = execute_plan(plan, uri=uri or "", rebuild_dir=rebuild)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    result = execute_plan(plan, uri=uri, rebuild_dir=rebuild)
    if "status_text" in result:
        print(result["status_text"])
        result = {k: v for k, v in result.items() if k != "status_text"}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
