"""CLI: load rag_rebuild export into graph_distiller.* schema (Phase D1 + E6 bridges).

    python cli_rag_load.py --from output/rag_rebuild/rag_YYYYMMDD_HHMMSS -y
    python cli_rag_load.py --status
    python cli_rag_load.py --cleanup-schema -y
    python cli_rag_load.py --interactive

Never writes public.*. Requires RAG_DB_URI / DB_URI.
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
from graph_distiller.rag_cli_bridge import (  # noqa: E402
    run_cleanup_schema,
    run_interactive_tent,
    run_status,
)
from graph_distiller.rag_load import count_graph_distiller_rows, load_export_dir  # noqa: E402


def _uri(cli: str | None) -> str:
    uri = (cli or default_settings.rag_db_uri or "").strip()
    if not uri:
        print("[FATAL] RAG_DB_URI / DB_URI required.", file=sys.stderr)
        raise SystemExit(2)
    return uri


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Load rag_rebuild JSON into graph_distiller.* (E6: also --status/--cleanup-schema/--interactive)"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--status",
        action="store_true",
        help="E1/E6: leave-no-trace STATUS (read-only).",
    )
    mode.add_argument(
        "--cleanup-schema",
        action="store_true",
        help="E3/E6: DROP SCHEMA graph_distiller CASCADE (fixed name).",
    )
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="E4/E6: open tent console (STATUS→OPTIONS→PLAN).",
    )
    p.add_argument(
        "--from",
        dest="export_dir",
        type=Path,
        default=None,
        help="Path to output/rag_rebuild/<run_id>/ (required for load).",
    )
    p.add_argument("--uri", default=None, help="Override DB URI (not logged).")
    p.add_argument(
        "--no-replace",
        action="store_true",
        help="Do not DELETE prior rows for the same run_id before insert.",
    )
    p.add_argument(
        "--also-export-dir",
        action="store_true",
        help="With --cleanup-schema: also remove local export dirs under rag_rebuild.",
    )
    p.add_argument(
        "--text",
        action="store_true",
        default=True,
        help="With --status: human text (default).",
    )
    p.add_argument(
        "--json",
        dest="status_json",
        action="store_true",
        help="With --status: JSON instead of text.",
    )
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirm.")
    args = p.parse_args(argv)

    if args.status:
        return run_status(
            uri=args.uri,
            offline=False,
            text=not args.status_json,
        )
    if args.cleanup_schema:
        return run_cleanup_schema(
            uri=args.uri,
            yes=args.yes,
            also_export_dir=args.also_export_dir,
        )
    if args.interactive:
        return run_interactive_tent(["-y"] if args.yes else None)

    if args.export_dir is None:
        print(
            "[FATAL] --from required for load (or use --status / --cleanup-schema / --interactive).",
            file=sys.stderr,
        )
        return 2

    export_dir = Path(args.export_dir)
    if not export_dir.is_dir():
        print(f"[FATAL] not a directory: {export_dir}", file=sys.stderr)
        return 2

    print(f"[RAG-LOAD] target schema=graph_distiller export={export_dir}")
    print("[RAG-LOAD] Will NOT write public.labor_law / public.labor_qna.")
    if not args.yes:
        ans = input("Proceed with graph_distiller.* load? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("Aborted.")
            return 1

    uri = _uri(args.uri)
    result = load_export_dir(
        uri,
        export_dir,
        replace=not args.no_replace,
    )
    counts = count_graph_distiller_rows(uri, result.run_id)
    payload = {
        "load": {
            "run_id": result.run_id,
            "export_dir": result.export_dir,
            "nodes": result.nodes,
            "edges": result.edges,
            "chapters": result.chapters,
            "source_map_rows": result.source_map_rows,
        },
        "db_counts": counts,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
