"""CLI: Phase E3 leave-no-trace Cleanup (graph_distiller tent only).

    python cli_rag_cleanup.py --schema -y
    python cli_rag_cleanup.py --run-id rag_20260915_130414 -y
    python cli_rag_cleanup.py --run-id rag_... --also-export-dir -y

Never accepts a schema/table name argument. Target is always ``graph_distiller``.
Never writes public.labor_*.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_distiller.config import settings as default_settings  # noqa: E402
from graph_distiller.rag_cleanup import cleanup_with_optional_exports  # noqa: E402
from graph_distiller.rag_load import SCHEMA  # noqa: E402
from graph_distiller.rag_status import collect_status, status_to_dict  # noqa: E402


def _uri(cli: str | None) -> str:
    uri = (cli or default_settings.rag_db_uri or "").strip()
    if not uri:
        print("[FATAL] RAG_DB_URI / DB_URI required.", file=sys.stderr)
        raise SystemExit(2)
    return uri


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=f"Cleanup tent schema `{SCHEMA}` only (ADR-040 E3)"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--schema",
        action="store_true",
        help=f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE",
    )
    g.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=f"DELETE FROM {SCHEMA}.runs WHERE run_id=… (CASCADE children)",
    )
    p.add_argument("--uri", default=None, help="Override DB URI (never logged).")
    p.add_argument(
        "--also-export-dir",
        action="store_true",
        help="Also delete local export folder(s) under rag_rebuild (default off).",
    )
    p.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional single export path (must be under rebuild root).",
    )
    p.add_argument(
        "--rebuild-dir",
        type=Path,
        default=None,
        help="Local rag_rebuild root (default: Settings).",
    )
    p.add_argument(
        "--status-after",
        action="store_true",
        help="Print STATUS JSON after cleanup.",
    )
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirm.")
    args = p.parse_args(argv)

    rebuild = args.rebuild_dir or default_settings.rag_rebuild_dir
    action = (
        f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"
        if args.schema
        else f"DELETE FROM {SCHEMA}.runs WHERE run_id={args.run_id!r}"
    )
    print(f"[RAG-CLEANUP] target_schema={SCHEMA}")
    print(f"[RAG-CLEANUP] plan: {action}")
    print("[RAG-CLEANUP] Will NOT touch public.labor_law / public.labor_qna.")
    if args.also_export_dir:
        print(f"[RAG-CLEANUP] also remove export dirs under {rebuild}")

    if not args.yes:
        ans = input(f"Proceed with graph_distiller-only cleanup? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1

    uri = _uri(args.uri)
    result = cleanup_with_optional_exports(
        uri,
        drop_schema=bool(args.schema),
        run_id=args.run_id,
        also_export_dirs=bool(args.also_export_dir),
        rebuild_root=rebuild if args.also_export_dir else None,
        export_dir=args.export_dir,
    )
    payload = {"cleanup": asdict(result)}
    if args.status_after:
        report = collect_status(uri=uri, rebuild_dir=rebuild, skip_db=False)
        payload["status"] = status_to_dict(report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
