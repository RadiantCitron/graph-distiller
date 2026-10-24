"""Phase E6: thin CLI bridges so load/skeleton expose tent lifecycle flags.

Delegates to existing STATUS / Cleanup / tent modules — no parallel WRITE paths.
App never GRANTs roles (see docs/design/attachments/graph-distiller-db-role.md).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from graph_distiller.config import settings as default_settings
from graph_distiller.rag_cleanup import cleanup_with_optional_exports
from graph_distiller.rag_load import SCHEMA
from graph_distiller.rag_status import collect_status, format_status_text, status_to_dict


def resolve_uri(cli_uri: str | None) -> str:
    uri = (cli_uri or default_settings.rag_db_uri or "").strip()
    if not uri:
        print("[FATAL] RAG_DB_URI / DB_URI required.", file=sys.stderr)
        raise SystemExit(2)
    return uri


def run_status(
    *,
    uri: str | None = None,
    rebuild_dir: Path | None = None,
    offline: bool = False,
    text: bool = True,
) -> int:
    """E1 STATUS via shared API (used by --status flags)."""
    rebuild = rebuild_dir or default_settings.rag_rebuild_dir
    resolved = "" if offline else resolve_uri(uri)
    report = collect_status(
        uri=resolved or None,
        rebuild_dir=rebuild,
        skip_db=offline,
    )
    if text:
        print(format_status_text(report))
    else:
        print(json.dumps(status_to_dict(report), ensure_ascii=False, indent=2))
    return 0


def run_cleanup_schema(
    *,
    uri: str | None = None,
    yes: bool = False,
    also_export_dir: bool = False,
    rebuild_dir: Path | None = None,
    status_after: bool = True,
) -> int:
    """E3 DROP SCHEMA graph_distiller only (fixed name — no schema CLI arg)."""
    rebuild = rebuild_dir or default_settings.rag_rebuild_dir
    print(f"[RAG-CLEANUP] target_schema={SCHEMA}")
    print(f"[RAG-CLEANUP] plan: DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    print("[RAG-CLEANUP] Will NOT touch public.labor_law / public.labor_qna.")
    if not yes:
        ans = input("Proceed with graph_distiller-only cleanup? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1
    resolved = resolve_uri(uri)
    result = cleanup_with_optional_exports(
        resolved,
        drop_schema=True,
        also_export_dirs=also_export_dir,
        rebuild_root=rebuild if also_export_dir else None,
    )
    payload: dict = {"cleanup": asdict(result)}
    if status_after:
        report = collect_status(uri=resolved, rebuild_dir=rebuild, skip_db=False)
        payload["status"] = status_to_dict(report)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def run_interactive_tent(argv: Optional[Sequence[str]] = None) -> int:
    """E4 tent console — import root CLI main with sys.path already set by callers."""
    # Prefer package-free root launcher when present.
    from graph_distiller.config import PROJECT_ROOT

    root = PROJECT_ROOT
    launcher = root / "cli_rag_tent.py"
    if not launcher.is_file():
        print("[FATAL] cli_rag_tent.py not found.", file=sys.stderr)
        return 2
    import runpy

    # runpy as __main__ would re-exec; call main() instead
    sys.path.insert(0, str(root))
    import cli_rag_tent  # type: ignore

    return int(cli_rag_tent.main(list(argv) if argv is not None else None))
