"""CLI: Phase E1 leave-no-trace STATUS (read-only).

    python cli_rag_status.py
    python cli_rag_status.py --offline
    python cli_rag_status.py --text

Never writes public.* or graph_distiller.*. Requires RAG_DB_URI / DB_URI unless --offline.
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
from graph_distiller.rag_status import (  # noqa: E402
    collect_status,
    format_status_text,
    status_to_dict,
)


def _uri(cli: str | None) -> str:
    return (cli or default_settings.rag_db_uri or "").strip()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Leave-no-trace STATUS: public COUNT + graph_distiller inventory + exports"
    )
    p.add_argument("--uri", default=None, help="Override DB URI (never logged).")
    p.add_argument(
        "--rebuild-dir",
        type=Path,
        default=None,
        help="Local rag_rebuild root (default: Settings.rag_rebuild_dir).",
    )
    p.add_argument(
        "--offline",
        action="store_true",
        help="Skip DB; report local exports only.",
    )
    p.add_argument(
        "--text",
        action="store_true",
        help="Human-readable text instead of JSON.",
    )
    args = p.parse_args(argv)

    rebuild = args.rebuild_dir or default_settings.rag_rebuild_dir
    uri = _uri(args.uri)
    if not args.offline and not uri:
        print(
            "[FATAL] RAG_DB_URI / DB_URI required (or pass --offline).",
            file=sys.stderr,
        )
        return 2

    report = collect_status(uri=uri or None, rebuild_dir=rebuild, skip_db=args.offline)
    if args.text:
        print(format_status_text(report))
    else:
        print(json.dumps(status_to_dict(report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
