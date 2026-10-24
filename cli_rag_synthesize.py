"""CLI: Phase D2a — synthesize chapters from a rag_rebuild export.

    python cli_rag_synthesize.py --from output/rag_rebuild/rag_... --max-chapters 2 -y
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

from graph_distiller.rag_synthesize import run_rag_synthesize  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 4 on rag_rebuild export → codex/")
    p.add_argument("--from", dest="export_dir", type=Path, required=True)
    p.add_argument(
        "--max-chapters",
        type=int,
        default=2,
        help="Cap communities for Phase 4 (default 2, sample cost control).",
    )
    p.add_argument("-y", "--yes", action="store_true")
    args = p.parse_args(argv)

    export_dir = Path(args.export_dir)
    if not export_dir.is_dir():
        print(f"[FATAL] missing export dir: {export_dir}", file=sys.stderr)
        return 2
    if args.max_chapters < 1:
        print("[FATAL] --max-chapters >= 1", file=sys.stderr)
        return 2

    print(
        "[DEPRECATED] cli_rag_synthesize.py -> python cli_rag_compiler.py "
        "(Phase 4 is part of the compiler). This script still synthesizes an "
        "existing rag_rebuild export.",
        file=sys.stderr,
    )
    print(
        f"[RAG-SYNTH] export={export_dir} max_chapters={args.max_chapters} "
        "(uses SYNTHESIS_MODEL — costs money)"
    )
    if not args.yes:
        ans = input("Proceed with sample Phase 4? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("Aborted.")
            return 1

    result = run_rag_synthesize(export_dir, max_chapters=args.max_chapters)
    print(
        json.dumps(
            {
                "codex_dir": str(result.codex_dir),
                "drafts": result.drafts,
                "poisoned": result.poisoned,
                "chapters_kept": result.chapters_kept,
                "written": result.written,
                "phase4_usage": result.phase4_usage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.drafts > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
