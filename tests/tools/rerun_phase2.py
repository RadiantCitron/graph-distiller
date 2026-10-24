"""Re-run Phase 1+2 only and overwrite the Phase 2 checkpoint.

Harness tool (ADR-030): lives under tests/tools, never imported by main.
Prints the Phase 2 cost estimate before spending; use --yes to skip the
interactive confirmation (batch mode).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graph_distiller.config import settings
from graph_distiller.cost import estimate_phase2_cost, format_cost_status
from graph_distiller.graph.extraction import run_phase2
from graph_distiller.ingestion.pipeline import run_phase1
from graph_distiller.pricing import fetch_openrouter_rate_book
from graph_distiller.schema import empty_state


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-run Phase 1+2, overwrite checkpoint")
    p.add_argument("--yes", "-y", action="store_true", help="skip confirmation")
    return p.parse_args()


def main() -> int:
    state = empty_state()

    print("[Phase 1] Semantic Chunking ...")
    t1 = time.perf_counter()
    state = run_phase1(state)
    n_chunks = len(state.get("chunks") or [])
    print(f"  chunks={n_chunks} ({time.perf_counter() - t1:.1f}s)")

    book = fetch_openrouter_rate_book(
        [settings.extraction_model], base_url=settings.openrouter_base_url
    )
    est = estimate_phase2_cost(
        list(state.get("chunks") or []),
        settings.extraction_model,
        rate=book.rate_for(settings.extraction_model),
    )
    print(
        f"[cost preview] Phase 2 model={settings.extraction_model} "
        f"chunks={est['chunk_count']} "
        f"in~={est['input_tokens']} out~={est['output_tokens']} "
        f"-> {format_cost_status(est['cost_usd'], est['cost_status'])}"
    )
    if not args.yes:
        if not sys.stdin.isatty():
            print("[FATAL] no TTY; pass --yes for batch mode", file=sys.stderr)
            return 1
        if input("Proceed with Phase 2? [y/N] ").strip().lower() != "y":
            print("aborted")
            return 1

    print("[Phase 2] LLM Graph Extraction ...")
    t2 = time.perf_counter()
    state, usage = run_phase2(state)
    print(
        f"  nodes={len(state['nodes'])} edges={len(state['edges'])} "
        f"errors={len(state['errors'])} ({time.perf_counter() - t2:.1f}s)"
    )
    print(usage.summary_line("Phase 2"))
    print(f"  checkpoint overwritten: {settings.checkpoint_dir / 'phase2_extraction.json'}")
    return 0


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main())
