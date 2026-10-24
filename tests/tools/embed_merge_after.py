"""W2-C after: apply Reduce 2 (embed merge) to a Phase 2 checkpoint.

Does **not** flip product default ``CANON_EMBED_MERGE`` (stays off).
Does **not** re-run Phase 2 Map (LLM). Measures Reduce 2 on the saved graph.

ADR-030: executable under tests/tools (not tests/eval).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graph_distiller.config import PROJECT_ROOT, settings as default_settings
from graph_distiller.graph.checkpoint import (
    load_extraction_checkpoint,
    save_extraction_checkpoint,
)
from graph_distiller.graph.extraction import canonicalize_extracted_graph
from graph_distiller.graph.health import compute_graph_health
from graph_distiller.schema import GraphEdge, GraphNode

_VS = importlib.util.spec_from_file_location(
    "verify_skeleton",
    Path(__file__).resolve().parent / "verify_skeleton.py",
)
_vs_mod = importlib.util.module_from_spec(_VS)
assert _VS and _VS.loader
_VS.loader.exec_module(_vs_mod)
DEFAULT_CHECKPOINT = _vs_mod.DEFAULT_CHECKPOINT
compute_skeleton_stats = _vs_mod.compute_skeleton_stats
format_report = _vs_mod.format_report

REPORTS_DIR = PROJECT_ROOT / "tests" / "reports"


class MockEmbedder:
    """Deterministic $0 embedder for harness unit tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            h = abs(hash(text.strip().lower())) % (10**9)
            out.append([((h >> (i * 3)) & 7) / 7.0 for i in range(8)])
        return out


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> Dict[str, Any]:
    keys = sorted(set(before) | set(after))
    out: Dict[str, Any] = {}
    for k in keys:
        b, a = before.get(k), after.get(k)
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            out[k] = a - b
        else:
            out[k] = {"before": b, "after": a}
    return out


def apply_embed_merge_after(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    *,
    embeddings: Any,
    sim_threshold: float,
    max_embed_nodes: int,
) -> Tuple[List[GraphNode], List[GraphEdge], Dict[str, Any]]:
    """Before → Reduce 2 canonicalize → after. Returns comparison dict."""
    before_skel = compute_skeleton_stats(nodes, edges)
    before_health = compute_graph_health(nodes, edges)

    after_nodes, after_edges = canonicalize_extracted_graph(
        list(nodes),
        list(edges),
        embed_merge=True,
        embeddings=embeddings,
        sim_threshold=sim_threshold,
        max_embed_nodes=max_embed_nodes,
    )

    after_skel = compute_skeleton_stats(after_nodes, after_edges)
    after_health = compute_graph_health(after_nodes, after_edges)

    comparison = {
        "before": {"skeleton": before_skel, "graph_health": before_health},
        "after": {"skeleton": after_skel, "graph_health": after_health},
        "delta": {
            "skeleton": _delta(before_skel, after_skel),
            "graph_health": _delta(before_health, after_health),
        },
        "sim_threshold": sim_threshold,
        "max_embed_nodes": max_embed_nodes,
        "product_default_canon_embed_merge": False,
    }
    return after_nodes, after_edges, comparison


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Apply embed-merge (Reduce 2) to Phase 2 checkpoint — after harness"
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock embedder ($0). Default: live Embedder (embedding cost).",
    )
    p.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Write comparison JSON (default: tests/reports/embed_merge_after_<ts>.json)",
    )
    p.add_argument(
        "--write-checkpoint",
        type=Path,
        default=None,
        help="Optional: write after nodes/edges JSON (explicit path; never auto-overwrite default)",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path: Path = args.checkpoint
    if not path.is_file():
        print(f"[FATAL] checkpoint not found: {path}", file=sys.stderr)
        return 1

    nodes, edges = load_extraction_checkpoint(path.parent, filename=path.name)
    sim = float(default_settings.canon_embed_sim_threshold)
    max_n = int(default_settings.canon_embed_max_nodes)

    if args.mock:
        embeddings: Any = MockEmbedder()
        mode = "mock"
    else:
        from graph_distiller.adapters.openrouter import default_embedder

        embeddings = default_embedder(default_settings)
        mode = "live"

    after_nodes, after_edges, comparison = apply_embed_merge_after(
        nodes,
        edges,
        embeddings=embeddings,
        sim_threshold=sim,
        max_embed_nodes=max_n,
    )
    comparison["mode"] = mode
    comparison["checkpoint"] = str(path)

    print("=== BEFORE (checkpoint as stored) ===")
    sys.stdout.write(format_report(comparison["before"]["skeleton"]))
    print("=== AFTER (Reduce 2 embed merge applied) ===")
    sys.stdout.write(format_report(comparison["after"]["skeleton"]))
    dskel = comparison["delta"]["skeleton"]
    print(
        f"mode={mode} sim_threshold={sim} "
        f"delta_nodes={dskel.get('total_nodes')} "
        f"delta_edges={dskel.get('total_edges')} "
        f"delta_heavy_pct={dskel.get('edges_weight_ge_2_pct')}"
    )
    print("NOTE: product default CANON_EMBED_MERGE remains off (ADR-042).")

    out_json = args.json
    if out_json is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_json = REPORTS_DIR / f"embed_merge_after_{ts}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_json}")

    if args.write_checkpoint is not None:
        dest = args.write_checkpoint
        save_extraction_checkpoint(
            after_nodes,
            after_edges,
            dest.parent,
            filename=dest.name,
        )
        print(f"wrote after checkpoint {dest}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
