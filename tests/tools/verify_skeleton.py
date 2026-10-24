"""W2-V: Phase 2 checkpoint skeleton health ($0, no LLM).

ADR-030: executable lives under tests/tools, not tests/eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import networkx as nx

from graph_distiller.config import PROJECT_ROOT
from graph_distiller.graph.checkpoint import load_extraction_checkpoint
from graph_distiller.schema import GraphEdge, GraphNode

DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "phase2_extraction.json"
WEIGHT_HEAVY = 2.0


def compute_skeleton_stats(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    *,
    heavy_min: float = WEIGHT_HEAVY,
) -> Dict[str, Any]:
    """Three skeleton metrics from an already-extracted graph."""
    n_nodes = len(nodes)
    n_edges = len(edges)
    weights = [float(e.weight) for e in edges]
    max_w = max(weights) if weights else 0.0
    heavy = sum(1 for w in weights if w >= heavy_min)
    heavy_pct = (100.0 * heavy / n_edges) if n_edges else 0.0

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id)
    for edge in edges:
        if edge.source == edge.target:
            continue
        if edge.source not in graph or edge.target not in graph:
            continue
        graph.add_edge(edge.source, edge.target)

    giant_pct = 0.0
    giant_size = 0
    if n_nodes:
        components: List[int] = [len(c) for c in nx.connected_components(graph)]
        giant_size = max(components) if components else 0
        giant_pct = 100.0 * giant_size / n_nodes

    return {
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "max_edge_weight": max_w,
        "edges_weight_ge_2_count": heavy,
        "edges_weight_ge_2_pct": heavy_pct,
        "giant_component_nodes": giant_size,
        "giant_component_pct": giant_pct,
    }


def format_report(stats: Dict[str, Any]) -> str:
    return (
        f"Total Nodes: {stats['total_nodes']}\n"
        f"Total Edges: {stats['total_edges']}\n"
        f"Max Edge Weight: {stats['max_edge_weight']:.4f}\n"
        f"Edges with weight >= 2.0: {stats['edges_weight_ge_2_count']}"
        f" ({stats['edges_weight_ge_2_pct']:.2f}%)\n"
        f"Giant Component: {stats['giant_component_nodes']} nodes"
        f" ({stats['giant_component_pct']:.2f}%)\n"
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 skeleton health ($0)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="phase2_extraction.json path",
    )
    parser.add_argument("--json", type=Path, default=None, help="optional JSON dump")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path: Path = args.checkpoint
    if not path.is_file():
        print(f"[FATAL] checkpoint not found: {path}", file=sys.stderr)
        return 1
    nodes, edges = load_extraction_checkpoint(path.parent, filename=path.name)
    stats = compute_skeleton_stats(nodes, edges)
    report = format_report(stats)
    sys.stdout.write(report)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
