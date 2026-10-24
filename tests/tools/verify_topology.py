"""Skeleton topology verification — W2-C follow-up.

Four graph-theory metrics on the Phase 2 checkpoint:
1. Hub concentration (top-5% nodes' degree share) — target >= 0.30
2. Modularity of the Phase 3 community partition — target >= 0.3
3. Edge weight skewness (max, top-1% mean) — measured, no target
4. PageRank Gini coefficient — target >= 0.5

Default is measurement-only ($0). Optional --apply-lexicon / --embed-merge
apply Reduce 3/2 before metrics (embedding cost if --embed-merge).
ADR-030: executable lives under tests/tools, interpretation in tests/eval.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import networkx as nx

from graph_distiller.config import PROJECT_ROOT, settings as default_settings
from graph_distiller.graph.build import (
    build_nx_graph,
    compute_pagerank,
    detect_communities,
    drop_orphan_nodes,
)
from graph_distiller.graph.checkpoint import load_extraction_checkpoint
from graph_distiller.graph.extraction import canonicalize_extracted_graph
from graph_distiller.graph.lexicon import load_entity_alias_lexicon
from graph_distiller.schema import GraphEdge, GraphNode

DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "phase2_extraction.json"
REPORTS_DIR = PROJECT_ROOT / "tests" / "reports"

HUB_TOP_PCT = 0.05
HUB_TARGET = 0.30
MODULARITY_TARGET = 0.3
GINI_TARGET = 0.5
TOP_WEIGHT_PCT = 0.01


def hub_concentration(graph: nx.Graph, top_pct: float = HUB_TOP_PCT) -> float:
    """Degree-sum share of the top ``top_pct`` nodes by degree."""
    degrees = sorted((d for _n, d in graph.degree()), reverse=True)
    total = sum(degrees)
    if not degrees or total <= 0:
        return 0.0
    top_n = max(1, math.ceil(len(degrees) * top_pct))
    return sum(degrees[:top_n]) / total


def edge_weight_skew(
    edges: Sequence[GraphEdge], top_pct: float = TOP_WEIGHT_PCT
) -> Dict[str, float]:
    """Max weight and mean weight of the top ``top_pct`` edges."""
    weights = sorted((float(e.weight) for e in edges), reverse=True)
    if not weights:
        return {"max_weight": 0.0, "top1pct_avg_weight": 0.0}
    top_n = max(1, math.ceil(len(weights) * top_pct))
    top = weights[:top_n]
    return {
        "max_weight": weights[0],
        "top1pct_avg_weight": sum(top) / len(top),
    }


def gini(values: Sequence[float]) -> float:
    """Gini coefficient of a non-negative distribution."""
    vals = sorted(float(v) for v in values if float(v) > 0.0)
    n = len(vals)
    if n == 0:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    weighted = sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(vals))
    return weighted / (n * total)


def community_modularity(
    graph: nx.Graph, toc: Dict[int, List[str]]
) -> Optional[float]:
    """nx.community.modularity over the Phase 3 partition."""
    if not toc or graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        return None
    communities = [sorted(members) for members in toc.values() if members]
    if not communities:
        return None
    return float(
        nx.community.modularity(graph, communities, weight="weight")
    )


def apply_graph_reduces(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    *,
    apply_lexicon: bool = False,
    embed_merge: bool = False,
    embeddings: Any = None,
    sim_threshold: Optional[float] = None,
    max_embed_nodes: Optional[int] = None,
    lexicon_path: Optional[Path] = None,
) -> tuple[List[GraphNode], List[GraphEdge], Dict[str, Any]]:
    """Optional Reduce 3 (lexicon) + Reduce 1/2 (canonicalize) before metrics.

    Default is a no-op so measurement-only runs stay comparable.
    """
    applied: Dict[str, Any] = {
        "lexicon": False,
        "embed_merge": False,
        "sim_threshold": None,
        "lexicon_path": None,
        "lexicon_entries": 0,
    }
    if not apply_lexicon and not embed_merge:
        return list(nodes), list(edges), applied

    alias = None
    if apply_lexicon:
        path = lexicon_path or default_settings.canon_alias_lexicon_path
        alias = load_entity_alias_lexicon(path)
        applied["lexicon"] = True
        applied["lexicon_path"] = str(path)
        applied["lexicon_entries"] = len({c for c in alias.values()}) if alias else 0

    thresh = (
        float(sim_threshold)
        if sim_threshold is not None
        else float(default_settings.canon_embed_sim_threshold)
    )
    max_n = (
        int(max_embed_nodes)
        if max_embed_nodes is not None
        else int(default_settings.canon_embed_max_nodes)
    )
    if embed_merge:
        applied["embed_merge"] = True
        applied["sim_threshold"] = thresh

    reduced_nodes, reduced_edges = canonicalize_extracted_graph(
        list(nodes),
        list(edges),
        embed_merge=embed_merge,
        embeddings=embeddings,
        sim_threshold=thresh,
        max_embed_nodes=max_n,
        alias_lexicon=alias,
    )
    applied["nodes_after"] = len(reduced_nodes)
    applied["edges_after"] = len(reduced_edges)
    return reduced_nodes, reduced_edges, applied


def compute_topology_report(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
) -> Dict[str, Any]:
    """All four metrics from an extracted graph (checkpoint-shaped input)."""
    raw_graph = build_nx_graph(list(nodes), list(edges))
    hub_share = hub_concentration(raw_graph)
    skew = edge_weight_skew(edges)

    pruned = raw_graph.copy()
    _orphan_errors: List[str] = []
    pruned, orphans = drop_orphan_nodes(pruned, _orphan_errors)
    toc, algorithm = detect_communities(pruned)
    modularity = community_modularity(pruned, toc)
    pagerank = compute_pagerank(pruned)
    pr_gini = gini(list(pagerank.values()))

    metrics = {
        "hub_top5_degree_share": {
            "value": hub_share,
            "target": HUB_TARGET,
            "pass": hub_share >= HUB_TARGET,
        },
        "modularity": {
            "value": modularity,
            "target": MODULARITY_TARGET,
            "pass": modularity is not None and modularity >= MODULARITY_TARGET,
        },
        "edge_weight_skew": {
            "value": skew,
            "target": None,
            "pass": None,
        },
        "pagerank_gini": {
            "value": pr_gini,
            "target": GINI_TARGET,
            "pass": pr_gini >= GINI_TARGET,
        },
    }

    return {
        "meta": {
            "nodes_in": len(nodes),
            "edges_in": len(edges),
            "nodes_pruned": pruned.number_of_nodes(),
            "orphans_dropped": len(orphans),
            "communities": len(toc),
            "algorithm": algorithm,
        },
        "metrics": metrics,
        "targets": {
            "hub_top5_degree_share": HUB_TARGET,
            "modularity": MODULARITY_TARGET,
            "pagerank_gini": GINI_TARGET,
        },
        "note": (
            "Measurement only. PASS/FAIL vs reference targets; merge/threshold "
            "tuning is a separate directive (ADR-032/042)."
        ),
    }


def format_report(report: Dict[str, Any]) -> str:
    meta = report["meta"]
    m = report["metrics"]
    lines = [
        f"Input: {meta['nodes_in']} nodes / {meta['edges_in']} edges"
        f" -> pruned {meta['nodes_pruned']} (orphans {meta['orphans_dropped']})"
        f" / {meta['communities']} communities [{meta['algorithm']}]",
        f"1. Hub top-5% degree share: {m['hub_top5_degree_share']['value']:.4f}"
        f" (target >= {m['hub_top5_degree_share']['target']})"
        f" -> {'PASS' if m['hub_top5_degree_share']['pass'] else 'FAIL'}",
        (
            f"2. Modularity: {m['modularity']['value']:.4f}"
            if m["modularity"]["value"] is not None
            else "2. Modularity: n/a"
        )
        + f" (target >= {m['modularity']['target']})"
        + f" -> {'PASS' if m['modularity']['pass'] else 'FAIL'}",
        "3. Edge weight skew: "
        f"max={m['edge_weight_skew']['value']['max_weight']:.4f}"
        f" top-1% avg={m['edge_weight_skew']['value']['top1pct_avg_weight']:.4f}"
        " (measured, no target)",
        f"4. PageRank Gini: {m['pagerank_gini']['value']:.4f}"
        f" (target >= {m['pagerank_gini']['target']})"
        f" -> {'PASS' if m['pagerank_gini']['pass'] else 'FAIL'}",
        "NOTE: measurement only - no canonicalize/merge tuning (ADR-032/042).",
    ]
    applied = report.get("applied") or {}
    if applied.get("lexicon") or applied.get("embed_merge"):
        lines[-1] = (
            "NOTE: Reduce applied before metrics — "
            f"lexicon={applied.get('lexicon')} "
            f"embed_merge={applied.get('embed_merge')} "
            f"sim_threshold={applied.get('sim_threshold')}."
        )
    return "\n".join(lines) + "\n"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Skeleton topology verify ($0)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="phase2_extraction.json path",
    )
    parser.add_argument("--json", type=Path, default=None, help="optional JSON dump")
    parser.add_argument(
        "--apply-lexicon",
        action="store_true",
        help="Apply Reduce 3 (entity_aliases.json) then spelling canonicalize",
    )
    parser.add_argument(
        "--embed-merge",
        action="store_true",
        help="Apply Reduce 2 using CANON_EMBED_SIM_THRESHOLD (live embedder)",
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=None,
        help="Override CANON_ALIAS_LEXICON_PATH",
    )
    parser.add_argument(
        "--sim-threshold",
        type=float,
        default=None,
        help="Override CANON_EMBED_SIM_THRESHOLD for --embed-merge",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path: Path = args.checkpoint
    if not path.is_file():
        print(f"[FATAL] checkpoint not found: {path}", file=sys.stderr)
        return 1
    nodes, edges = load_extraction_checkpoint(path.parent, filename=path.name)
    embeddings = None
    if args.embed_merge:
        from graph_distiller.adapters.openrouter import default_embedder

        embeddings = default_embedder(default_settings)
    nodes, edges, applied = apply_graph_reduces(
        nodes,
        edges,
        apply_lexicon=bool(args.apply_lexicon),
        embed_merge=bool(args.embed_merge),
        embeddings=embeddings,
        sim_threshold=args.sim_threshold,
        lexicon_path=args.lexicon,
    )
    report = compute_topology_report(nodes, edges)
    report["checkpoint"] = str(path)
    report["applied"] = applied
    if applied.get("lexicon") or applied.get("embed_merge"):
        report["note"] = (
            "Reduce applied before metrics: lexicon="
            f"{applied.get('lexicon')} embed_merge={applied.get('embed_merge')} "
            f"sim_threshold={applied.get('sim_threshold')}."
        )
    sys.stdout.write(format_report(report))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
