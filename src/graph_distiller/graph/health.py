"""Phase 2/3 graph health metrics ($0) — W2-A / ADR-042.

Pure functions. No LLM, no Settings, no ledger I/O.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

import networkx as nx

from graph_distiller.schema import GraphEdge, GraphNode


def compute_graph_health(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
) -> Dict[str, Any]:
    """
    Skeleton health for ledger / harness.

    - orphan_ratio: degree==0 nodes / total nodes (pre Phase-3 drop)
    - avg_edge_weight_by_provenance: mean weight per provenance channel
    """
    n_nodes = len(nodes)
    n_edges = len(edges)
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node.id)
    for edge in edges:
        if edge.source == edge.target:
            continue
        if edge.source not in graph or edge.target not in graph:
            continue
        graph.add_edge(edge.source, edge.target)

    orphan_count = sum(1 for _n, deg in graph.degree() if deg == 0)
    orphan_ratio = (orphan_count / n_nodes) if n_nodes else 0.0

    by_prov: Dict[str, List[float]] = defaultdict(list)
    all_w: List[float] = []
    for edge in edges:
        w = float(edge.weight)
        all_w.append(w)
        prov = (edge.provenance or "unknown").strip() or "unknown"
        by_prov[prov].append(w)

    avg_overall = (sum(all_w) / len(all_w)) if all_w else 0.0
    avg_by_prov = {
        k: (sum(vs) / len(vs) if vs else 0.0) for k, vs in sorted(by_prov.items())
    }
    counts_by_prov = {k: len(vs) for k, vs in sorted(by_prov.items())}

    return {
        "total_nodes": n_nodes,
        "total_edges": n_edges,
        "orphan_count": orphan_count,
        "orphan_ratio": orphan_ratio,
        "avg_edge_weight": avg_overall,
        "avg_edge_weight_by_provenance": avg_by_prov,
        "edge_count_by_provenance": counts_by_prov,
    }
