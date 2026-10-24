"""Phase 2→3 graph scale preflight ($0) — W3-A.

Pure functions. No LLM, no Settings, no ledger I/O.
Estimates in-memory Phase 3 peak and slices or aborts when over limits.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Sequence, Tuple

from graph_distiller.schema import GraphEdge, GraphNode

logger = logging.getLogger(__name__)

# Heuristic: NetworkX + community + PageRank peak ≫ raw payload.
PHASE3_MEMORY_FACTOR = 4.0
NODE_OVERHEAD_BYTES = 256
EDGE_OVERHEAD_BYTES = 128
CHUNK_ID_OVERHEAD_BYTES = 64

DEFAULT_GRAPH_MAX_NODES = 100_000
DEFAULT_GRAPH_MAX_EDGES = 500_000
DEFAULT_GRAPH_MAX_EST_MB = 1024.0
DEFAULT_GRAPH_SCALE_ON_EXCEED: Literal["abort", "slice"] = "abort"


def estimate_graph_bytes(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
) -> int:
    """Rough Phase-3 peak RAM (bytes) from node/edge payloads × factor."""
    node_bytes = 0
    for node in nodes:
        node_bytes += NODE_OVERHEAD_BYTES
        node_bytes += len((node.id or "").encode("utf-8"))
        node_bytes += len((node.type or "").encode("utf-8"))
        node_bytes += len((node.description or "").encode("utf-8"))
        node_bytes += CHUNK_ID_OVERHEAD_BYTES * len(node.source_chunk_ids or [])
        for cid in node.source_chunk_ids or []:
            node_bytes += len(str(cid).encode("utf-8"))

    edge_bytes = 0
    for edge in edges:
        edge_bytes += EDGE_OVERHEAD_BYTES
        edge_bytes += len((edge.source or "").encode("utf-8"))
        edge_bytes += len((edge.target or "").encode("utf-8"))
        edge_bytes += len((edge.relationship or "").encode("utf-8"))
        edge_bytes += len((edge.provenance or "").encode("utf-8"))

    return int((node_bytes + edge_bytes) * PHASE3_MEMORY_FACTOR)


def _degree_map(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
) -> Dict[str, int]:
    deg: Dict[str, int] = {n.id: 0 for n in nodes}
    for edge in edges:
        if edge.source == edge.target:
            continue
        if edge.source in deg:
            deg[edge.source] += 1
        if edge.target in deg:
            deg[edge.target] += 1
    return deg


def _first_breach(
    n_nodes: int,
    n_edges: int,
    est_bytes: int,
    *,
    max_nodes: int,
    max_edges: int,
    max_est_bytes: int,
) -> str | None:
    if n_nodes > max_nodes:
        return "max_nodes"
    if n_edges > max_edges:
        return "max_edges"
    if est_bytes > max_est_bytes:
        return "max_est_mb"
    return None


def _slice_to_limits(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    *,
    max_nodes: int,
    max_edges: int,
    max_est_bytes: int,
) -> Tuple[List[GraphNode], List[GraphEdge], str]:
    """
    Keep highest-degree nodes (then chunk-support, then id), then heaviest
    edges among survivors, shrinking until under all three caps.
    """
    if not nodes:
        return [], [], "none"

    deg = _degree_map(nodes, edges)
    ranked = sorted(
        nodes,
        key=lambda n: (
            -deg.get(n.id, 0),
            -len(n.source_chunk_ids or []),
            n.id,
        ),
    )

    lo, hi = 1, min(len(ranked), max(1, max_nodes))
    best_nodes: List[GraphNode] = ranked[:hi]
    best_edges: List[GraphEdge] = []
    stop_reason = "max_nodes"

    while lo <= hi:
        mid = (lo + hi) // 2
        keep_ids = {n.id for n in ranked[:mid]}
        cand_nodes = [n for n in ranked if n.id in keep_ids]

        cand_edges = [
            e
            for e in edges
            if e.source in keep_ids
            and e.target in keep_ids
            and e.source != e.target
        ]
        cand_edges.sort(
            key=lambda e: (-float(e.weight), e.source, e.target, e.relationship)
        )
        if len(cand_edges) > max_edges:
            cand_edges = cand_edges[:max_edges]
            stop_reason = "max_edges"

        est = estimate_graph_bytes(cand_nodes, cand_edges)
        breach = _first_breach(
            len(cand_nodes),
            len(cand_edges),
            est,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_est_bytes=max_est_bytes,
        )
        if breach is None:
            best_nodes, best_edges = cand_nodes, cand_edges
            lo = mid + 1
            stop_reason = "none"
        else:
            stop_reason = breach
            hi = mid - 1

    keep_ids = {n.id for n in best_nodes}
    final_edges = [
        e
        for e in edges
        if e.source in keep_ids
        and e.target in keep_ids
        and e.source != e.target
    ]
    final_edges.sort(
        key=lambda e: (-float(e.weight), e.source, e.target, e.relationship)
    )
    if len(final_edges) > max_edges:
        final_edges = final_edges[:max_edges]
        stop_reason = "max_edges"

    while best_nodes and _first_breach(
        len(best_nodes),
        len(final_edges),
        estimate_graph_bytes(best_nodes, final_edges),
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_est_bytes=max_est_bytes,
    ):
        stop_reason = "max_est_mb"
        best_nodes = best_nodes[:-1]
        keep_ids = {n.id for n in best_nodes}
        final_edges = [
            e
            for e in final_edges
            if e.source in keep_ids and e.target in keep_ids
        ]

    return best_nodes, final_edges, stop_reason


def guard_graph_scale(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    *,
    max_nodes: int = DEFAULT_GRAPH_MAX_NODES,
    max_edges: int = DEFAULT_GRAPH_MAX_EDGES,
    max_est_mb: float = DEFAULT_GRAPH_MAX_EST_MB,
    on_exceed: Literal["abort", "slice"] = DEFAULT_GRAPH_SCALE_ON_EXCEED,
) -> Tuple[List[GraphNode], List[GraphEdge], Dict[str, Any]]:
    """
    Phase 2→3 scale gate.

    Returns (nodes, edges, info). info['action'] is pass|slice|abort.
    On abort, returned lists are the originals (caller must stop Phase 3).
    """
    node_list = list(nodes)
    edge_list = list(edges)
    max_nodes_i = max(1, int(max_nodes))
    max_edges_i = max(0, int(max_edges))
    max_est_mb_f = max(0.0, float(max_est_mb))
    max_est_bytes = int(max_est_mb_f * 1024 * 1024)
    mode: Literal["abort", "slice"] = (
        "slice" if str(on_exceed).strip().lower() == "slice" else "abort"
    )

    est_before = estimate_graph_bytes(node_list, edge_list)
    breach = _first_breach(
        len(node_list),
        len(edge_list),
        est_before,
        max_nodes=max_nodes_i,
        max_edges=max_edges_i,
        max_est_bytes=max_est_bytes,
    )

    limits = {
        "max_nodes": max_nodes_i,
        "max_edges": max_edges_i,
        "max_est_mb": max_est_mb_f,
    }
    base_info: Dict[str, Any] = {
        "nodes_before": len(node_list),
        "edges_before": len(edge_list),
        "est_bytes_before": est_before,
        "est_mb_before": round(est_before / (1024 * 1024), 6),
        "limits": limits,
        "on_exceed": mode,
    }

    if breach is None:
        info = {
            **base_info,
            "action": "pass",
            "stop_reason": "none",
            "nodes_after": len(node_list),
            "edges_after": len(edge_list),
            "nodes_dropped": 0,
            "edges_dropped": 0,
            "est_bytes_after": est_before,
            "est_mb_after": round(est_before / (1024 * 1024), 6),
        }
        return node_list, edge_list, info

    if mode == "abort":
        info = {
            **base_info,
            "action": "abort",
            "stop_reason": breach,
            "nodes_after": len(node_list),
            "edges_after": len(edge_list),
            "nodes_dropped": 0,
            "edges_dropped": 0,
            "est_bytes_after": est_before,
            "est_mb_after": round(est_before / (1024 * 1024), 6),
        }
        logger.warning(
            "Graph scale abort: reason=%s nodes=%d edges=%d est_mb=%.3f limits=%s",
            breach,
            len(node_list),
            len(edge_list),
            est_before / (1024 * 1024),
            limits,
        )
        return node_list, edge_list, info

    kept_nodes, kept_edges, slice_reason = _slice_to_limits(
        node_list,
        edge_list,
        max_nodes=max_nodes_i,
        max_edges=max_edges_i,
        max_est_bytes=max_est_bytes,
    )
    est_after = estimate_graph_bytes(kept_nodes, kept_edges)
    still = _first_breach(
        len(kept_nodes),
        len(kept_edges),
        est_after,
        max_nodes=max_nodes_i,
        max_edges=max_edges_i,
        max_est_bytes=max_est_bytes,
    )
    if still is not None and not kept_nodes:
        info = {
            **base_info,
            "action": "abort",
            "stop_reason": still,
            "nodes_after": 0,
            "edges_after": 0,
            "nodes_dropped": len(node_list),
            "edges_dropped": len(edge_list),
            "est_bytes_after": 0,
            "est_mb_after": 0.0,
        }
        return [], [], info

    info = {
        **base_info,
        "action": "slice",
        "stop_reason": slice_reason if slice_reason != "none" else breach,
        "nodes_after": len(kept_nodes),
        "edges_after": len(kept_edges),
        "nodes_dropped": len(node_list) - len(kept_nodes),
        "edges_dropped": len(edge_list) - len(kept_edges),
        "est_bytes_after": est_after,
        "est_mb_after": round(est_after / (1024 * 1024), 6),
    }
    logger.info(
        "Graph scale slice: reason=%s dropped_nodes=%d dropped_edges=%d "
        "after_nodes=%d after_edges=%d est_mb=%.3f",
        info["stop_reason"],
        info["nodes_dropped"],
        info["edges_dropped"],
        info["nodes_after"],
        info["edges_after"],
        info["est_mb_after"],
    )
    return kept_nodes, kept_edges, info
