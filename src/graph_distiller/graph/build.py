"""Phase 3: NetworkX graph build, community detection, PageRank (SPEC §3/§6)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

import networkx as nx

from graph_distiller.config import get_settings
from graph_distiller.schema import (
    ChunkMeta,
    GraphEdge,
    GraphNode,
    GraphDistillerState,
    Phase3Stats,
    empty_phase3_stats,
    empty_state,
)

if TYPE_CHECKING:
    from graph_distiller.ports import GraphCompute

logger = logging.getLogger(__name__)


def select_top_communities(
    toc: Dict[int, List[str]],
    top_k: int,
    errors: List[str] | None = None,
) -> Dict[int, List[str]]:
    """
    Optional PoC filter (ADR-055): keep only the top_k communities by node count.

    top_k <= 0 → no cap (return toc unchanged). Opt-in only for smoke/cost PoC.
    When capping, smaller communities are dropped from TOC (not synthesized)
    and chapter ids are reindexed to 0..kept-1.
    """
    if not toc:
        return {}
    if top_k <= 0:
        logger.info(
            "PoC TOC filter skipped (poc_top_chapters=%s): keeping all %d communities",
            top_k,
            len(toc),
        )
        return {int(k): sorted(members) for k, members in toc.items()}

    ranked = sorted(
        toc.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    kept = ranked[:top_k]
    dropped = ranked[top_k:]

    new_toc: Dict[int, List[str]] = {
        new_id: sorted(members) for new_id, (_, members) in enumerate(kept)
    }

    for new_id, (old_id, members) in enumerate(kept):
        logger.info(
            "PoC TOC keep: chapter %s (was %s) nodes=%d",
            new_id,
            old_id,
            len(members),
        )

    for old_id, members in dropped:
        msg = (
            f"[PoC] Chapter excluded from Phase 4 "
            f"(old_id={old_id}, nodes={len(members)}): not in top {top_k}"
        )
        logger.info(msg)
        if errors is not None:
            errors.append(msg)

    logger.info(
        "PoC TOC filter: %d communities -> top %d for Phase 4",
        len(toc),
        len(new_toc),
    )
    return new_toc


def build_nx_graph(nodes: List[GraphNode], edges: List[GraphEdge]) -> nx.Graph:
    """State nodes/edges → undirected weighted networkx.Graph()."""
    graph = nx.Graph()
    for node in nodes:
        graph.add_node(
            node.id,
            type=node.type,
            description=node.description,
            source_chunk_ids=list(node.source_chunk_ids),
        )

    for edge in edges:
        if edge.source not in graph or edge.target not in graph:
            continue
        if edge.source == edge.target:
            continue
        if graph.has_edge(edge.source, edge.target):
            graph[edge.source][edge.target]["weight"] += float(edge.weight)
        else:
            graph.add_edge(
                edge.source,
                edge.target,
                weight=float(edge.weight),
                relationship=edge.relationship,
            )
    return graph


def drop_orphan_nodes(
    graph: nx.Graph, errors: List[str]
) -> Tuple[nx.Graph, Set[str]]:
    """
    MUST (SPEC §6): remove degree==0 nodes before community detection,
    log `Orphan Node Dropped: {node_id}` into errors.
    """
    orphans = {n for n, deg in graph.degree() if deg == 0}
    for node_id in sorted(orphans):
        msg = f"Orphan Node Dropped: {node_id}"
        errors.append(msg)
        logger.info(msg)
        graph.remove_node(node_id)
    return graph, orphans


def absorb_small_communities(
    graph: nx.Graph,
    communities: List[List[str]],
    min_size: int,
) -> List[List[str]]:
    """
    Merge communities smaller than min_size into the neighbor with the
    strongest cut (sum of edge weights). Isolated tiny components stay
    as-is — no edges means no invented chapter grouping (ADR-025 stage 2).
    """
    if min_size <= 1 or len(communities) <= 1:
        return [list(c) for c in communities if c]

    groups: List[Set[str]] = [set(members) for members in communities if members]
    frozen: Set[int] = set()

    def node_index() -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for idx, members in enumerate(groups):
            for node_id in members:
                mapping[node_id] = idx
        return mapping

    while True:
        index = node_index()
        candidates = [
            i
            for i, members in enumerate(groups)
            if i not in frozen and 0 < len(members) < min_size
        ]
        if not candidates:
            break
        candidates.sort(key=lambda i: (len(groups[i]), min(groups[i])))
        small_i = candidates[0]
        cuts: Dict[int, float] = {}
        for node_id in groups[small_i]:
            if node_id not in graph:
                continue
            for nbr in graph.neighbors(node_id):
                other = index.get(nbr)
                if other is None or other == small_i:
                    continue
                cuts[other] = cuts.get(other, 0.0) + float(
                    graph[node_id][nbr].get("weight", 1.0)
                )
        if not cuts:
            frozen.add(small_i)
            continue
        target = max(cuts, key=lambda j: (cuts[j], len(groups[j]), -j))
        groups[target] |= groups[small_i]
        groups[small_i].clear()
        frozen.clear()

    return [sorted(members) for members in groups if members]


def drop_thin_communities(
    toc: Dict[int, List[str]],
    nodes: List[GraphNode],
    chunks: List[ChunkMeta],
    min_chunks: int,
    errors: List[str],
) -> Dict[int, List[str]]:
    """
    Exclude chapters whose members share too little grounding material
    (ADR-057): a chapter backed by a single short source chunk cannot
    sustain the multi-paragraph, per-paragraph-cited chapter shape that
    ``SYNTH_SYSTEM`` requires, which forces the Synthesizer to either
    fabricate ungrounded claims or omit citations (verified poison root
    cause — not Part 2 truncation, not few-shot wording).

    min_chunks <= 1 disables this filter (no exclusion). A chapter's
    grounding size is the number of *unique* non-empty source chunks
    referenced by its member nodes — the same chunk pool RepoMap Part 2
    draws from (``agents/repomap.py``), computed independently here to
    keep Phase 3 free of a Phase 4 import (architecture boundary).
    """
    if min_chunks <= 1 or not toc or not chunks:
        # No chunk data to judge grounding by (e.g. state built without
        # Phase 1 chunks, as in some unit tests) — fail open, don't drop.
        return {int(k): sorted(v) for k, v in toc.items()}

    chunk_has_content = {
        c.chunk_id for c in chunks if (c.content or "").strip()
    }
    node_chunks: Dict[str, List[str]] = {
        n.id: list(n.source_chunk_ids) for n in nodes
    }

    kept: List[Tuple[int, List[str]]] = []
    for old_id, members in sorted(toc.items()):
        unique_chunks: Set[str] = set()
        for node_id in members:
            for chunk_id in node_chunks.get(node_id, []):
                if chunk_id in chunk_has_content:
                    unique_chunks.add(chunk_id)
        if len(unique_chunks) < min_chunks:
            msg = (
                f"[MinGrounding] Chapter excluded from Phase 4 "
                f"(old_id={old_id}, members={len(members)}, "
                f"unique_chunks={len(unique_chunks)}): below "
                f"min_chapter_grounding_chunks={min_chunks}"
            )
            errors.append(msg)
            logger.info(msg)
            continue
        kept.append((old_id, sorted(members)))

    return {new_id: members for new_id, (_, members) in enumerate(kept)}


def detect_communities(
    graph: nx.Graph,
    *,
    resolution: float | None = None,
    min_community_size: int | None = None,
) -> Tuple[Dict[int, List[str]], str]:
    """
    Community detection → TOC (chapter map).

    Prefer cdlib Leiden; fall back to networkx Louvain (SPEC allows either).
    Returns ({community_id: [node_id, ...]}, algorithm) where algorithm is
    "leiden" | "louvain" | "".
    """
    if graph.number_of_nodes() == 0:
        return {}, ""

    res = (
        get_settings().louvain_resolution if resolution is None else resolution
    )
    min_size = (
        get_settings().min_community_size
        if min_community_size is None
        else min_community_size
    )

    communities: List[List[str]]
    algorithm = ""

    try:
        from cdlib.algorithms import leiden  # type: ignore

        try:
            result = leiden(graph, weights="weight", resolution=res)
        except TypeError:
            result = leiden(graph)
        communities = [list(members) for members in result.communities]
        algorithm = "leiden"
        logger.info("Community detection: cdlib Leiden (%d communities)", len(communities))
    except Exception as exc:  # noqa: BLE001 — fallback path
        logger.warning(
            "cdlib Leiden unavailable (%s); falling back to networkx Louvain",
            exc,
        )
        from networkx.algorithms.community import louvain_communities

        raw = louvain_communities(
            graph, weight="weight", resolution=res, seed=42
        )
        communities = [list(members) for members in raw]
        algorithm = "louvain"
        logger.info(
            "Community detection: networkx Louvain resolution=%s (%d communities)",
            res,
            len(communities),
        )

    before = len(communities)
    communities = absorb_small_communities(graph, communities, min_size)
    if len(communities) != before:
        logger.info(
            "Phase 3 absorb min_community_size=%s: %d -> %d communities",
            min_size,
            before,
            len(communities),
        )

    # Stable chapter ids: sort communities by size desc, then by min node id
    communities.sort(key=lambda members: (-len(members), min(members) if members else ""))

    toc: Dict[int, List[str]] = {}
    for chapter_id, members in enumerate(communities):
        toc[chapter_id] = sorted(members)
    return toc, algorithm


def compute_pagerank(graph: nx.Graph) -> Dict[str, float]:
    """
    nx.pagerank(weight='weight') over the (orphan-pruned) graph.

    NetworkX 3.x defaults to a SciPy backend; if SciPy is missing we fall back
    to a pure-Python power iteration so Phase 3 does not crash.
    """
    if graph.number_of_nodes() == 0:
        return {}
    try:
        return nx.pagerank(graph, weight="weight")
    except ModuleNotFoundError as exc:
        # Typically: No module named 'scipy' (or numpy)
        logger.warning(
            "nx.pagerank backend unavailable (%s); using pure-Python PageRank",
            exc,
        )
        return _pagerank_python(graph, weight="weight")


def _pagerank_python(
    graph: nx.Graph,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
    weight: str = "weight",
) -> Dict[str, float]:
    """Minimal power-iteration PageRank (no SciPy/NumPy required)."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    index = {node: i for i, node in enumerate(nodes)}
    scores = [1.0 / n] * n

    # Out-weight sums
    out_weight = [0.0] * n
    for u, v, data in graph.edges(data=True):
        w = float(data.get(weight, 1.0))
        out_weight[index[u]] += w
        if not graph.is_directed():
            out_weight[index[v]] += w

    for _ in range(max_iter):
        new_scores = [(1.0 - alpha) / n] * n
        dangling_sum = sum(
            scores[i] for i, ow in enumerate(out_weight) if ow == 0.0
        )
        dangling_share = alpha * dangling_sum / n

        for u, v, data in graph.edges(data=True):
            w = float(data.get(weight, 1.0))
            iu, iv = index[u], index[v]
            if out_weight[iu] > 0:
                new_scores[iv] += alpha * scores[iu] * (w / out_weight[iu])
            if not graph.is_directed() and out_weight[iv] > 0:
                new_scores[iu] += alpha * scores[iv] * (w / out_weight[iv])

        for i in range(n):
            new_scores[i] += dangling_share

        err = sum(abs(new_scores[i] - scores[i]) for i in range(n))
        scores = new_scores
        if err < n * tol:
            break

    return {node: scores[index[node]] for node in nodes}

def apply_phase3_to_state(
    state: GraphDistillerState,
    graph: nx.Graph,
    toc: Dict[int, List[str]],
    scores: Dict[str, float],
    dropped: Set[str],
) -> GraphDistillerState:
    """Filter dropped orphans and attach pagerank_score onto GraphNode list."""
    kept_nodes: List[GraphNode] = []
    for node in state.get("nodes") or []:
        if node.id in dropped:
            continue
        if node.id not in graph:
            continue
        kept_nodes.append(
            node.model_copy(update={"pagerank_score": float(scores.get(node.id, 0.0))})
        )

    # Drop edges that touch removed orphans
    kept_edges: List[GraphEdge] = []
    for edge in state.get("edges") or []:
        if edge.source in dropped or edge.target in dropped:
            continue
        if edge.source not in graph or edge.target not in graph:
            continue
        kept_edges.append(edge)

    state["nodes"] = kept_nodes
    state["edges"] = kept_edges
    state["toc"] = toc
    return state


def run_phase3_networkx(state: GraphDistillerState | None = None) -> GraphDistillerState:
    """
    Default GraphCompute body (NetworkX).

    Prefer ``run_phase3`` / ``NetworkXGraphCompute`` at call sites (W1-B).
    """
    out: GraphDistillerState = state if state is not None else empty_state()
    errors = list(out.get("errors") or [])
    nodes = list(out.get("nodes") or [])
    edges = list(out.get("edges") or [])
    chunks = list(out.get("chunks") or [])

    settings = get_settings()
    top_k = settings.poc_top_chapters
    min_grounding_chunks = settings.min_chapter_grounding_chunks

    if not nodes:
        errors.append("[WARN] Phase 3 skipped: no nodes in state.")
        out["errors"] = errors
        out["toc"] = {}
        out["phase3_stats"] = _phase3_stats(
            extracted_nodes=0,
            nodes_after_orphan=0,
            edges_after_orphan=0,
            orphans_dropped=0,
            communities_detected=0,
            communities_kept=0,
            thin_dropped=0,
            poc_top_chapters=top_k,
            algorithm="",
        )
        return out

    graph = build_nx_graph(nodes, edges)
    extracted_nodes = graph.number_of_nodes()
    graph, dropped = drop_orphan_nodes(graph, errors)

    if graph.number_of_nodes() == 0:
        errors.append("[WARN] Phase 3: graph empty after orphan removal.")
        out["errors"] = errors
        out["nodes"] = []
        out["edges"] = []
        out["toc"] = {}
        out["phase3_stats"] = _phase3_stats(
            extracted_nodes=extracted_nodes,
            nodes_after_orphan=0,
            edges_after_orphan=0,
            orphans_dropped=len(dropped),
            communities_detected=0,
            communities_kept=0,
            thin_dropped=0,
            poc_top_chapters=top_k,
            algorithm="",
        )
        return out

    detected_toc, algorithm = detect_communities(graph)
    communities_detected = len(detected_toc)
    grounded_toc = drop_thin_communities(
        detected_toc, nodes, chunks, min_grounding_chunks, errors
    )
    thin_dropped = communities_detected - len(grounded_toc)
    toc = select_top_communities(
        grounded_toc,
        top_k=top_k,
        errors=errors,
    )
    scores = compute_pagerank(graph)
    out["errors"] = errors
    out["phase3_stats"] = _phase3_stats(
        extracted_nodes=extracted_nodes,
        nodes_after_orphan=graph.number_of_nodes(),
        edges_after_orphan=graph.number_of_edges(),
        orphans_dropped=len(dropped),
        communities_detected=communities_detected,
        communities_kept=len(toc),
        thin_dropped=thin_dropped,
        poc_top_chapters=top_k,
        algorithm=algorithm,
    )
    return apply_phase3_to_state(out, graph, toc, scores, dropped)


def run_phase3(
    state: GraphDistillerState | None = None,
    *,
    compute: GraphCompute | None = None,
) -> GraphDistillerState:
    """
    Phase 3 entrypoint — delegates to GraphCompute (W1-B).

    Default Adapter = NetworkXGraphCompute. Inject another Adapter only with
    an explicit product decision (Neo4j etc. still out of scope).
    """
    from graph_distiller.adapters.graph_compute import default_graph_compute

    out: GraphDistillerState = state if state is not None else empty_state()
    engine = compute if compute is not None else default_graph_compute()
    return engine.compute(out)


def _phase3_stats(
    *,
    extracted_nodes: int,
    nodes_after_orphan: int,
    edges_after_orphan: int,
    orphans_dropped: int,
    communities_detected: int,
    communities_kept: int,
    thin_dropped: int = 0,
    poc_top_chapters: int,
    algorithm: str,
) -> Phase3Stats:
    stats = empty_phase3_stats()
    stats.update(
        {
            "extracted_nodes": extracted_nodes,
            "nodes_after_orphan": nodes_after_orphan,
            "edges_after_orphan": edges_after_orphan,
            "orphans_dropped": orphans_dropped,
            "communities_detected": communities_detected,
            "communities_kept": communities_kept,
            "thin_dropped": thin_dropped,
            "poc_dropped": max(
                0, communities_detected - thin_dropped - communities_kept
            ),
            "poc_top_chapters": poc_top_chapters,
            "algorithm": algorithm,
        }
    )
    return stats
