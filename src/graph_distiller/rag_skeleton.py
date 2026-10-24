"""RAG skeleton pipeline — cites-first (+ optional sample Phase2). No Phase 4.

ADR-037: sample only. Full-DB LLM is a failure mode.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from graph_distiller.adapters.graph_store import default_graph_store
from graph_distiller.agents.repomap import REPOMAP_TOKEN_BUDGET, build_repo_map, core_keep_count
from graph_distiller.budget import clamp_budget_usd, slice_chunks_by_budget
from graph_distiller.config import Settings, get_settings, set_runtime_settings, settings as default_settings
from graph_distiller.cost import count_tokens
from graph_distiller.graph.build import run_phase3
from graph_distiller.graph.extraction import canonicalize_extracted_graph, run_phase2
from graph_distiller.ingestion.law_cites import parse_jo_refs
from graph_distiller.ingestion.rdb import (
    EdgeMeta,
    SamplePayload,
    build_cites_graph,
    connect,
    counts_as_dict,
    fetch_counts,
    load_full,
    load_sample,
    require_sample_sizes,
    sample_to_chunks,
)
from graph_distiller.ports import GraphStore
from graph_distiller.rag_export import create_export_dir
from graph_distiller.rag_gates import assert_sample_gate, phase2_llm_default_enabled
from graph_distiller.schema import ChunkMeta, GraphEdge, GraphNode, GraphDistillerState, empty_state

logger = logging.getLogger(__name__)


@dataclass
class SkeletonResult:
    sample: SamplePayload
    chunks: List[ChunkMeta]
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    edges_meta: List[EdgeMeta]
    toc: Dict[int, List[str]]
    phase3_stats: Dict[str, Any]
    llm_mode: str
    export_dir: Optional[Path] = None
    export_paths: Dict[str, str] = field(default_factory=dict)
    phase2_usage: Dict[str, Any] = field(default_factory=dict)
    verify: Dict[str, Any] = field(default_factory=dict)


def _run_phase3_with_settings(
    state: GraphDistillerState,
    settings: Settings,
) -> GraphDistillerState:
    """Phase 3 under explicit Settings (ADR-055: default poc_top=0 needs no hack)."""
    prev = get_settings()
    set_runtime_settings(settings)
    try:
        return run_phase3(state)
    finally:
        set_runtime_settings(prev)


def merge_llm_into_cites(
    cites_nodes: Sequence[GraphNode],
    cites_edges: Sequence[GraphEdge],
    cites_meta: Sequence[EdgeMeta],
    llm_nodes: Sequence[GraphNode],
    llm_edges: Sequence[GraphEdge],
) -> Tuple[List[GraphNode], List[GraphEdge], List[EdgeMeta]]:
    """Union cites + Phase2 LLM graph, then canonicalize aliases/weights."""
    merged_meta = list(cites_meta)
    for e in llm_edges:
        merged_meta.append(
            EdgeMeta(
                source=e.source,
                target=e.target,
                provenance="llm",
                relationship=e.relationship,
                chunk_id="",
                note="phase2",
            )
        )
    nodes, edges = canonicalize_extracted_graph(
        list(cites_nodes) + list(llm_nodes),
        list(cites_edges) + list(llm_edges),
    )
    return nodes, edges, merged_meta


def run_sample_phase2(
    chunks: List[ChunkMeta],
    *,
    settings: Settings | None = None,
    budget_usd: float = 0.25,
    max_chunk_chars: int = 4000,
) -> Tuple[List[GraphNode], List[GraphEdge], Dict[str, Any], List[ChunkMeta]]:
    """Phase 2 on sample chunks only (ADR-037). Never full DB."""
    cfg = settings or get_settings()
    capped = clamp_budget_usd(budget_usd)
    # Truncate long QnA bodies before LLM — prevents reasoning-token blowups on sample.
    trimmed: List[ChunkMeta] = []
    for ch in chunks:
        if len(ch.content) <= max_chunk_chars:
            trimmed.append(ch)
            continue
        cut = ch.content[:max_chunk_chars] + "\n…[truncated for sample Phase2]"
        trimmed.append(
            ChunkMeta(
                chunk_id=ch.chunk_id,
                source_doc_id=ch.source_doc_id,
                content=cut,
                token_count=count_tokens(cut),
            )
        )
    kept, slice_info = slice_chunks_by_budget(
        trimmed,
        cfg.extraction_model,
        budget_usd=capped,
        token_budget=1_000_000,
    )
    state = empty_state()
    state["chunks"] = kept
    state, usage = run_phase2(state, settings=cfg, persist_checkpoint=False)
    usage_dict = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "total_cost_usd": usage.total_cost_usd,
        "chunks_in": len(chunks),
        "chunks_trimmed": len(trimmed),
        "chunks_kept": len(kept),
        "max_chunk_chars": max_chunk_chars,
        "slice": slice_info,
        "budget_usd": capped,
    }
    return list(state["nodes"]), list(state["edges"]), usage_dict, kept


def verify_skeleton(
    sample: SamplePayload,
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    chunks: Sequence[ChunkMeta],
    toc: Mapping[int, List[str]] | None = None,
    *,
    rng_seed: int = 42,
) -> Dict[str, Any]:
    """
    Wiring checks A–D on the sample (ADR-034/037).

    B: explicit 제N조 cites that target a graph node survive as an edge endpoint.
    C: PageRank core retains grounded law nodes better than a random subset.
    D: Token Diet — RepoMap tokens << raw chunk tokens (positive compression).
    A: community count vs distinct 장 in sample (partial; needs jang>=2).
    """
    toc = toc or {}
    node_ids = {n.id for n in nodes}
    by_id = {n.id: n for n in nodes}
    edge_pairs = set()
    for e in edges:
        a, b = (e.source, e.target) if e.source <= e.target else (e.target, e.source)
        edge_pairs.add((a, b))

    gt_links: List[Tuple[str, str]] = []
    for c in sample.qna_cases:
        for ref in parse_jo_refs(c.content):
            gt_links.append((c.node_id, ref.node_id))
    for b in sample.jo_bundles:
        for ref in parse_jo_refs(b.content_merged):
            if ref.node_id != b.node_id:
                gt_links.append((b.node_id, ref.node_id))

    gt_unique = list(dict.fromkeys(gt_links))
    measurable = [
        (s, t) for s, t in gt_unique if s in node_ids and t in node_ids
    ]
    survived = 0
    for s, t in measurable:
        a, b = (s, t) if s <= t else (t, s)
        if (a, b) in edge_pairs:
            survived += 1
    b_score = (survived / len(measurable)) if measurable else None

    grounded_law = {
        b.node_id
        for b in sample.jo_bundles
        if b.node_id in by_id
        and any(
            (min(c.node_id, b.node_id), max(c.node_id, b.node_id)) in edge_pairs
            for c in sample.qna_cases
            if c.node_id in node_ids
        )
    }
    if not grounded_law:
        grounded_law = {b.node_id for b in sample.jo_bundles if b.node_id in by_id}

    ranked = sorted(list(nodes), key=lambda n: n.pagerank_score, reverse=True)
    keep_n = core_keep_count(len(ranked), small_keep_all_below=10)
    core_ids = {n.id for n in ranked[:keep_n]} if ranked else set()
    core_hit = len(core_ids & grounded_law)

    rng = random.Random(rng_seed)
    trials = 30
    random_hits: List[int] = []
    for _ in range(trials):
        if keep_n <= 0 or not ranked:
            random_hits.append(0)
            continue
        pick = set(rng.sample([n.id for n in ranked], min(keep_n, len(ranked))))
        random_hits.append(len(pick & grounded_law))
    random_mean = sum(random_hits) / len(random_hits) if random_hits else 0.0
    c_ok = core_hit >= random_mean if grounded_law else None

    raw_tokens = sum(c.token_count for c in chunks) or sum(
        count_tokens(c.content) for c in chunks
    )
    member_ids = [n.id for n in ranked]
    repo = build_repo_map(
        0,
        member_ids,
        list(nodes),
        list(chunks),
        token_budget=REPOMAP_TOKEN_BUDGET,
        small_keep_all_below=10,
    )
    repo_tokens = count_tokens(repo)
    compression_ratio = (repo_tokens / raw_tokens) if raw_tokens else None
    node_diet_ratio = (keep_n / len(ranked)) if ranked else None
    if len(ranked) <= 10:
        # ADR-025: small chapters keep all nodes — diet is N/A; still report ratio.
        d_ok = True
        d_note = "small_graph_keep_all_policy"
    else:
        # Structural Token Diet: core keep fraction < 1, or token ratio < 1.
        d_ok = bool(
            (node_diet_ratio is not None and node_diet_ratio < 1.0)
            or (
                compression_ratio is not None
                and compression_ratio < 1.0
                and raw_tokens > 0
            )
        )
        d_note = "top20pct_node_diet_or_token_ratio"

    jang_ids = {b.jang_id for b in sample.jo_bundles if b.jang_id is not None}
    a_partial = {
        "distinct_jang_in_sample": len(jang_ids),
        "communities_in_toc": len(toc),
        "measurable": len(jang_ids) >= 2,
        "note": "partial - compare only when sample spans >=2 jang",
    }

    return {
        "B_citation_survival": {
            "gt_links": len(gt_unique),
            "measurable_both_endpoints_in_graph": len(measurable),
            "survived_as_edge": survived,
            "rate": b_score,
            "pass": (b_score is not None and b_score >= 0.5),
        },
        "C_pagerank_vs_random": {
            "grounded_law_nodes": sorted(grounded_law),
            "core_keep_n": keep_n,
            "core_grounded_hits": core_hit,
            "random_mean_hits": random_mean,
            "pass": c_ok,
        },
        "D_token_diet": {
            "raw_chunk_tokens": raw_tokens,
            "repomap_tokens": repo_tokens,
            "compression_ratio": compression_ratio,
            "node_diet_ratio": node_diet_ratio,
            "note": d_note,
            "pass": d_ok,
        },
        "A_community_vs_jang": a_partial,
    }


def run_rag_skeleton(
    *,
    n_jo: int | None = None,
    m_qna: int | None = None,
    rag_db_uri: str,
    settings: Settings | None = None,
    llm: bool = False,
    budget_usd: float = 0.25,
    export: bool = True,
    export_root: Path | None = None,
    run_id: str | None = None,
    graph_store: GraphStore | None = None,
    full_db: bool = False,
) -> SkeletonResult:
    """
    End-to-end skeleton: COUNT → sample|full → cites → (optional Phase2) → Phase3 → export.

    full_db=True is explicit opt-in only (compiler --full-db). Silent omit of
    n_jo/m_qna remains fatal (ADR-037).
    """
    if full_db:
        if llm and phase2_llm_default_enabled():
            raise RuntimeError("Phase 2 LLM default must stay off (ADR-037 / W3-B)")
    else:
        assert_sample_gate(n_jo=n_jo, m_qna=m_qna, llm=llm)
        require_sample_sizes(int(n_jo), int(m_qna))
        if llm and phase2_llm_default_enabled():
            raise RuntimeError("Phase 2 LLM default must stay off (ADR-037 / W3-B)")
    cfg = settings or get_settings()
    conn = connect(rag_db_uri)
    try:
        sample = (
            load_full(conn) if full_db else load_sample(conn, int(n_jo), int(m_qna))
        )
    finally:
        conn.close()

    chunks = sample_to_chunks(sample)
    cites_nodes, cites_edges, edges_meta = build_cites_graph(sample)
    llm_mode = "cites_only"
    phase2_usage: Dict[str, Any] = {}
    nodes = cites_nodes
    edges = cites_edges

    if llm:
        llm_mode = "phase2_sample"
        llm_nodes, llm_edges, phase2_usage, kept = run_sample_phase2(
            chunks, settings=cfg, budget_usd=budget_usd
        )
        nodes, edges, edges_meta = merge_llm_into_cites(
            cites_nodes, cites_edges, edges_meta, llm_nodes, llm_edges
        )
        chunks = kept
        phase2_usage["cites_only_edges"] = len(cites_edges)
        phase2_usage["llm_edges"] = len(llm_edges)
        phase2_usage["merged_edges_before_phase3"] = len(edges)

    state = empty_state()
    state["chunks"] = chunks
    state["nodes"] = nodes
    state["edges"] = edges
    state = _run_phase3_with_settings(state, cfg)
    nodes = list(state["nodes"])
    edges = list(state["edges"])
    toc = dict(state.get("toc") or {})
    stats = dict(state.get("phase3_stats") or {})
    verify = verify_skeleton(sample, nodes, edges, chunks, toc)

    export_dir = None
    export_paths: Dict[str, str] = {}
    if export:
        export_dir = create_export_dir(export_root, run_id=run_id)
        store = graph_store if graph_store is not None else default_graph_store()
        export_paths = dict(
            store.persist_export(
                export_dir,
                {
                    "sample": sample,
                    "nodes": nodes,
                    "edges": edges,
                    "edges_meta": edges_meta,
                    "toc": toc,
                    "chunks": chunks,
                    "phase3_stats": stats,
                    "llm_mode": llm_mode,
                    "extra_meta": {
                        "verify": verify,
                        "phase2_usage": phase2_usage,
                    },
                },
            )
        )

    return SkeletonResult(
        sample=sample,
        chunks=list(chunks),
        nodes=nodes,
        edges=edges,
        edges_meta=edges_meta,
        toc=toc,
        phase3_stats=stats,
        llm_mode=llm_mode,
        export_dir=export_dir,
        export_paths=export_paths,
        phase2_usage=phase2_usage,
        verify=verify,
    )


def preview_counts(rag_db_uri: str) -> Dict[str, Any]:
    conn = connect(rag_db_uri)
    try:
        return counts_as_dict(fetch_counts(conn))
    finally:
        conn.close()
