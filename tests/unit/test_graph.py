"""Unit: graph helpers — no network / $0."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest

from graph_distiller.graph.build import (
    _pagerank_python,
    absorb_small_communities,
    drop_orphan_nodes,
    drop_thin_communities,
    run_phase3,
)
from graph_distiller.schema import ChunkMeta, GraphEdge, GraphNode, empty_state
from graph_distiller.graph.extraction import _normalize_node_id

pytestmark = pytest.mark.unit


def test_normalize_node_id_upper_and_underscores() -> None:
    assert _normalize_node_id("  foo bar  ") == "FOO_BAR"
    assert _normalize_node_id("Already_OK") == "ALREADY_OK"
    assert _normalize_node_id("multi   space") == "MULTI_SPACE"


def test_drop_orphan_nodes_removes_degree_zero() -> None:
    g = nx.Graph()
    g.add_edge("A", "B")
    g.add_node("ORPHAN")
    errors: list[str] = []
    g2, orphans = drop_orphan_nodes(g, errors)
    assert orphans == {"ORPHAN"}
    assert "ORPHAN" not in g2
    assert set(g2.nodes()) == {"A", "B"}
    assert any("Orphan Node Dropped: ORPHAN" in e for e in errors)


def test_pagerank_python_sums_and_hub_bias() -> None:
    g = nx.Graph()
    g.add_edge("HUB", "A", weight=1.0)
    g.add_edge("HUB", "B", weight=1.0)
    g.add_edge("HUB", "C", weight=1.0)
    scores = _pagerank_python(g, weight="weight")
    assert set(scores) == {"HUB", "A", "B", "C"}
    total = sum(scores.values())
    assert abs(total - 1.0) < 1e-5
    assert scores["HUB"] >= scores["A"]
    assert scores["HUB"] >= scores["B"]


def _concept(name: str) -> GraphNode:
    return GraphNode(
        id=name,
        type="CONCEPT",
        description="x",
        source_chunk_ids=["c"],
    )


def test_run_phase3_stats_when_no_nodes() -> None:
    out = run_phase3(empty_state())
    stats = out["phase3_stats"]
    assert stats["extracted_nodes"] == 0
    assert stats["communities_detected"] == 0
    assert stats["communities_kept"] == 0
    assert stats["poc_dropped"] == 0


def test_select_top_communities_zero_keeps_all() -> None:
    from graph_distiller.graph.build import select_top_communities

    toc = {0: ["A", "B"], 1: ["C"]}
    out = select_top_communities(toc, top_k=0)
    assert out == {0: ["A", "B"], 1: ["C"]}


def test_run_phase3_stats_detected_vs_poc_cap() -> None:
    from graph_distiller.config import get_settings, set_runtime_settings

    prev = get_settings()
    set_runtime_settings(replace(prev, poc_top_chapters=1))
    try:
        state = empty_state()
        state["nodes"] = [_concept("A"), _concept("B"), _concept("C"), _concept("D")]
        state["edges"] = [
            GraphEdge(source="A", target="B", relationship="r"),
            GraphEdge(source="C", target="D", relationship="r"),
        ]
        out = run_phase3(state)
        stats = out["phase3_stats"]
        assert stats["extracted_nodes"] == 4
        assert stats["orphans_dropped"] == 0
        assert stats["communities_detected"] == 2
        assert stats["communities_kept"] == 1
        assert stats["poc_dropped"] == 1
        assert stats["poc_top_chapters"] == 1
        assert len(out["toc"]) == 1
        assert stats["algorithm"] in {"leiden", "louvain"}
    finally:
        set_runtime_settings(prev)


def test_absorb_small_communities_joins_neighbor_not_disconnected() -> None:
    g = nx.Graph()
    g.add_edge("A", "B", weight=1.0)
    g.add_edge("B", "C", weight=1.0)
    g.add_edge("C", "D", weight=1.0)
    g.add_edge("X", "Y", weight=1.0)
    merged = absorb_small_communities(
        g,
        [["A"], ["B", "C", "D"], ["X", "Y"]],
        min_size=3,
    )
    sizes = sorted(len(c) for c in merged)
    assert sizes == [2, 4]
    big = next(c for c in merged if len(c) == 4)
    assert set(big) == {"A", "B", "C", "D"}
    small = next(c for c in merged if len(c) == 2)
    assert set(small) == {"X", "Y"}


def test_run_phase3_min_size_does_not_glue_disconnected_pairs() -> None:
    from graph_distiller.config import get_settings, set_runtime_settings

    prev = get_settings()
    set_runtime_settings(
        replace(
            prev,
            poc_top_chapters=10,
            min_community_size=8,
            louvain_resolution=0.5,
        )
    )
    try:
        state = empty_state()
        state["nodes"] = [_concept("A"), _concept("B"), _concept("C"), _concept("D")]
        state["edges"] = [
            GraphEdge(source="A", target="B", relationship="r"),
            GraphEdge(source="C", target="D", relationship="r"),
        ]
        out = run_phase3(state)
        assert out["phase3_stats"]["communities_detected"] == 2
        assert len(out["toc"]) == 2
    finally:
        set_runtime_settings(prev)


def _chunk(chunk_id: str, content: str = "some grounding text") -> ChunkMeta:
    return ChunkMeta(
        chunk_id=chunk_id,
        source_doc_id="doc",
        content=content,
        token_count=len(content.split()),
    )


def test_drop_thin_communities_excludes_single_chunk_chapters() -> None:
    """ADR-057: a chapter backed by only 1 unique source chunk is excluded."""
    toc = {0: ["A", "B"], 1: ["C", "D"]}
    nodes = [
        GraphNode(id="A", type="CONCEPT", description="x", source_chunk_ids=["c1"]),
        GraphNode(id="B", type="CONCEPT", description="x", source_chunk_ids=["c1"]),
        GraphNode(id="C", type="CONCEPT", description="x", source_chunk_ids=["c2"]),
        GraphNode(id="D", type="CONCEPT", description="x", source_chunk_ids=["c3"]),
    ]
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
    errors: list[str] = []

    out = drop_thin_communities(toc, nodes, chunks, min_chunks=2, errors=errors)

    # Chapter 0 (A,B) shares only c1 (1 unique chunk) -> dropped.
    # Chapter 1 (C,D) spans c2+c3 (2 unique chunks) -> kept, reindexed to 0.
    assert len(out) == 1
    assert set(out[0]) == {"C", "D"}
    assert any("MinGrounding" in e and "old_id=0" in e for e in errors)


def test_drop_thin_communities_disabled_below_two_or_without_chunks() -> None:
    toc = {0: ["A", "B"]}
    nodes = [
        GraphNode(id="A", type="CONCEPT", description="x", source_chunk_ids=["c1"]),
        GraphNode(id="B", type="CONCEPT", description="x", source_chunk_ids=["c1"]),
    ]
    chunks = [_chunk("c1")]

    # min_chunks<=1 disables the filter outright.
    assert drop_thin_communities(toc, nodes, chunks, min_chunks=1, errors=[]) == {
        0: ["A", "B"]
    }
    # No chunk data available at all -> fail open (don't drop).
    assert drop_thin_communities(toc, nodes, [], min_chunks=2, errors=[]) == {
        0: ["A", "B"]
    }


def test_run_phase3_drops_thin_chapter_end_to_end() -> None:
    """Two disconnected pairs, one grounded by 2 chunks, one by only 1."""
    from graph_distiller.config import get_settings, set_runtime_settings

    prev = get_settings()
    set_runtime_settings(
        replace(
            prev,
            poc_top_chapters=0,
            min_community_size=1,
            min_chapter_grounding_chunks=2,
        )
    )
    try:
        state = empty_state()
        state["nodes"] = [
            GraphNode(id="A", type="CONCEPT", description="x", source_chunk_ids=["c1"]),
            GraphNode(id="B", type="CONCEPT", description="x", source_chunk_ids=["c1"]),
            GraphNode(id="C", type="CONCEPT", description="x", source_chunk_ids=["c2"]),
            GraphNode(id="D", type="CONCEPT", description="x", source_chunk_ids=["c3"]),
        ]
        state["edges"] = [
            GraphEdge(source="A", target="B", relationship="r"),
            GraphEdge(source="C", target="D", relationship="r"),
        ]
        state["chunks"] = [_chunk("c1"), _chunk("c2"), _chunk("c3")]
        out = run_phase3(state)
        stats = out["phase3_stats"]
        assert stats["communities_detected"] == 2
        assert stats["thin_dropped"] == 1
        assert stats["communities_kept"] == 1
        assert len(out["toc"]) == 1
        assert set(next(iter(out["toc"].values()))) == {"C", "D"}
    finally:
        set_runtime_settings(prev)
