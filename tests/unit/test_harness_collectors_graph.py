"""Unit: chunks/graph audit collectors reuse product checkpoint IO + tools/verify_*, $0."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.graph.checkpoint import save_extraction_checkpoint
from graph_distiller.ingestion.checkpoint import save_chunks_checkpoint
from graph_distiller.schema import ChunkMeta, GraphEdge, GraphNode

from harness.collectors import chunks as chunks_collector
from harness.collectors import graph as graph_collector

pytestmark = pytest.mark.unit


def _metric(collected, name):
    hits = [m for m in collected.metrics if m.name == name]
    assert len(hits) == 1, f"{name}: {hits}"
    return hits[0].value


def _check(collected, check_id):
    hits = [c for c in collected.checks if c.check_id == check_id]
    assert len(hits) == 1, f"{check_id}: {hits}"
    return hits[0]


def test_chunks_missing_checkpoint_is_warn_not_crash(tmp_path: Path) -> None:
    out = chunks_collector.collect(tmp_path)
    assert _check(out, "chunks.input_present").passed is False
    assert out.metrics == []


def test_chunks_collect_flags_oversized_and_duplicate_ids(tmp_path: Path) -> None:
    chunks = [
        ChunkMeta(chunk_id="a", source_doc_id="doc1", content="hello", token_count=10),
        ChunkMeta(chunk_id="a", source_doc_id="doc1", content="", token_count=1200),
    ]
    save_chunks_checkpoint(chunks, tmp_path)
    out = chunks_collector.collect(tmp_path)

    assert _metric(out, "chunks") == 2
    assert _metric(out, "source_docs") == 1
    assert _metric(out, "empty_content") == 1
    assert _metric(out, "oversized_over_1000_tokens") == 1
    assert _check(out, "chunks.no_oversized_chunks").passed is False
    assert _check(out, "chunks.unique_chunk_ids").passed is False


def test_chunks_collect_clean_checkpoint_passes_checks(tmp_path: Path) -> None:
    chunks = [ChunkMeta(chunk_id=f"c{i}", source_doc_id="doc1", content="x" * 5, token_count=5) for i in range(3)]
    save_chunks_checkpoint(chunks, tmp_path)
    out = chunks_collector.collect(tmp_path)
    assert _check(out, "chunks.no_oversized_chunks").passed is True
    assert _check(out, "chunks.unique_chunk_ids").passed is True


def _sample_graph() -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes = [
        GraphNode(id="A", type="CONCEPT", description="a", source_chunk_ids=["c1"]),
        GraphNode(id="B", type="CONCEPT", description="b", source_chunk_ids=["c1"]),
        GraphNode(id="C", type="CONCEPT", description="c", source_chunk_ids=["c2"]),
        GraphNode(id="ORPHAN", type="CONCEPT", description="o", source_chunk_ids=["c2"]),
    ]
    edges = [
        GraphEdge(source="A", target="B", relationship="USES", weight=3.0),
        GraphEdge(source="B", target="C", relationship="RELATES_TO", weight=1.0),
    ]
    return nodes, edges


def test_graph_missing_checkpoint_is_warn_not_crash(tmp_path: Path) -> None:
    out = graph_collector.collect(tmp_path)
    assert _check(out, "graph.input_present").passed is False
    assert out.metrics == []


def test_graph_collect_reuses_skeleton_and_topology_functions(tmp_path: Path) -> None:
    nodes, edges = _sample_graph()
    save_extraction_checkpoint(nodes, edges, tmp_path)
    out = graph_collector.collect(tmp_path)

    assert _metric(out, "skeleton.total_nodes") == 4
    assert _metric(out, "skeleton.total_edges") == 2
    assert _metric(out, "topology.meta.orphans_dropped") == 1
    assert _check(out, "graph.giant_component_present").passed is True
    assert any(c.check_id == "graph.topology.hub_top5_degree_share" for c in out.checks)
