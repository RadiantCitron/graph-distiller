"""Unit: repomap audit collector — recomputed via the real Phase 3 function, $0."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.graph.checkpoint import save_extraction_checkpoint
from graph_distiller.ingestion.checkpoint import save_chunks_checkpoint
from graph_distiller.schema import ChunkMeta, GraphEdge, GraphNode

from harness.collectors import repomap as repomap_collector

pytestmark = pytest.mark.unit


def _metric(collected, name):
    hits = [m for m in collected.metrics if m.name == name]
    assert len(hits) == 1, f"{name}: {hits}"
    return hits[0].value


def test_repomap_missing_checkpoints_is_warn_not_crash(tmp_path: Path) -> None:
    out = repomap_collector.collect(tmp_path)
    assert out.metrics == []
    assert any(c.check_id == "repomap.input_present" and not c.passed for c in out.checks)


def _write_small_corpus(tmp_path: Path) -> None:
    chunks = [
        ChunkMeta(chunk_id="c1", source_doc_id="doc1", content="A와 B와 C에 대한 이야기.", token_count=10),
        ChunkMeta(chunk_id="c2", source_doc_id="doc2", content="A와 B와 C가 다시 등장.", token_count=10),
    ]
    save_chunks_checkpoint(chunks, tmp_path)
    nodes = [
        GraphNode(id="A", type="CONCEPT", description="a", source_chunk_ids=["c1", "c2"]),
        GraphNode(id="B", type="CONCEPT", description="b", source_chunk_ids=["c1", "c2"]),
        GraphNode(id="C", type="CONCEPT", description="c", source_chunk_ids=["c1", "c2"]),
    ]
    edges = [
        GraphEdge(source="A", target="B", relationship="USES", weight=3.0),
        GraphEdge(source="B", target="C", relationship="USES", weight=3.0),
        GraphEdge(source="A", target="C", relationship="USES", weight=3.0),
    ]
    save_extraction_checkpoint(nodes, edges, tmp_path)


def test_repomap_recomputes_compression_and_retention(tmp_path: Path) -> None:
    _write_small_corpus(tmp_path)
    out = repomap_collector.collect(tmp_path)

    chapters = _metric(out, "chapters")
    assert chapters >= 1, "single dense triangle should survive as one community"
    assert _metric(out, "phase1_tokens") > 0
    assert _metric(out, "repo_map_tokens") > 0
    assert _metric(out, "core_retention_pct") == 100.0
    assert any(c.check_id == "repomap.token_budget_respected" and c.passed for c in out.checks)
    assert any(c.check_id == "repomap.no_missing_core_nodes" and c.passed for c in out.checks)
