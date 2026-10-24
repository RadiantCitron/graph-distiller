"""Unit: load state / cap toc for synthesize — $0 (no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_distiller.rag_synthesize import cap_toc_chapters, load_state_from_export
from graph_distiller.schema import ChunkMeta, GraphEdge, GraphNode, empty_state

pytestmark = pytest.mark.unit


def test_load_and_cap_toc(tmp_path: Path) -> None:
    nodes = [
        GraphNode(id="A", type="CONCEPT", description="a", source_chunk_ids=["c1"]),
        GraphNode(id="B", type="CONCEPT", description="b", source_chunk_ids=["c1"]),
        GraphNode(id="C", type="CONCEPT", description="c", source_chunk_ids=["c2"]),
    ]
    edges = [GraphEdge(source="A", target="B", relationship="CITES", weight=1.0)]
    chunks = [
        ChunkMeta(chunk_id="c1", source_doc_id="d1", content="x", token_count=1),
        ChunkMeta(chunk_id="c2", source_doc_id="d2", content="y", token_count=1),
    ]
    toc = {"0": ["A", "B"], "1": ["C"], "2": ["A"]}
    (tmp_path / "nodes.json").write_text(
        json.dumps([n.model_dump() for n in nodes]), encoding="utf-8"
    )
    (tmp_path / "edges.json").write_text(
        json.dumps([e.model_dump() for e in edges]), encoding="utf-8"
    )
    (tmp_path / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in chunks]), encoding="utf-8"
    )
    (tmp_path / "toc.json").write_text(json.dumps(toc), encoding="utf-8")

    state = load_state_from_export(tmp_path)
    assert len(state["nodes"]) == 3
    assert len(state["toc"]) == 3
    capped = cap_toc_chapters(state, max_chapters=1)
    assert len(capped["toc"]) == 1
