"""Integration: Phase1-shaped chunks -> Phase2 with mocked LLM ($0)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from graph_distiller.config import settings
from graph_distiller.graph.extraction import GraphExtractor, run_phase2
from graph_distiller.observability import PhaseUsage
from graph_distiller.schema import ChunkMeta, ExtractionResult, GraphEdge, GraphNode, empty_state

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_chunks() -> list[ChunkMeta]:
    return [
        ChunkMeta(
            chunk_id="c1",
            source_doc_id="memo1",
            content="OAuth connects Auth Service to Billing.",
            token_count=10,
        )
    ]


def test_phase2_consumes_chunks_with_mocked_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_chunks: list[ChunkMeta],
) -> None:
    """Chunks in Phase2 input shape become normalized graph state (no network)."""
    key_field = "_".join(("openrouter", "api", "key"))
    test_settings = replace(
        settings,
        checkpoint_dir=tmp_path,
        **{key_field: "local-test-only"},
    )

    async def fake_extract_all(self, chunks, errors):  # noqa: ANN001
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "c1"
        assert chunks[0].source_doc_id == "memo1"
        from graph_distiller.graph.extraction import _merge_nodes

        raw_nodes = [
            GraphNode(
                id="auth service",
                type="SYSTEM",
                description="Auth",
                source_chunk_ids=["c1"],
            ),
            GraphNode(
                id="billing",
                type="SYSTEM",
                description="Billing",
                source_chunk_ids=["c1"],
            ),
        ]
        node_map: dict = {}
        _merge_nodes(node_map, raw_nodes)
        edges = [
            GraphEdge(
                source="AUTH_SERVICE",
                target="BILLING",
                relationship="CONNECTS_TO",
                weight=1.0,
            )
        ]
        return list(node_map.values()), edges, PhaseUsage()

    monkeypatch.setattr(GraphExtractor, "extract_all", fake_extract_all)

    state = empty_state()
    state["chunks"] = sample_chunks
    out, usage = run_phase2(
        state, settings=test_settings, persist_checkpoint=False
    )

    ids = {n.id for n in out["nodes"]}
    assert "AUTH_SERVICE" in ids
    assert "BILLING" in ids
    assert len(out["edges"]) == 1
    assert out["edges"][0].source == "AUTH_SERVICE"
    assert out["edges"][0].target == "BILLING"
    assert usage.total_tokens == 0


def test_phase2_chunk_fields_match_extraction_contract(
    sample_chunks: list[ChunkMeta],
) -> None:
    """Phase2 input contract: ChunkMeta has id, source_doc_id, content."""
    c = sample_chunks[0]
    assert c.chunk_id
    assert c.source_doc_id
    assert c.content
    sample = ExtractionResult(
        nodes=[
            GraphNode(
                id="X",
                type="CONCEPT",
                description="d",
                source_chunk_ids=[c.chunk_id],
            )
        ],
        edges=[],
    )
    assert sample.nodes[0].source_chunk_ids == [c.chunk_id]
