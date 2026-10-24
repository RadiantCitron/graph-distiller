"""Unit: W2-C after harness — mock embed merge, no network / $0."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from graph_distiller.schema import GraphEdge, GraphNode

_SPEC = importlib.util.spec_from_file_location(
    "embed_merge_after",
    Path(__file__).resolve().parents[1] / "tools" / "embed_merge_after.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_mod)

apply_embed_merge_after = _mod.apply_embed_merge_after
MockEmbedder = _mod.MockEmbedder

pytestmark = pytest.mark.unit


class _SynonymEmbedder:
    """Maps known synonym texts to the same vector so Reduce 2 merges them."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            key = text.strip().lower()
            if "korea" in key or "한국" in key or "대한민국" in key:
                out.append([1.0, 0.0, 0.0])
            else:
                out.append([0.0, 1.0, 0.0])
        return out


def test_apply_embed_merge_after_merges_synonyms_and_reports_delta() -> None:
    nodes = [
        GraphNode(
            id="KOREA",
            type="CONCEPT",
            description="Korea",
            source_chunk_ids=["c1"],
        ),
        GraphNode(
            id="한국",
            type="CONCEPT",
            description="Korea",
            source_chunk_ids=["c2"],
        ),
        GraphNode(
            id="OTHER",
            type="CONCEPT",
            description="Other",
            source_chunk_ids=["c3"],
        ),
    ]
    edges = [
        GraphEdge(source="KOREA", target="OTHER", relationship="R", weight=1.0),
        GraphEdge(source="한국", target="OTHER", relationship="R", weight=1.0),
    ]
    after_nodes, after_edges, cmp = apply_embed_merge_after(
        nodes,
        edges,
        embeddings=_SynonymEmbedder(),
        sim_threshold=0.9,
        max_embed_nodes=100,
    )
    assert len(after_nodes) == 2  # Korea synonyms merged
    assert cmp["before"]["skeleton"]["total_nodes"] == 3
    assert cmp["after"]["skeleton"]["total_nodes"] == 2
    assert cmp["delta"]["skeleton"]["total_nodes"] == -1
    assert cmp["product_default_canon_embed_merge"] is False
    # Collapsed parallel edges should raise weight
    assert any(e.weight >= 2.0 for e in after_edges)


def test_mock_embedder_is_deterministic() -> None:
    emb = MockEmbedder()
    a = emb.embed_documents(["Hello World"])
    b = emb.embed_documents(["Hello World"])
    assert a == b
