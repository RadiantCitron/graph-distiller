"""Unit: Phase 2 alias canonicalize — no network / $0."""

from __future__ import annotations

import pytest

from graph_distiller.graph.extraction import (
    _are_alias_ids,
    canonicalize_extracted_graph,
)
from graph_distiller.schema import GraphEdge, GraphNode

pytestmark = pytest.mark.unit


def _node(nid: str, chunks: list[str], desc: str = "d") -> GraphNode:
    return GraphNode(
        id=nid,
        type="CONCEPT",
        description=desc,
        source_chunk_ids=chunks,
    )


def test_are_alias_ids_fingerprint_and_underscores() -> None:
    assert _are_alias_ids("NA_JASIN", "NAJASIN")
    assert _are_alias_ids("FOO_BAR", "FOO-BAR")
    assert not _are_alias_ids("KOREA", "KOREA_AND_FRANCE")
    assert not _are_alias_ids("SELF", "MYSELF")


def test_canonicalize_merges_spelling_variants_and_chunk_ids() -> None:
    nodes, edges = canonicalize_extracted_graph(
        [
            _node("NA_JASIN", ["c1"], "inner self"),
            _node("NAJASIN", ["c2"], "self"),
        ],
        [
            GraphEdge(source="NA_JASIN", target="TRUST", relationship="HAS"),
            GraphEdge(source="NAJASIN", target="TRUST", relationship="HAS"),
        ],
    )
    ids = {n.id for n in nodes}
    assert "NA_JASIN" in ids
    assert "NAJASIN" not in ids
    kept = next(n for n in nodes if n.id == "NA_JASIN")
    assert kept.source_chunk_ids == ["c1", "c2"]


def test_canonicalize_sums_weights_and_drops_self_loop() -> None:
    nodes, edges = canonicalize_extracted_graph(
        [
            _node("TRUST", ["c1"]),
            _node("NA_JASIN", ["c1"]),
            _node("NAJASIN", ["c2"]),
        ],
        [
            GraphEdge(
                source="NA_JASIN",
                target="TRUST",
                relationship="HAS",
                weight=1.0,
            ),
            GraphEdge(
                source="TRUST",
                target="NAJASIN",
                relationship="HAS",
                weight=1.0,
            ),
            GraphEdge(
                source="NA_JASIN",
                target="NAJASIN",
                relationship="SAME",
                weight=1.0,
            ),
        ],
    )
    assert {n.id for n in nodes} == {"TRUST", "NA_JASIN"}
    assert len(edges) == 1
    assert {edges[0].source, edges[0].target} == {"NA_JASIN", "TRUST"}
    assert edges[0].weight == 2.0
    assert edges[0].provenance == "alias-merge"


def test_canonicalize_does_not_merge_distinct_concepts() -> None:
    nodes, edges = canonicalize_extracted_graph(
        [
            _node("KOREA", ["c1"]),
            _node("KOREA_AND_FRANCE", ["c2"]),
            _node("FREEDOM", ["c2"]),
        ],
        [
            GraphEdge(source="KOREA", target="FREEDOM", relationship="R"),
            GraphEdge(
                source="KOREA_AND_FRANCE",
                target="FREEDOM",
                relationship="R",
            ),
        ],
    )
    assert {n.id for n in nodes} == {"KOREA", "KOREA_AND_FRANCE", "FREEDOM"}
    assert len(edges) == 2


class _MockEmbed:
    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        out: list[list[float]] = []
        for text in texts:
            key = text.split()[0]
            out.append(self.table[key])
        return out


class _BoomEmbed:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("network down")


def test_embed_merge_joins_korean_and_roman_synonyms() -> None:
    vec = [1.0, 0.0, 0.0]
    other = [0.0, 1.0, 0.0]
    mock = _MockEmbed(
        {
            "AUTH_SERVER": vec,
            "인증_서버": vec,
            "FREEDOM": other,
        }
    )
    nodes, edges = canonicalize_extracted_graph(
        [
            _node("AUTH_SERVER", ["c1"], "login gateway"),
            _node("인증_서버", ["c2"], "the auth system"),
            _node("FREEDOM", ["c2"], "liberty"),
        ],
        [
            GraphEdge(source="AUTH_SERVER", target="FREEDOM", relationship="R", weight=1.0),
            GraphEdge(source="인증_서버", target="FREEDOM", relationship="R", weight=1.0),
        ],
        embed_merge=True,
        embeddings=mock,
        sim_threshold=0.99,
    )
    ids = {n.id for n in nodes}
    assert "FREEDOM" in ids
    assert len(ids) == 2
    assert len(edges) == 1
    assert edges[0].weight == 2.0
    assert edges[0].provenance == "alias-merge"
    assert mock.calls == 1


def test_embed_merge_respects_type_and_subset_guards() -> None:
    same = [1.0, 0.0]
    mock = _MockEmbed(
        {
            "KOREA": same,
            "KOREA_AND_FRANCE": same,
            "ALICE": same,
        }
    )
    nodes, _edges = canonicalize_extracted_graph(
        [
            _node("KOREA", ["c1"]),
            _node("KOREA_AND_FRANCE", ["c2"]),
            GraphNode(
                id="ALICE",
                type="PERSON",
                description="a person",
                source_chunk_ids=["c3"],
            ),
        ],
        [],
        embed_merge=True,
        embeddings=mock,
        sim_threshold=0.0,
    )
    assert {n.id for n in nodes} == {"KOREA", "KOREA_AND_FRANCE", "ALICE"}


def test_embed_merge_falls_back_when_embedder_raises() -> None:
    nodes, _edges = canonicalize_extracted_graph(
        [
            _node("AUTH_SERVER", ["c1"]),
            _node("인증_서버", ["c2"]),
        ],
        [],
        embed_merge=True,
        embeddings=_BoomEmbed(),
        sim_threshold=0.0,
    )
    assert {n.id for n in nodes} == {"AUTH_SERVER", "인증_서버"}


def test_unmerged_edge_keeps_llm_provenance() -> None:
    nodes, edges = canonicalize_extracted_graph(
        [_node("A", ["c1"]), _node("B", ["c1"])],
        [GraphEdge(source="A", target="B", relationship="R", weight=1.0)],
    )
    assert len(edges) == 1
    assert edges[0].provenance == "llm"


def test_cites_provenance_preserved_when_no_alias_collapse() -> None:
    nodes, edges = canonicalize_extracted_graph(
        [_node("JO_1", ["c1"]), _node("JO_2", ["c1"])],
        [
            GraphEdge(
                source="JO_1",
                target="JO_2",
                relationship="CITES",
                weight=1.0,
                provenance="cites",
            )
        ],
    )
    assert edges[0].provenance == "cites"
    nodes, edges = canonicalize_extracted_graph(
        [
            _node("KOREA", ["c1"]),
            _node("KOREA_AND_FRANCE", ["c2"]),
            _node("FREEDOM", ["c2"]),
        ],
        [
            GraphEdge(source="KOREA", target="FREEDOM", relationship="R"),
            GraphEdge(
                source="KOREA_AND_FRANCE",
                target="FREEDOM",
                relationship="R",
            ),
        ],
    )
    assert {n.id for n in nodes} == {"KOREA", "KOREA_AND_FRANCE", "FREEDOM"}
    assert len(edges) == 2
