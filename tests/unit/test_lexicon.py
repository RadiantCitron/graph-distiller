"""Unit: W2-D domain alias lexicon — no network / $0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_distiller.graph.extraction import canonicalize_extracted_graph
from graph_distiller.graph.lexicon import (
    lexicon_prompt_hint,
    load_entity_alias_lexicon,
)
from graph_distiller.schema import GraphEdge, GraphNode

pytestmark = pytest.mark.unit


def _node(nid: str, chunks: list[str] | None = None) -> GraphNode:
    return GraphNode(
        id=nid,
        type="CONCEPT",
        description="d",
        source_chunk_ids=chunks or ["c1"],
    )


def test_load_entity_alias_lexicon_normalizes(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps({"Auth Server": ["auth-server", "인증 서버"]}),
        encoding="utf-8",
    )
    mapping = load_entity_alias_lexicon(path)
    assert mapping["AUTH_SERVER"] == "AUTH_SERVER"
    assert mapping["AUTH-SERVER"] == "AUTH_SERVER"
    assert mapping["인증_서버"] == "AUTH_SERVER"


def test_load_missing_or_empty_is_noop(tmp_path: Path) -> None:
    assert load_entity_alias_lexicon(None) == {}
    assert load_entity_alias_lexicon(tmp_path / "nope.json") == {}
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    assert load_entity_alias_lexicon(empty) == {}


def test_canonicalize_with_lexicon_merges_aliases(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(
        json.dumps({"KOREA": ["대한민국", "한국"]}),
        encoding="utf-8",
    )
    lex = load_entity_alias_lexicon(path)
    nodes, edges = canonicalize_extracted_graph(
        [_node("대한민국", ["c1"]), _node("한국", ["c2"]), _node("FREEDOM")],
        [
            GraphEdge(
                source="대한민국",
                target="FREEDOM",
                relationship="R",
                weight=1.0,
            ),
            GraphEdge(
                source="한국",
                target="FREEDOM",
                relationship="R",
                weight=1.0,
            ),
        ],
        alias_lexicon=lex,
    )
    assert {n.id for n in nodes} == {"KOREA", "FREEDOM"}
    kept = next(n for n in nodes if n.id == "KOREA")
    assert kept.source_chunk_ids == ["c1", "c2"]
    assert len(edges) == 1
    assert edges[0].weight == 2.0
    assert edges[0].provenance == "alias-merge"


def test_empty_lexicon_preserves_spelling_only_golden() -> None:
    nodes, edges = canonicalize_extracted_graph(
        [_node("NA_JASIN", ["c1"]), _node("NAJASIN", ["c2"])],
        [
            GraphEdge(source="NA_JASIN", target="TRUST", relationship="HAS"),
            GraphEdge(source="NAJASIN", target="TRUST", relationship="HAS"),
        ],
        alias_lexicon={},
    )
    assert "NA_JASIN" in {n.id for n in nodes}
    assert "NAJASIN" not in {n.id for n in nodes}


def test_lexicon_prompt_hint_lists_aliases() -> None:
    hint = lexicon_prompt_hint(
        {"KOREA": "KOREA", "한국": "KOREA", "AUTH": "AUTH"}
    )
    assert "Preferred canonical node ids" in hint
    assert "KOREA" in hint
    assert "한국" in hint
    assert lexicon_prompt_hint({}) == ""
