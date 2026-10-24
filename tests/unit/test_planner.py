"""Unit: Planner title coerce / leak guard — no network / $0."""

from __future__ import annotations

import pytest

from graph_distiller.agents.planner import (
    coerce_chapter_title,
    looks_like_schema_leak,
    planner_node,
)
from graph_distiller.schema import ChapterTitleResult, GraphNode, empty_state

pytestmark = pytest.mark.unit


def test_looks_like_schema_leak_detects_dumped_model() -> None:
    assert looks_like_schema_leak(
        "ChapterTitleResult','description':'부부와 일 아이'"
    )
    assert looks_like_schema_leak("{'title': 'x', 'description': 'y'}")
    assert not looks_like_schema_leak("자기 신뢰와 상담")


def test_coerce_chapter_title_uses_title_field_not_str_dump() -> None:
    fallback = "Auth / Billing"
    assert (
        coerce_chapter_title(ChapterTitleResult(title="신뢰와 상담"), fallback)
        == "신뢰와 상담"
    )
    assert (
        coerce_chapter_title(
            {"title": "신뢰와 상담", "description": "should ignore"},
            fallback,
        )
        == "신뢰와 상담"
    )
    leaked = coerce_chapter_title(
        {"title": "ChapterTitleResult','description':'부부'"},
        fallback,
    )
    assert leaked == fallback
    assert coerce_chapter_title(object(), fallback) == fallback


def test_planner_reuses_cached_title(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def boom(_entities: list[str]) -> str:
        calls["n"] += 1
        raise AssertionError("should not regenerate")

    monkeypatch.setattr("graph_distiller.agents.planner._generate_title", boom)
    state = empty_state()
    state["toc"] = {0: ["AUTH"]}
    state["nodes"] = [
        GraphNode(
            id="AUTH",
            type="SYSTEM",
            description="d",
            source_chunk_ids=["c"],
            pagerank_score=1.0,
        )
    ]
    state["chapter_titles"] = {0: "Cached Good Title"}
    out = planner_node(state)
    assert out["chapter_titles"][0] == "Cached Good Title"
    assert calls["n"] == 0


def test_planner_replaces_leaky_cached_title(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "graph_distiller.agents.planner._generate_title",
        lambda _entities: "Clean Title",
    )
    state = empty_state()
    state["toc"] = {0: ["AUTH"]}
    state["nodes"] = [
        GraphNode(
            id="AUTH",
            type="SYSTEM",
            description="d",
            source_chunk_ids=["c"],
            pagerank_score=1.0,
        )
    ]
    state["chapter_titles"] = {
        0: "ChapterTitleResult','description':'누수'"
    }
    out = planner_node(state)
    assert out["chapter_titles"][0] == "Clean Title"
    assert not looks_like_schema_leak(out["chapter_titles"][0])
