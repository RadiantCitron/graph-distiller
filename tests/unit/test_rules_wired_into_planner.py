"""Unit: data/rules/phase4_planner.toml actually changes planner behavior, $0."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.agents.planner import coerce_chapter_title, looks_like_schema_leak
from graph_distiller.rules import load_rules, set_runtime_rules

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_rules():
    import graph_distiller.rules as rules_mod

    rules_mod._runtime_rules = None
    yield
    rules_mod._runtime_rules = None


def test_custom_leak_marker_from_toml_is_detected() -> None:
    assert not looks_like_schema_leak("정상 제목", leak_markers=("chaptertitleresult",))
    assert looks_like_schema_leak("mycustomleaktoken", leak_markers=("mycustomleaktoken",))


def test_max_title_words_from_toml_truncates_shorter() -> None:
    title = coerce_chapter_title("one two three four five six seven", "fallback", max_words=3)
    assert title == "one two three"


def test_planner_node_uses_toml_leak_markers(tmp_path: Path) -> None:
    from graph_distiller.agents.planner import planner_node
    from graph_distiller.schema import GraphNode, empty_state

    (tmp_path / "phase4_planner.toml").write_text('leak_markers = ["bannedword"]\n', encoding="utf-8")
    set_runtime_rules(load_rules(tmp_path))

    state = empty_state()
    state["toc"] = {0: ["A"]}
    state["nodes"] = [GraphNode(id="A", type="CONCEPT", description="d", source_chunk_ids=["c"])]
    state["chapter_titles"] = {0: "This has bannedword in it"}

    out = planner_node(state)
    assert out["chapter_titles"][0] != "This has bannedword in it"
