"""Unit: data/rules/phase2_canonicalize.toml actually changes alias merge decisions, $0."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.graph.extraction import canonicalize_extracted_graph
from graph_distiller.rules import load_rules, set_runtime_rules
from graph_distiller.schema import GraphNode

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_rules():
    import graph_distiller.rules as rules_mod

    rules_mod._runtime_rules = None
    yield
    rules_mod._runtime_rules = None


def _two_similar_nodes() -> list[GraphNode]:
    # "OMNITOME_AI" vs "OMNITOME_ENGINE" share token OMNITOME only -> jaccard 1/3 (~0.33)
    return [
        GraphNode(id="OMNITOME_AI", type="CONCEPT", description="a", source_chunk_ids=["c1"]),
        GraphNode(id="OMNITOME_ENGINE", type="CONCEPT", description="b", source_chunk_ids=["c2"]),
    ]


def test_default_threshold_does_not_merge_loosely_related_ids(tmp_path: Path) -> None:
    set_runtime_rules(load_rules(tmp_path))  # no files -> hardcoded defaults (jaccard 0.8)
    nodes, _ = canonicalize_extracted_graph(_two_similar_nodes(), [])
    assert len(nodes) == 2


def test_lowered_jaccard_threshold_from_toml_merges_them(tmp_path: Path) -> None:
    (tmp_path / "phase2_canonicalize.toml").write_text("jaccard_threshold = 0.3\n", encoding="utf-8")
    set_runtime_rules(load_rules(tmp_path))
    nodes, _ = canonicalize_extracted_graph(_two_similar_nodes(), [])
    assert len(nodes) == 1
