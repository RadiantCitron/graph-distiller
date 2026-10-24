"""Unit: data/rules/phase4_repomap.toml actually changes RepoMap output, $0."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.agents.repomap import (
    build_repo_map,
    chapter_grounding_budget,
    core_keep_count,
    resolve_repomap_token_budget,
)
from graph_distiller.rules import load_rules, set_runtime_rules
from graph_distiller.schema import ChunkMeta, GraphNode

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_rules():
    import graph_distiller.rules as rules_mod

    rules_mod._runtime_rules = None
    yield
    rules_mod._runtime_rules = None


def _nodes(n: int) -> list[GraphNode]:
    return [
        GraphNode(id=f"N{i}", type="CONCEPT", description="d", source_chunk_ids=["c"], pagerank_score=1.0 - i * 0.01)
        for i in range(n)
    ]


def test_top_fraction_from_toml_changes_core_keep_count(tmp_path: Path) -> None:
    (tmp_path / "phase4_repomap.toml").write_text("top_fraction = 0.5\n", encoding="utf-8")
    set_runtime_rules(load_rules(tmp_path))
    assert core_keep_count(20, small_keep_all_below=0) == 10  # 50% of 20


def test_token_budget_from_toml_lowers_the_ceiling(tmp_path: Path) -> None:
    (tmp_path / "phase4_repomap.toml").write_text("token_budget = 5000\n", encoding="utf-8")
    set_runtime_rules(load_rules(tmp_path))
    assert resolve_repomap_token_budget(None) == 5000
    assert resolve_repomap_token_budget(10_000) == 10_000  # explicit request still wins


def test_grounding_budget_fraction_from_toml(tmp_path: Path) -> None:
    (tmp_path / "phase4_repomap.toml").write_text("grounding_budget_fraction = 0.5\n", encoding="utf-8")
    set_runtime_rules(load_rules(tmp_path))
    chunks = [ChunkMeta(chunk_id="c1", source_doc_id="d1", content="x" * 40, token_count=10)]
    nodes = [GraphNode(id="A", type="CONCEPT", description="a", source_chunk_ids=["c1"])]
    budget_half = chapter_grounding_budget(["A"], nodes, chunks)

    set_runtime_rules(load_rules(tmp_path.parent / "no_such_dir"))
    budget_default = chapter_grounding_budget(["A"], nodes, chunks)
    assert budget_half < budget_default


def test_lowered_token_budget_actually_truncates_build_repo_map(tmp_path: Path) -> None:
    nodes = _nodes(3)
    chunks = [ChunkMeta(chunk_id="c", source_doc_id="doc", content="본문 " * 2000, token_count=2000)]
    for n in nodes:
        n.source_chunk_ids.append("c")

    set_runtime_rules(load_rules(tmp_path))  # defaults: 80k ceiling
    full = build_repo_map(0, [n.id for n in nodes], nodes, chunks)

    (tmp_path / "phase4_repomap.toml").write_text("token_budget = 1000\n", encoding="utf-8")
    set_runtime_rules(load_rules(tmp_path))
    truncated = build_repo_map(0, [n.id for n in nodes], nodes, chunks)
    assert "[TRUNCATED" in truncated
    assert len(truncated) < len(full)
