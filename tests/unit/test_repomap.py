"""Unit: RepoMap keep rule, narrative 2-part structure, Lost-in-Middle lock — $0."""

from __future__ import annotations

import pytest

from graph_distiller.agents.repomap import (
    GROUNDING_BUDGET_FRACTION,
    REPOMAP_TOKEN_BUDGET,
    TRUNCATED_MARKER,
    build_repo_map,
    chapter_grounding_budget,
    core_keep_count,
    repomap_node,
    resolve_repomap_token_budget,
)
from graph_distiller.cost import PHASE4_REPOMAP_UPPER_PER_CHAPTER
from graph_distiller.schema import ChunkMeta, GraphNode

pytestmark = pytest.mark.unit


def _node(nid: str, score: float, chunk_ids: list[str]) -> GraphNode:
    return GraphNode(
        id=nid,
        type="CONCEPT",
        description="d",
        source_chunk_ids=chunk_ids,
        pagerank_score=score,
    )


def _chunk(cid: str, doc: str = "doc", content: str = "body") -> ChunkMeta:
    return ChunkMeta(
        chunk_id=cid, source_doc_id=doc, content=content, token_count=1
    )


def test_repomap_hard_ceiling_is_80k() -> None:
    """W4-B: Lost-in-the-Middle — do not raise the ceiling to stuff more."""
    assert REPOMAP_TOKEN_BUDGET == 80_000
    assert resolve_repomap_token_budget(None) == 80_000
    assert resolve_repomap_token_budget(80_000) == 80_000
    assert resolve_repomap_token_budget(10_000) == 10_000
    assert resolve_repomap_token_budget(200_000) == 80_000
    assert resolve_repomap_token_budget(-1) == 0
    # cost.py must stay pure (no repomap import) but share the same physics constant.
    assert PHASE4_REPOMAP_UPPER_PER_CHAPTER == REPOMAP_TOKEN_BUDGET


def test_build_repo_map_clamps_oversized_budget(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        text = build_repo_map(
            0,
            ["A"],
            [_node("A", 1.0, [])],
            [],
            token_budget=500_000,
            small_keep_all_below=10,
        )
    assert "- A (PageRank: 1.0000)" in text
    assert any("clamped" in r.message.lower() for r in caplog.records)


def test_core_keep_count_keeps_all_small_chapters() -> None:
    assert core_keep_count(0, small_keep_all_below=10) == 0
    assert core_keep_count(1, small_keep_all_below=10) == 1
    assert core_keep_count(10, small_keep_all_below=10) == 10
    assert core_keep_count(11, small_keep_all_below=10) == 3  # ceil(11*0.2)


def test_two_part_structure_and_full_chunk_bodies() -> None:
    """ADR-050: Part 1 names only; Part 2 dedup'd full chunk content."""
    nodes = [
        _node("N0", 1.0, ["c1", "c2"]),
        _node("N1", 0.9, ["c2", "c3"]),  # c2 shared -> must appear once
    ]
    chunks = [
        _chunk("c1", content="첫 번째 청크 원문"),
        _chunk("c2", content="공유 청크 원문"),
        _chunk("c3", content="세 번째 청크 원문"),
    ]
    text = build_repo_map(0, ["N0", "N1"], nodes, chunks, small_keep_all_below=10)
    assert "## Part 1: Core Concepts" in text
    assert "## Part 2: Grounding Context" in text
    part1 = text.split("## Part 2")[0]
    assert "DOC_ID" not in part1  # no snippets in Part 1
    assert text.count("공유 청크 원문") == 1  # dedup
    assert "첫 번째 청크 원문" in text and "세 번째 청크 원문" in text
    assert "[DOC_ID: doc]" in text


def test_part2_truncates_at_budget_chunkwise() -> None:
    big = "alpha " * 500  # ~1000 tokens per chunk
    nodes = [_node("N0", 1.0, ["c1", "c2", "c3"])]
    chunks = [_chunk(f"c{i}", content=big) for i in (1, 2, 3)]
    text = build_repo_map(
        0, ["N0"], nodes, chunks, token_budget=800, small_keep_all_below=10
    )
    assert TRUNCATED_MARKER in text
    assert text.count("alpha") > 0
    # budget respected: only as many full chunks as fit
    assert text.count("[DOC_ID: doc]") == 1


def test_part2_orders_chunks_by_summed_pagerank() -> None:
    """ADR-050 v3: Part 2 chunks sorted by sum of citing core nodes' PageRank."""
    nodes = [
        _node("N0", 1.0, ["c1"]),
        _node("N1", 0.9, ["c2"]),
        _node("N2", 0.8, ["c2"]),  # c2 weight 1.7 > c1 weight 1.0
    ]
    chunks = [
        _chunk("c1", content="낮은 가중치 청크"),
        _chunk("c2", content="높은 가중치 청크"),
    ]
    text = build_repo_map(
        0, ["N0", "N1", "N2"], nodes, chunks, small_keep_all_below=10
    )
    part2 = text.split("## Part 2: Grounding Context")[1]
    assert part2.index("높은 가중치 청크") < part2.index("낮은 가중치 청크")


def test_grounding_budget_truncates_part2_only() -> None:
    """grounding_budget caps Part 2 below the 80k ceiling (harness-injected)."""
    big = "alpha " * 500  # ~1000 tokens per chunk
    nodes = [_node("N0", 1.0, ["c1", "c2", "c3"])]
    chunks = [_chunk(f"c{i}", content=big) for i in (1, 2, 3)]
    text = build_repo_map(
        0,
        ["N0"],
        nodes,
        chunks,
        small_keep_all_below=10,
        grounding_budget=700,
    )
    assert TRUNCATED_MARKER in text
    assert text.count("[DOC_ID: doc]") == 1
    # without grounding_budget the same input fits more chunks under 80k
    text_full = build_repo_map(0, ["N0"], nodes, chunks, small_keep_all_below=10)
    assert TRUNCATED_MARKER not in text_full
    assert text_full.count("[DOC_ID: doc]") == 3


def test_tiny_grounding_budget_still_emits_one_part2_doc_id() -> None:
    """ADR-055: do not silence Part 2 when budget < first chunk (Verifier allowlist)."""
    from graph_distiller.agents.repomap import part2_doc_ids

    nodes = [_node("N0", 1.0, ["c1"])]
    chunks = [_chunk("c1", content="OAuth Auth Billing")]
    text = build_repo_map(
        0,
        ["N0"],
        nodes,
        chunks,
        small_keep_all_below=10,
        grounding_budget=2,
    )
    ids = part2_doc_ids(text)
    assert ids == {"doc"}
    assert TRUNCATED_MARKER in text


def test_chapter_grounding_budget_production_path() -> None:
    """ADR-050 v3 production wiring: repomap_node injects raw*0.9 so Part 2
    truncates below the raw chapter size; no chunk bodies -> None fallback."""
    big = "alpha " * 500  # ~1000 tokens per chunk
    nodes = [_node("N0", 1.0, ["c1", "c2", "c3"])]
    chunks = [_chunk(f"c{i}", content=big) for i in (1, 2, 3)]

    budget = chapter_grounding_budget(["N0"], nodes, chunks)
    assert budget is not None
    raw = sum(len(big.split()) for _ in (1, 2, 3))  # rough sanity only
    assert 0 < budget < raw * 4  # well under any raw-token upper bound

    state = {
        "current_chapter_id": 0,
        "toc": {0: ["N0"]},
        "nodes": nodes,
        "chunks": chunks,
    }
    repo_map = repomap_node(state)["repo_map"]
    assert TRUNCATED_MARKER in repo_map
    assert repo_map.count("[DOC_ID: doc]") < 3  # 0.9 * raw < raw

    # no chunk bodies -> None (plain 80k ceiling, Part 2 not silenced)
    empty = [_chunk("c1", content="  ")]
    assert chapter_grounding_budget(["N0"], [_node("N0", 1.0, ["c1"])], empty) is None


def test_build_repo_map_large_chapter_still_top_fraction() -> None:
    nodes = [_node(f"N{i:02d}", float(20 - i), []) for i in range(20)]
    text = build_repo_map(
        0, [n.id for n in nodes], nodes, [], small_keep_all_below=10
    )
    kept = [ln for ln in text.splitlines() if ln.startswith("- N")]
    assert len(kept) == 4  # ceil(20 * 0.20)
    assert "- N00" in text
    assert "- N19" not in text
    assert "(no linked chunks)" in text
