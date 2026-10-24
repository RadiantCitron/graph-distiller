"""Unit: offline RAG A/B probe — $0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_distiller.rag_ab_probe import run_ab_probe
from graph_distiller.schema import ChunkMeta, GraphNode

pytestmark = pytest.mark.unit


def _write_mini_export(tmp: Path) -> Path:
    nodes = [
        GraphNode(
            id="LAW_JO_10",
            type="CONCEPT",
            description="제10조",
            source_chunk_ids=["a"],
            pagerank_score=0.5,
        ),
        GraphNode(
            id="LAW_JO_11",
            type="CONCEPT",
            description="제11조",
            source_chunk_ids=["b"],
            pagerank_score=0.1,
        ),
        GraphNode(
            id="LAW_JO_99",
            type="CONCEPT",
            description="제99조",
            source_chunk_ids=["c"],
            pagerank_score=0.05,
        ),
        GraphNode(
            id="QNA_1",
            type="CONCEPT",
            description="q",
            source_chunk_ids=["q"],
            pagerank_score=0.2,
        ),
    ]
    chunks = [
        ChunkMeta(
            chunk_id="q1",
            source_doc_id="labor_qna_case_1",
            content="질의: 제10조와 제11조 적용 여부",
            token_count=10,
        ),
    ]
    (tmp / "nodes.json").write_text(
        json.dumps([n.model_dump() for n in nodes], ensure_ascii=False),
        encoding="utf-8",
    )
    (tmp / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False),
        encoding="utf-8",
    )
    return tmp


def test_ab_probe_pagerank_beats_or_ties_random(tmp_path: Path) -> None:
    root = _write_mini_export(tmp_path)
    report = run_ab_probe(root, k=1, random_trials=40, seed=0)
    assert report.measurable_queries == 1
    # top-1 by PR is LAW_JO_10 which is in GT
    assert report.queries[0].graph_hits == 1
    assert report.mean_graph_recall == 0.5  # 1 of 2 GT
    assert report.pass_overall is True
