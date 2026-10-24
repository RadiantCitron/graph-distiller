"""Unit: `--judge` is opt-in — default audit never touches the LLM, $0."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import cli
from graph_distiller.config import Settings
from graph_distiller.graph.checkpoint import save_extraction_checkpoint
from graph_distiller.ingestion.checkpoint import save_chunks_checkpoint
from graph_distiller.schema import ChunkMeta, GraphEdge, GraphNode

pytestmark = pytest.mark.unit


def _make_root_with_checkpoints(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for sub in ("data/prompts", "data/lexicon", "data/checkpoints", "output/codex", "output/runs"):
        (root / sub).mkdir(parents=True)
    (root / "output" / "errors.log").write_text("", encoding="utf-8")

    chunks = [
        ChunkMeta(chunk_id="c1", source_doc_id="doc1", content="A와 B 이야기", token_count=10),
        ChunkMeta(chunk_id="c2", source_doc_id="doc2", content="A와 B가 다시 등장", token_count=10),
    ]
    save_chunks_checkpoint(chunks, root / "data" / "checkpoints")
    nodes = [
        GraphNode(id="A", type="CONCEPT", description="a", source_chunk_ids=["c1", "c2"]),
        GraphNode(id="B", type="CONCEPT", description="b", source_chunk_ids=["c1", "c2"]),
    ]
    edges = [GraphEdge(source="A", target="B", relationship="USES", weight=2.0)]
    save_extraction_checkpoint(nodes, edges, root / "data" / "checkpoints")
    return root


def test_default_audit_has_no_judge_scope(tmp_path: Path) -> None:
    root = _make_root_with_checkpoints(tmp_path)
    report = cli.build_report(root)
    assert not any(m.scope == "judge" for m in report.metrics)


def test_judge_flag_invokes_the_judge_collector_with_a_fake_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harness.judge_schema import EvaluationResult

    class _Structured:
        def invoke(self, messages):  # noqa: ANN001
            return EvaluationResult(compression_score=5, information_retention=6, feedback="ok")

    class _FakeLLM:
        def with_structured_output(self, schema):  # noqa: ANN001
            return _Structured()

    monkeypatch.setattr(Settings, "build_chat_model", lambda self, *a, **k: _FakeLLM())

    root = _make_root_with_checkpoints(tmp_path)
    report = cli.build_report(root, judge=True)
    judge_metrics = {m.name: m.value for m in report.metrics if m.scope == "judge"}
    assert judge_metrics["compression_score"] == 5
    assert judge_metrics["information_retention"] == 6
