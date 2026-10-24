"""Integration: Phase4 routing with mocked LLM ($0) — simple loop default (W4-C)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph_distiller.agents.helpers import pending_chapter_ids
from graph_distiller.agents.workflow import LangGraphSynthesisOrchestrator, run_phase4
from graph_distiller.config import Settings
from graph_distiller.schema import (
    ChapterTitleResult,
    ChunkMeta,
    GraphEdge,
    GraphNode,
    empty_state,
)

pytestmark = pytest.mark.integration


class _FakeStructured:
    def invoke(self, _messages):  # noqa: ANN001
        return ChapterTitleResult(title="Mocked Chapter Title")


class _FakeLLM:
    def with_structured_output(self, _schema):  # noqa: ANN001
        return _FakeStructured()

    def invoke(self, _messages):  # noqa: ANN001
        return SimpleNamespace(
            content=(
                "# Mocked Chapter Title\n\n"
                "Auth connects to billing via OAuth. [DOC_ID: memo1]\n"
            )
        )


class _BadCiteLLM(_FakeLLM):
    def invoke(self, _messages):  # noqa: ANN001
        return SimpleNamespace(
            content="# Bad\n\nNo citation here, so verifier fails.\n"
        )


class _LearnsFromFeedbackLLM(_FakeLLM):
    """
    ADR-057 조치2 e2e: cites a hallucinated DOC_ID until it sees the retry
    feedback block (proving the Verifier's failure reason actually reaches
    the next Synthesizer attempt), then writes a valid citation.
    """

    def invoke(self, messages):  # noqa: ANN001
        user_prompt = messages[1].content
        if "이전 시도 재작성 지시" in user_prompt:
            return SimpleNamespace(
                content="# Fixed\n\nAuth connects to billing. [DOC_ID: memo1]\n"
            )
        return SimpleNamespace(
            content="# Bad\n\nAuth uses a fake source. [DOC_ID: NOT_A_REAL_ID]\n"
        )


@pytest.fixture
def phase4_seed_state():
    state = empty_state()
    state["chunks"] = [
        ChunkMeta(
            chunk_id="c1",
            source_doc_id="memo1",
            content="OAuth Auth Billing",
            token_count=8,
        )
    ]
    state["nodes"] = [
        GraphNode(
            id="AUTH",
            type="SYSTEM",
            description="Auth system",
            source_chunk_ids=["c1"],
            pagerank_score=0.5,
        ),
        GraphNode(
            id="BILLING",
            type="SYSTEM",
            description="Billing system",
            source_chunk_ids=["c1"],
            pagerank_score=0.3,
        ),
    ]
    state["edges"] = [
        GraphEdge(
            source="AUTH",
            target="BILLING",
            relationship="CONNECTS_TO",
            weight=1.0,
        )
    ]
    state["toc"] = {0: ["AUTH", "BILLING"]}
    return state


def test_run_phase4_planner_to_verifier_with_mocked_llm(
    monkeypatch: pytest.MonkeyPatch,
    phase4_seed_state,
) -> None:
    monkeypatch.setattr(
        Settings,
        "build_chat_model",
        lambda self, *a, **k: _FakeLLM(),
    )

    out, _usage = run_phase4(phase4_seed_state)

    assert 0 in (out.get("chapter_titles") or {})
    assert out["chapter_titles"][0]

    drafts = out.get("drafts") or []
    assert len(drafts) == 1
    draft = drafts[0]
    assert draft.chapter_id == 0
    assert "[DOC_ID: memo1]" in draft.content
    assert "memo1" in draft.cited_doc_ids

    assert not (out.get("poisoned_chapters") or [])
    assert pending_chapter_ids(out) == []


def test_run_phase4_poison_after_max_retries_simple_loop(
    monkeypatch: pytest.MonkeyPatch,
    phase4_seed_state,
) -> None:
    """W4-C: verifier poison/retry must still terminate the loop."""
    monkeypatch.setattr(
        Settings,
        "build_chat_model",
        lambda self, *a, **k: _BadCiteLLM(),
    )
    from dataclasses import replace

    from graph_distiller.config import get_settings, set_runtime_settings

    prev = get_settings()
    set_runtime_settings(replace(prev, max_chapter_retries=2))
    try:
        out, _usage = run_phase4(phase4_seed_state)
        assert 0 in (out.get("poisoned_chapters") or [])
        assert pending_chapter_ids(out) == []
        assert (out.get("retry_counts") or {}).get(0, 0) >= 2
    finally:
        set_runtime_settings(prev)


def test_run_phase4_retry_feedback_recovers_chapter(
    monkeypatch: pytest.MonkeyPatch,
    phase4_seed_state,
) -> None:
    """ADR-057 조치2: 재시도 피드백이 실제로 재시도 프롬프트에 도달해 챕터를 구제한다."""
    monkeypatch.setattr(
        Settings,
        "build_chat_model",
        lambda self, *a, **k: _LearnsFromFeedbackLLM(),
    )
    from dataclasses import replace

    from graph_distiller.config import get_settings, set_runtime_settings

    prev = get_settings()
    set_runtime_settings(replace(prev, max_chapter_retries=3))
    try:
        out, _usage = run_phase4(phase4_seed_state)
        assert not (out.get("poisoned_chapters") or [])
        drafts = out.get("drafts") or []
        assert len(drafts) == 1
        assert "[DOC_ID: memo1]" in drafts[0].content
        assert out["chapter_fail_reasons"].get(0, "").startswith(
            "Unknown DOC_IDs cited"
        )
        assert out["retry_counts"].get(0, 0) >= 1
    finally:
        set_runtime_settings(prev)


def test_run_phase4_langgraph_adapter_still_works(
    monkeypatch: pytest.MonkeyPatch,
    phase4_seed_state,
) -> None:
    monkeypatch.setattr(
        Settings,
        "build_chat_model",
        lambda self, *a, **k: _FakeLLM(),
    )
    out, _usage = run_phase4(
        phase4_seed_state, orchestrator=LangGraphSynthesisOrchestrator()
    )
    assert len(out.get("drafts") or []) == 1
    assert pending_chapter_ids(out) == []
