"""Unit: Synthesizer Part 1 / Part 2 citation constraint ($0)."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from graph_distiller.agents.synthesizer import SYNTH_SYSTEM, synthesizer_node
from graph_distiller.config import Settings
from graph_distiller.schema import empty_state

pytestmark = pytest.mark.unit


def test_synth_system_forbids_part1_node_names_as_doc_id() -> None:
    text = SYNTH_SYSTEM.lower()
    assert "part 1" in text
    assert "core concepts" in text
    assert "never" in text
    assert "doc_id" in text or "doc_ids" in text


def test_synth_system_requires_part2_grounding_ids() -> None:
    text = SYNTH_SYSTEM.lower()
    assert "part 2" in text
    assert "grounding context" in text
    assert "verbatim" in text or "exactly" in text


def test_synth_user_prompt_wedge_is_in_source() -> None:
    src = inspect.getsource(synthesizer_node)
    assert "Part 1(Core Concepts)" in src or "Part 1 (Core Concepts)" in src
    assert "Part 2(Grounding Context)" in src or "Part 2 (Grounding Context)" in src
    assert "노드 이름" in src


class _CapturingLLM:
    def __init__(self) -> None:
        self.last_messages = None

    def invoke(self, messages):  # noqa: ANN001
        self.last_messages = messages
        return SimpleNamespace(content="# T\n\nbody [DOC_ID: memo1]\n")


def test_synthesizer_node_injects_prior_fail_reason_into_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-057 조치2: 재시도 시 이전 Verifier 실패 사유가 프롬프트에 들어가야 한다."""
    fake = _CapturingLLM()
    monkeypatch.setattr(Settings, "build_chat_model", lambda self, *a, **k: fake)

    state = empty_state()
    state["current_chapter_id"] = 0
    state["chapter_titles"] = {0: "T"}
    state["repo_map"] = "[Chapter: 0]\n## Part 1: Core Concepts\n"
    state["chapter_fail_reasons"] = {0: "Unknown DOC_IDs cited: ['PROGRAMMER']"}

    synthesizer_node(state)

    assert fake.last_messages is not None
    user_prompt = fake.last_messages[1].content
    assert "이전 시도 재작성 지시" in user_prompt
    assert "Unknown DOC_IDs cited: ['PROGRAMMER']" in user_prompt


def test_synthesizer_node_no_feedback_block_on_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _CapturingLLM()
    monkeypatch.setattr(Settings, "build_chat_model", lambda self, *a, **k: fake)

    state = empty_state()
    state["current_chapter_id"] = 0
    state["chapter_titles"] = {0: "T"}
    state["repo_map"] = "[Chapter: 0]\n## Part 1: Core Concepts\n"
    # chapter_fail_reasons left empty (first attempt).

    synthesizer_node(state)

    user_prompt = fake.last_messages[1].content
    assert "이전 시도 재작성 지시" not in user_prompt
