"""Unit: W4-A/C SynthesisOrchestrator Protocol + simple-loop default ($0)."""

from __future__ import annotations

from typing import Tuple

import pytest

from graph_distiller.agents.workflow import (
    LangGraphSynthesisOrchestrator,
    SimpleLoopSynthesisOrchestrator,
    default_synthesis_orchestrator,
    run_phase4,
)
from graph_distiller.observability import PhaseUsage
from graph_distiller.ports import SynthesisOrchestrator
from graph_distiller.schema import GraphDistillerState, empty_state

pytestmark = pytest.mark.unit


class _StubOrchestrator:
    """Minimal Adapter: marks a sentinel on the state (no LangGraph / LLM)."""

    def __init__(self) -> None:
        self.calls = 0

    def run(self, state: GraphDistillerState) -> Tuple[GraphDistillerState, PhaseUsage]:
        self.calls += 1
        out = dict(state)
        out["errors"] = list(state.get("errors") or []) + ["stub-orchestrator"]
        return out, PhaseUsage(total_tokens=0, total_cost_usd=0.0)  # type: ignore[return-value]


def test_default_orchestrator_is_simple_loop() -> None:
    orch = default_synthesis_orchestrator()
    assert isinstance(orch, SimpleLoopSynthesisOrchestrator)
    assert isinstance(orch, SynthesisOrchestrator)
    assert isinstance(LangGraphSynthesisOrchestrator(), SynthesisOrchestrator)


def test_run_phase4_empty_toc_no_llm() -> None:
    state = empty_state()
    out, usage = run_phase4(state)
    assert out is state or out == state
    assert usage.total_tokens == 0
    assert usage.total_cost_usd == 0.0


def test_run_phase4_accepts_injected_orchestrator() -> None:
    stub = _StubOrchestrator()
    state = empty_state()
    state["toc"] = {0: ["A"]}
    out, usage = run_phase4(state, orchestrator=stub)
    assert stub.calls == 1
    assert "stub-orchestrator" in (out.get("errors") or [])
    assert usage.total_cost_usd == 0.0
