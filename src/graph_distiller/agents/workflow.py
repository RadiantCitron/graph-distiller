"""Phase 4 SynthesisOrchestrator — simple loop default (W4-C).

Protocol: `graph_distiller.ports.SynthesisOrchestrator` (ADR-041).
Default Adapter = `SimpleLoopSynthesisOrchestrator` (W4-C).
`LangGraphSynthesisOrchestrator` remains injectable for comparison.
Poison / retry semantics stay in verifier / planner nodes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from langchain_community.callbacks.manager import get_openai_callback
from langgraph.graph import END, StateGraph

from graph_distiller.agents.helpers import pending_chapter_ids
from graph_distiller.agents.planner import planner_node
from graph_distiller.agents.repomap import repomap_node
from graph_distiller.agents.synthesizer import synthesizer_node
from graph_distiller.agents.verifier import verifier_node
from graph_distiller.config import get_settings
from graph_distiller.contracts import TAG_FATAL
from graph_distiller.observability import (
    PhaseUsage,
    make_phase_progress,
    quiet_console_logs,
)
from graph_distiller.ports import SynthesisOrchestrator
from graph_distiller.schema import GraphDistillerState

logger = logging.getLogger(__name__)

_AGENT_STATUS = {
    "planner": "planner - selecting chapter",
    "repomap": "repomap - building context",
    "synthesizer": "synthesizer - writing chapter",
    "verifier": "verifier - checking citations",
}

# Hard stop: chapters × (retries+1) × agents-per-chapter, with headroom.
_MAX_LOOP_STEPS_FACTOR = 32


def _route_after_planner(state: GraphDistillerState) -> Literal["repomap", "end"]:
    if state.get("current_chapter_id") is None:
        return "end"
    return "repomap"


def _route_after_verifier(state: GraphDistillerState) -> Literal["planner", "end"]:
    if not pending_chapter_ids(state):
        return "end"
    return "planner"


def _merge_state(state: GraphDistillerState, patch: Mapping[str, Any] | None) -> GraphDistillerState:
    """Apply a node partial-update dict (LangGraph-compatible merge)."""
    if not patch:
        return state
    out: Dict[str, Any] = dict(state)
    out.update(dict(patch))
    return out  # type: ignore[return-value]


def _completed_chapter_count(state: GraphDistillerState) -> int:
    toc = state.get("toc") or {}
    if not toc:
        return 0
    drafted = {d.chapter_id for d in (state.get("drafts") or [])}
    poisoned = set(state.get("poisoned_chapters") or [])
    return sum(1 for cid in toc if cid in drafted or cid in poisoned)


def _announce_agent(agent_name: str, state: GraphDistillerState, pbar: Any) -> None:
    """Live agent heartbeat (does not shatter the bar)."""
    from graph_distiller.observability import emit_loop_error

    status = _AGENT_STATUS.get(agent_name, agent_name)
    chapter_id = state.get("current_chapter_id")
    title = ""
    if chapter_id is not None:
        title = (state.get("chapter_titles") or {}).get(chapter_id) or ""
    detail = f"chapter={chapter_id}"
    if title:
        detail = f"{detail} title={title!r}"
    line = f"[Phase 4] {status} ({detail})"
    emit_loop_error(line, level=logging.DEBUG)
    logger.debug(line)


def _tick_progress(
    pbar: Any,
    agent_name: str,
    state: GraphDistillerState,
    usage: PhaseUsage,
    completed: int,
) -> int:
    _announce_agent(agent_name, state, pbar)
    status = _AGENT_STATUS.get(agent_name, agent_name)
    pbar.set_postfix_str(f"{status} | {usage.cost_postfix}")
    now_done = _completed_chapter_count(state)
    if now_done > completed:
        pbar.update(now_done - completed)
        return now_done
    return completed


def build_synthesis_graph():
    """LangGraph multi-chapter loop (optional Adapter; W4-C default is simple loop)."""
    graph = StateGraph(GraphDistillerState)
    graph.add_node("planner", planner_node)
    graph.add_node("repomap", repomap_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"repomap": "repomap", "end": END},
    )
    graph.add_edge("repomap", "synthesizer")
    graph.add_edge("synthesizer", "verifier")
    graph.add_conditional_edges(
        "verifier",
        _route_after_verifier,
        {"planner": "planner", "end": END},
    )
    return graph.compile()


class SimpleLoopSynthesisOrchestrator:
    """
    Default SynthesisOrchestrator Adapter (W4-C).

    Same Planner→RepoMap→Synthesizer→Verifier routing as LangGraph, without
    the StateGraph runtime. Poison / retry remain in verifier_node.
    """

    def run(self, state: GraphDistillerState) -> Tuple[GraphDistillerState, PhaseUsage]:
        toc = state.get("toc") or {}
        total = len(toc)
        logger.debug(
            "Phase 4 (simple loop) start: toc_chapters=%s pending=%s",
            total,
            pending_chapter_ids(state),
        )
        if total == 0:
            return state, PhaseUsage()

        max_steps = max(_MAX_LOOP_STEPS_FACTOR, total * _MAX_LOOP_STEPS_FACTOR)
        pbar = make_phase_progress(total, desc="Phase 4", unit="chapter")
        completed = _completed_chapter_count(state)
        if completed:
            pbar.update(completed)

        final_state: GraphDistillerState = state
        usage = PhaseUsage()
        steps = 0

        with quiet_console_logs():
            with get_openai_callback() as cb:
                try:
                    while steps < max_steps:
                        steps += 1
                        final_state = _merge_state(
                            final_state, planner_node(final_state)
                        )
                        usage = PhaseUsage.from_callback(
                            cb, model=get_settings().synthesis_model
                        )
                        completed = _tick_progress(
                            pbar, "planner", final_state, usage, completed
                        )
                        if _route_after_planner(final_state) == "end":
                            break

                        final_state = _merge_state(
                            final_state, repomap_node(final_state)
                        )
                        usage = PhaseUsage.from_callback(
                            cb, model=get_settings().synthesis_model
                        )
                        completed = _tick_progress(
                            pbar, "repomap", final_state, usage, completed
                        )

                        final_state = _merge_state(
                            final_state, synthesizer_node(final_state)
                        )
                        usage = PhaseUsage.from_callback(
                            cb, model=get_settings().synthesis_model
                        )
                        completed = _tick_progress(
                            pbar, "synthesizer", final_state, usage, completed
                        )

                        final_state = _merge_state(
                            final_state, verifier_node(final_state)
                        )
                        usage = PhaseUsage.from_callback(
                            cb, model=get_settings().synthesis_model
                        )
                        completed = _tick_progress(
                            pbar, "verifier", final_state, usage, completed
                        )
                        if _route_after_verifier(final_state) == "end":
                            break
                    else:
                        msg = (
                            f"{TAG_FATAL} Phase 4 simple loop exceeded max_steps="
                            f"{max_steps} (possible infinite retry)"
                        )
                        errors = list(final_state.get("errors") or [])
                        errors.append(msg)
                        final_state = _merge_state(final_state, {"errors": errors})
                        logger.error(msg)
                finally:
                    usage = PhaseUsage.from_callback(
                        cb, model=get_settings().synthesis_model
                    )
                    pbar.set_postfix_str(usage.cost_postfix)
                    pbar.close()

        logger.debug(
            "Phase 4 (simple loop) done: drafts=%s poisoned=%s tokens=%s cost=$%s",
            len(final_state.get("drafts") or []),
            final_state.get("poisoned_chapters") or [],
            usage.total_tokens,
            usage.total_cost_usd,
        )
        return final_state, usage


class LangGraphSynthesisOrchestrator:
    """
    Optional SynthesisOrchestrator Adapter (pre-W4-C default).

    Kept for injection / A-B; poison/retry still live in verifier/planner.
    """

    def run(self, state: GraphDistillerState) -> Tuple[GraphDistillerState, PhaseUsage]:
        app = build_synthesis_graph()
        toc = state.get("toc") or {}
        total = len(toc)

        logger.debug(
            "Phase 4 (LangGraph) start: toc_chapters=%s pending=%s",
            total,
            pending_chapter_ids(state),
        )

        if total == 0:
            return state, PhaseUsage()

        pbar = make_phase_progress(total, desc="Phase 4", unit="chapter")
        completed = _completed_chapter_count(state)
        if completed:
            pbar.update(completed)

        final_state: GraphDistillerState = state
        usage = PhaseUsage()
        pending_agents: List[str] = []

        with quiet_console_logs():
            with get_openai_callback() as cb:
                try:
                    for mode, chunk in app.stream(
                        state, stream_mode=["updates", "values"]
                    ):
                        if mode == "updates" and isinstance(chunk, dict):
                            pending_agents = list(chunk.keys())
                            continue

                        if mode != "values":
                            continue

                        final_state = chunk  # type: ignore[assignment]
                        usage = PhaseUsage.from_callback(
                            cb, model=get_settings().synthesis_model
                        )

                        for agent_name in pending_agents:
                            completed = _tick_progress(
                                pbar, agent_name, final_state, usage, completed
                            )
                        pending_agents = []
                finally:
                    usage = PhaseUsage.from_callback(
                        cb, model=get_settings().synthesis_model
                    )
                    pbar.set_postfix_str(usage.cost_postfix)
                    pbar.close()

        logger.debug(
            "Phase 4 (LangGraph) done: drafts=%s poisoned=%s tokens=%s cost=$%s",
            len(final_state.get("drafts") or []),
            final_state.get("poisoned_chapters") or [],
            usage.total_tokens,
            usage.total_cost_usd,
        )
        return final_state, usage


def default_synthesis_orchestrator() -> SynthesisOrchestrator:
    """W4-C default = simple loop Adapter."""
    return SimpleLoopSynthesisOrchestrator()


def run_phase4(
    state: GraphDistillerState,
    *,
    orchestrator: Optional[SynthesisOrchestrator] = None,
) -> Tuple[GraphDistillerState, PhaseUsage]:
    """
    Phase 4 entry: delegates to SynthesisOrchestrator.

    Default Adapter is SimpleLoop (W4-C). Inject LangGraphSynthesisOrchestrator
    (or another Adapter) explicitly; poison/retry must stay equivalent.
    """
    orch = orchestrator or default_synthesis_orchestrator()
    return orch.run(state)
