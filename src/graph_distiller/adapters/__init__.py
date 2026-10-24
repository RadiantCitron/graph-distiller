"""Default Adapters for ADR-041 ports (W1-B).

Physical implementations stay where they are; these classes are the named
default Adapters (ADR-002 OpenRouter reinterpreted as default, not exclusive).
Neo4j / Neptune / LangGraph removal are out of scope.
"""

from __future__ import annotations

from graph_distiller.adapters.graph_compute import (
    NetworkXGraphCompute,
    default_graph_compute,
)
from graph_distiller.adapters.graph_store import (
    GraphDistillerGraphStore,
    default_graph_store,
)
from graph_distiller.adapters.openrouter import (
    OpenRouterEmbedder,
    OpenRouterLlmChat,
    default_embedder,
    default_llm_chat,
)
from graph_distiller.adapters.source import (
    FileSourcePull,
    RdbSourcePull,
    default_source_pull,
    rag_source_pull,
)

# Lazy (PEP 562): agents.workflow imports adapters.openrouter (planner), so a
# top-level import here would be a circular-import time bomb when workflow is
# imported first (broke the local pipeline entrypoint after W1-B/W4-C).
_WORKFLOW_NAMES = {
    "LangGraphSynthesisOrchestrator",
    "SimpleLoopSynthesisOrchestrator",
    "default_synthesis_orchestrator",
}


def __getattr__(name: str):
    if name in _WORKFLOW_NAMES:
        from graph_distiller.agents import workflow

        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "FileSourcePull",
    "RdbSourcePull",
    "OpenRouterEmbedder",
    "OpenRouterLlmChat",
    "NetworkXGraphCompute",
    "GraphDistillerGraphStore",
    "SimpleLoopSynthesisOrchestrator",
    "LangGraphSynthesisOrchestrator",
    "default_source_pull",
    "rag_source_pull",
    "default_embedder",
    "default_llm_chat",
    "default_graph_compute",
    "default_graph_store",
    "default_synthesis_orchestrator",
]
