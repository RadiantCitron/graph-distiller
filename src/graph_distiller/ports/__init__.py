"""Port contracts (ADR-041 / W1-A).

Design docs name Protocols only; default Adapters live in `graph_distiller.adapters`
(W1-B). Neo4j / Neptune / LangGraph removal remain out of scope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Protocol, Sequence, Tuple, runtime_checkable

from graph_distiller.observability import PhaseUsage
from graph_distiller.schema import GraphDistillerState

__all__ = [
    "SourcePull",
    "Embedder",
    "LlmChat",
    "GraphCompute",
    "GraphStore",
    "SynthesisOrchestrator",
]


@runtime_checkable
class SourcePull(Protocol):
    """
    COUNT · sample · load source text.

    Invariant: never WRITE the origin (file tree or public.*). Large corpora
    still go through the sample gate (ADR-037 / W3-B).
    """

    def load_documents(self, root: Path) -> List[Dict[str, Any]]:
        """Return memo-shaped dicts (`source_doc_id`, `content`, …)."""
        ...


@runtime_checkable
class Embedder(Protocol):
    """
    Chunk-boundary embeddings (Phase 1).

    Invariant: never log API keys. Product VectorDB is out of scope.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...


@runtime_checkable
class LlmChat(Protocol):
    """
    Structured / chat invoke (Phase 2 extraction, Phase 4 titles & body).

    Invariant: model id comes from Settings — no hardcoded model slugs in Adapters.
    """

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        ...


@runtime_checkable
class GraphCompute(Protocol):
    """
    Phase 3: build / orphan drop / community / PageRank.

    Invariant: edge weight contract; scale guard awareness (W3-A). Neo4j GDS
    Adapter is not built until the in-memory guard repeatedly fails.
    """

    def compute(self, state: GraphDistillerState) -> GraphDistillerState:
        ...


@runtime_checkable
class GraphStore(Protocol):
    """
    Persist / load tent · export artifacts.

    Invariant: WRITE only allowlisted schemas (graph_distiller.*); never public.*.
    """

    def persist_export(
        self, export_dir: Path, payload: Mapping[str, Any]
    ) -> Mapping[str, str]:
        """Write rebuild JSON (or equivalent). Returns logical name → path."""
        ...

    def load_tent(
        self,
        uri: str,
        export_dir: Path,
        *,
        run_id: str | None = None,
    ) -> Any:
        """Load an export directory into the tent store."""
        ...


@runtime_checkable
class SynthesisOrchestrator(Protocol):
    """
    Phase 4 loop: Planner → RepoMap → Synthesizer → Verifier.

    Invariant (W4-A / ADR-041): poison / retry semantics must be preserved by
    every Adapter. Default Adapter = SimpleLoop (W4-C); LangGraph remains injectable.
    """

    def run(self, state: GraphDistillerState) -> Tuple[GraphDistillerState, PhaseUsage]:
        """Drive synthesis until no pending chapters (or all poisoned)."""
        ...
