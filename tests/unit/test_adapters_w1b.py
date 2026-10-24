"""Unit: W1-B default Adapters + Protocol injection ($0)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pytest

from graph_distiller.adapters import (
    FileSourcePull,
    LangGraphSynthesisOrchestrator,
    NetworkXGraphCompute,
    GraphDistillerGraphStore,
    default_graph_compute,
    default_graph_store,
    default_source_pull,
    default_synthesis_orchestrator,
)
from graph_distiller.graph.build import run_phase3
from graph_distiller.ingestion.pipeline import run_phase1
from graph_distiller.observability import PhaseUsage
from graph_distiller.ports import (
    Embedder,
    GraphCompute,
    GraphStore,
    LlmChat,
    SourcePull,
    SynthesisOrchestrator,
)
from graph_distiller.schema import GraphDistillerState, empty_state

pytestmark = pytest.mark.unit


class _StubSource:
    def __init__(self) -> None:
        self.roots: List[Path] = []

    def load_documents(self, root: Path) -> List[Dict[str, Any]]:
        self.roots.append(root)
        return []


class _StubCompute:
    def __init__(self) -> None:
        self.calls = 0

    def compute(self, state: GraphDistillerState) -> GraphDistillerState:
        self.calls += 1
        out = dict(state)
        out["toc"] = {0: ["injected"]}
        return out  # type: ignore[return-value]


class _StubStore:
    def persist_export(
        self, export_dir: Path, payload: Mapping[str, Any]
    ) -> Mapping[str, str]:
        return {"ok": str(export_dir / "x.json")}

    def load_tent(
        self,
        uri: str,
        export_dir: Path,
        *,
        run_id: str | None = None,
    ) -> Any:
        return {"uri": uri, "run_id": run_id}


def test_default_adapters_satisfy_protocols() -> None:
    assert isinstance(FileSourcePull(), SourcePull)
    assert isinstance(default_source_pull(), SourcePull)
    from graph_distiller.adapters.source import RdbSourcePull, rag_source_pull

    assert isinstance(RdbSourcePull(), SourcePull)
    assert isinstance(rag_source_pull(), SourcePull)
    assert isinstance(NetworkXGraphCompute(), GraphCompute)
    assert isinstance(default_graph_compute(), GraphCompute)
    assert isinstance(GraphDistillerGraphStore(), GraphStore)
    assert isinstance(default_graph_store(), GraphStore)
    assert isinstance(LangGraphSynthesisOrchestrator(), SynthesisOrchestrator)
    assert isinstance(default_synthesis_orchestrator(), SynthesisOrchestrator)


def test_run_phase1_accepts_injected_source() -> None:
    stub = _StubSource()
    out = run_phase1(empty_state(), source=stub, persist_checkpoint=False)
    assert stub.roots  # called with raw_memos_dir
    assert out.get("chunks") == []
    assert out.get("raw_documents") == []


def test_run_phase3_accepts_injected_compute() -> None:
    stub = _StubCompute()
    out = run_phase3(empty_state(), compute=stub)
    assert stub.calls == 1
    assert out.get("toc") == {0: ["injected"]}


def test_openrouter_adapter_classes_are_protocol_shaped() -> None:
    """Structural check without constructing (would hit Settings/API)."""
    from graph_distiller.adapters.openrouter import OpenRouterEmbedder, OpenRouterLlmChat

    assert set(OpenRouterEmbedder.__dict__) >= set()  # class exists
    assert hasattr(OpenRouterEmbedder, "embed_documents")
    assert hasattr(OpenRouterLlmChat, "invoke")
    # Protocols remain importable for typing call sites
    assert Embedder is not None and LlmChat is not None


def test_stub_store_satisfies_graph_store() -> None:
    assert isinstance(_StubStore(), GraphStore)
