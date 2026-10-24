"""Unit: W1-A Protocol 6 contracts — structural only, no network / $0."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import pytest

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

_PORT_NAMES = (
    "SourcePull",
    "Embedder",
    "LlmChat",
    "GraphCompute",
    "GraphStore",
    "SynthesisOrchestrator",
)


class _StubSourcePull:
    def load_documents(self, root: Path) -> List[Dict[str, Any]]:
        return [{"source_doc_id": "x", "content": "y", "path": str(root)}]


class _StubEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 3 for _ in texts]


class _StubLlmChat:
    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return {"ok": True, "input": input}


class _StubGraphCompute:
    def compute(self, state: GraphDistillerState) -> GraphDistillerState:
        out = dict(state)
        out["toc"] = {0: ["N0"]}
        return out  # type: ignore[return-value]


class _StubGraphStore:
    def persist_export(
        self, export_dir: Path, payload: Mapping[str, Any]
    ) -> Mapping[str, str]:
        export_dir.mkdir(parents=True, exist_ok=True)
        return {"meta": str(export_dir / "meta.json")}

    def load_tent(
        self,
        uri: str,
        export_dir: Path,
        *,
        run_id: str | None = None,
    ) -> Any:
        return {"uri": uri, "dir": str(export_dir), "run_id": run_id}


class _StubOrchestrator:
    def run(self, state: GraphDistillerState) -> Tuple[GraphDistillerState, PhaseUsage]:
        return state, PhaseUsage(total_tokens=0, total_cost_usd=0.0)


def test_all_six_protocols_exported() -> None:
    import graph_distiller.ports as ports

    for name in _PORT_NAMES:
        assert hasattr(ports, name), name
        assert name in ports.__all__


def test_stubs_satisfy_runtime_checkable_protocols() -> None:
    assert isinstance(_StubSourcePull(), SourcePull)
    assert isinstance(_StubEmbedder(), Embedder)
    assert isinstance(_StubLlmChat(), LlmChat)
    assert isinstance(_StubGraphCompute(), GraphCompute)
    assert isinstance(_StubGraphStore(), GraphStore)
    assert isinstance(_StubOrchestrator(), SynthesisOrchestrator)


def test_incomplete_stub_does_not_satisfy_graph_store() -> None:
    class _ExportOnly:
        def persist_export(
            self, export_dir: Path, payload: Mapping[str, Any]
        ) -> Mapping[str, str]:
            return {}

    assert not isinstance(_ExportOnly(), GraphStore)


def test_stub_graph_compute_preserves_state_shape() -> None:
    state = empty_state()
    out = _StubGraphCompute().compute(state)
    assert 0 in (out.get("toc") or {})
