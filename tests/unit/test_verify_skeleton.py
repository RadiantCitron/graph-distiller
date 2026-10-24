"""Unit: W2-V skeleton stats — no network / $0."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from graph_distiller.schema import GraphEdge, GraphNode

_SPEC = importlib.util.spec_from_file_location(
    "verify_skeleton",
    Path(__file__).resolve().parents[1] / "tools" / "verify_skeleton.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_mod)
compute_skeleton_stats = _mod.compute_skeleton_stats

pytestmark = pytest.mark.unit


def _n(nid: str) -> GraphNode:
    return GraphNode(id=nid, type="CONCEPT", description="d", source_chunk_ids=["c"])


def test_skeleton_stats_on_fixed_graph() -> None:
    nodes = [_n("A"), _n("B"), _n("C"), _n("D")]
    edges = [
        GraphEdge(source="A", target="B", relationship="R", weight=1.0),
        GraphEdge(source="B", target="C", relationship="R", weight=2.5),
        GraphEdge(source="A", target="C", relationship="R", weight=3.0),
    ]
    stats = compute_skeleton_stats(nodes, edges)
    assert stats["total_nodes"] == 4
    assert stats["total_edges"] == 3
    assert stats["max_edge_weight"] == 3.0
    assert stats["edges_weight_ge_2_count"] == 2
    assert stats["edges_weight_ge_2_pct"] == pytest.approx(200.0 / 3.0)
    # A-B-C connected; D isolated → giant 3/4
    assert stats["giant_component_nodes"] == 3
    assert stats["giant_component_pct"] == pytest.approx(75.0)


def test_skeleton_stats_empty() -> None:
    stats = compute_skeleton_stats([], [])
    assert stats["total_nodes"] == 0
    assert stats["total_edges"] == 0
    assert stats["max_edge_weight"] == 0.0
    assert stats["edges_weight_ge_2_pct"] == 0.0
    assert stats["giant_component_pct"] == 0.0
