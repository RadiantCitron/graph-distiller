"""Unit: W2-A graph health metrics — no network / $0."""

from __future__ import annotations

import pytest

from graph_distiller.graph.health import compute_graph_health
from graph_distiller.schema import GraphEdge, GraphNode

pytestmark = pytest.mark.unit


def _n(nid: str) -> GraphNode:
    return GraphNode(id=nid, type="CONCEPT", description="d", source_chunk_ids=["c"])


def test_orphan_ratio_and_avg_weight_by_provenance() -> None:
    nodes = [_n("A"), _n("B"), _n("C"), _n("D")]
    edges = [
        GraphEdge(
            source="A", target="B", relationship="R", weight=1.0, provenance="llm"
        ),
        GraphEdge(
            source="B", target="C", relationship="R", weight=3.0, provenance="llm"
        ),
        GraphEdge(
            source="A",
            target="C",
            relationship="R",
            weight=2.0,
            provenance="cites",
        ),
    ]
    h = compute_graph_health(nodes, edges)
    assert h["total_nodes"] == 4
    assert h["total_edges"] == 3
    # D isolated → orphan 1/4
    assert h["orphan_count"] == 1
    assert h["orphan_ratio"] == pytest.approx(0.25)
    assert h["avg_edge_weight"] == pytest.approx(2.0)
    assert h["avg_edge_weight_by_provenance"]["llm"] == pytest.approx(2.0)
    assert h["avg_edge_weight_by_provenance"]["cites"] == pytest.approx(2.0)
    assert h["edge_count_by_provenance"] == {"cites": 1, "llm": 2}


def test_health_empty_graph() -> None:
    h = compute_graph_health([], [])
    assert h["orphan_ratio"] == 0.0
    assert h["avg_edge_weight"] == 0.0
    assert h["avg_edge_weight_by_provenance"] == {}
