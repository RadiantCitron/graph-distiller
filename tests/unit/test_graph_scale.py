"""Unit: W3-A graph scale preflight — no network / $0."""

from __future__ import annotations

import pytest

from graph_distiller.graph.scale import (
    estimate_graph_bytes,
    guard_graph_scale,
)
from graph_distiller.schema import GraphEdge, GraphNode

pytestmark = pytest.mark.unit


def _node(nid: str, chunks: list[str] | None = None) -> GraphNode:
    return GraphNode(
        id=nid,
        type="CONCEPT",
        description="d",
        source_chunk_ids=chunks or ["c1"],
    )


def _edge(a: str, b: str, w: float = 1.0) -> GraphEdge:
    return GraphEdge(source=a, target=b, relationship="R", weight=w)


def test_estimate_graph_bytes_grows_with_size() -> None:
    small = estimate_graph_bytes([_node("A")], [])
    big = estimate_graph_bytes(
        [_node(f"N{i}") for i in range(50)],
        [_edge(f"N{i}", f"N{(i + 1) % 50}") for i in range(50)],
    )
    assert big > small


def test_guard_pass_under_limits() -> None:
    nodes = [_node("A"), _node("B")]
    edges = [_edge("A", "B")]
    out_n, out_e, info = guard_graph_scale(
        nodes,
        edges,
        max_nodes=100,
        max_edges=100,
        max_est_mb=100.0,
        on_exceed="abort",
    )
    assert info["action"] == "pass"
    assert info["stop_reason"] == "none"
    assert len(out_n) == 2
    assert len(out_e) == 1
    assert info["nodes_dropped"] == 0


def test_guard_abort_on_max_nodes() -> None:
    nodes = [_node(f"N{i}") for i in range(10)]
    edges = [_edge(f"N{i}", f"N{(i + 1) % 10}") for i in range(10)]
    out_n, out_e, info = guard_graph_scale(
        nodes,
        edges,
        max_nodes=3,
        max_edges=1000,
        max_est_mb=100.0,
        on_exceed="abort",
    )
    assert info["action"] == "abort"
    assert info["stop_reason"] == "max_nodes"
    assert len(out_n) == 10
    assert len(out_e) == 10


def test_guard_slice_keeps_hub_nodes() -> None:
    # Hub H connected to leaves; leaves isolated from each other.
    nodes = [_node("H", ["c1", "c2"])] + [_node(f"L{i}") for i in range(5)]
    edges = [_edge("H", f"L{i}", w=2.0) for i in range(5)]
    out_n, out_e, info = guard_graph_scale(
        nodes,
        edges,
        max_nodes=3,
        max_edges=100,
        max_est_mb=100.0,
        on_exceed="slice",
    )
    assert info["action"] == "slice"
    assert info["nodes_after"] <= 3
    assert info["nodes_dropped"] >= 3
    ids = {n.id for n in out_n}
    assert "H" in ids
    for e in out_e:
        assert e.source in ids and e.target in ids


def test_guard_slice_respects_max_edges() -> None:
    nodes = [_node(f"N{i}") for i in range(6)]
    edges = [
        _edge(f"N{i}", f"N{j}", w=float(i + j))
        for i in range(6)
        for j in range(i + 1, 6)
    ]
    out_n, out_e, info = guard_graph_scale(
        nodes,
        edges,
        max_nodes=100,
        max_edges=3,
        max_est_mb=100.0,
        on_exceed="slice",
    )
    assert info["action"] == "slice"
    assert info["stop_reason"] == "max_edges"
    assert len(out_e) <= 3
    assert len(out_n) == 6
