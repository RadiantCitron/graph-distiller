"""Unit: topology verification metrics — synthetic graphs, $0."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import networkx as nx
import pytest

from graph_distiller.schema import GraphEdge, GraphNode

_SPEC = importlib.util.spec_from_file_location(
    "verify_topology",
    Path(__file__).resolve().parents[1] / "tools" / "verify_topology.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_mod)

hub_concentration = _mod.hub_concentration
edge_weight_skew = _mod.edge_weight_skew
gini = _mod.gini
community_modularity = _mod.community_modularity
compute_topology_report = _mod.compute_topology_report

pytestmark = pytest.mark.unit


def _node(node_id: str) -> GraphNode:
    return GraphNode(
        id=node_id,
        type="CONCEPT",
        description=f"node {node_id}",
        source_chunk_ids=["c1"],
    )


def _edge(source: str, target: str, weight: float = 1.0) -> GraphEdge:
    return GraphEdge(source=source, target=target, relationship="r", weight=weight)


def _star_graph(center: str = "HUB", leaves: int = 10) -> nx.Graph:
    g = nx.Graph()
    for i in range(leaves):
        g.add_edge(center, f"L{i}")
    return g


def test_hub_concentration_star() -> None:
    g = _star_graph(leaves=10)
    share = hub_concentration(g, top_pct=0.05)
    assert share == pytest.approx(10 / 20)


def test_hub_concentration_empty() -> None:
    assert hub_concentration(nx.Graph()) == 0.0


def test_hub_concentration_complete_graph_uniform() -> None:
    g = nx.complete_graph(10)
    share = hub_concentration(g, top_pct=0.1)
    assert share == pytest.approx(0.1)


def test_edge_weight_skew() -> None:
    edges = [_edge("A", "B", 3.0), _edge("B", "C", 1.0), _edge("C", "D", 1.0)]
    skew = edge_weight_skew(edges)
    assert skew["max_weight"] == pytest.approx(3.0)
    assert skew["top1pct_avg_weight"] == pytest.approx(3.0)


def test_edge_weight_skew_empty() -> None:
    assert edge_weight_skew([]) == {"max_weight": 0.0, "top1pct_avg_weight": 0.0}


def test_gini_uniform_is_zero() -> None:
    assert gini([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0)


def test_gini_single_winner_approaches_one() -> None:
    assert gini([1000.0, 1.0, 1.0, 1.0]) > 0.7


def test_gini_empty_and_zero() -> None:
    assert gini([]) == 0.0
    assert gini([0.0, 0.0]) == 0.0


def test_community_modularity_two_cliques_positive() -> None:
    g = nx.Graph()
    for pair in [("A", "B"), ("B", "C"), ("A", "C")]:
        g.add_edge(*pair, weight=1.0)
    for pair in [("D", "E"), ("E", "F"), ("D", "F")]:
        g.add_edge(*pair, weight=1.0)
    toc = {0: ["A", "B", "C"], 1: ["D", "E", "F"]}
    mod = community_modularity(g, toc)
    assert mod is not None
    assert mod > 0.4


def test_community_modularity_empty() -> None:
    assert community_modularity(nx.Graph(), {}) is None


def test_compute_topology_report_pass_fail_fields() -> None:
    nodes = [_node("HUB")] + [_node(f"L{i}") for i in range(20)]
    edges = [_edge("HUB", f"L{i}", 2.0) for i in range(20)]
    report = compute_topology_report(nodes, edges)
    assert report["meta"]["nodes_in"] == 21
    assert report["meta"]["edges_in"] == 20
    assert report["metrics"]["hub_top5_degree_share"]["pass"] is True
    assert report["metrics"]["pagerank_gini"]["value"] >= 0.0
    assert report["metrics"]["modularity"]["target"] == 0.3
    assert report["metrics"]["edge_weight_skew"]["pass"] is None
    assert set(report["targets"]) == {
        "hub_top5_degree_share",
        "modularity",
        "pagerank_gini",
    }


def test_apply_graph_reduces_lexicon_merges_without_embed(tmp_path: Path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(
        '{"HUB": ["HUB_ALIAS"]}',
        encoding="utf-8",
    )
    nodes = [_node("HUB"), _node("HUB_ALIAS")] + [_node(f"L{i}") for i in range(8)]
    edges = [_edge("HUB", f"L{i}") for i in range(4)] + [
        _edge("HUB_ALIAS", f"L{i}") for i in range(4, 8)
    ]
    out_nodes, out_edges, applied = _mod.apply_graph_reduces(
        nodes,
        edges,
        apply_lexicon=True,
        embed_merge=False,
        lexicon_path=path,
    )
    assert applied["lexicon"] is True
    assert applied["embed_merge"] is False
    ids = {n.id for n in out_nodes}
    assert "HUB" in ids
    assert "HUB_ALIAS" not in ids
    assert applied["nodes_after"] == len(out_nodes)
