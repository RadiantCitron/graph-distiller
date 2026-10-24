"""Phase 2/3 skeleton + topology metrics.

Reuses the existing pure measurement functions from tests/tools instead of
duplicating graph-theory code: `tools.verify_skeleton.compute_skeleton_stats`
and `tools.verify_topology.compute_topology_report`. Those CLI scripts and
their unit tests (test_verify_skeleton.py, test_verify_topology.py) are
untouched — this collector just calls the same functions and re-shapes the
result dicts into MetricRecord/CheckResult.
"""

from __future__ import annotations

from pathlib import Path

from graph_distiller.graph.checkpoint import load_extraction_checkpoint
from tools.verify_skeleton import compute_skeleton_stats
from tools.verify_topology import compute_topology_report

from harness.collectors import Collected

SCOPE = "graph"


def collect(checkpoint_dir: Path, *, filename: str = "phase2_extraction.json") -> Collected:
    out = Collected()
    path = Path(checkpoint_dir) / filename
    if not out.require_input(SCOPE, path):
        return out

    nodes, edges = load_extraction_checkpoint(checkpoint_dir, filename=filename)
    skeleton = compute_skeleton_stats(nodes, edges)
    for key in (
        "total_nodes",
        "total_edges",
        "max_edge_weight",
        "edges_weight_ge_2_count",
        "edges_weight_ge_2_pct",
        "giant_component_nodes",
        "giant_component_pct",
    ):
        out.metric(SCOPE, f"skeleton.{key}", skeleton[key])

    topo = compute_topology_report(nodes, edges)
    meta = topo["meta"]
    for key, value in meta.items():
        out.metric(SCOPE, f"topology.meta.{key}", value)
    for name, m in topo["metrics"].items():
        value = m["value"]
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                out.metric(SCOPE, f"topology.{name}.{sub_key}", sub_value)
            continue
        out.metric(SCOPE, f"topology.{name}", value, threshold=m.get("target"))
        if m.get("pass") is not None:
            out.check(
                f"graph.topology.{name}",
                SCOPE,
                bool(m["pass"]),
                severity="warn",
                message=f"{name}={value} target>={m.get('target')}",
            )

    out.check("graph.giant_component_present", SCOPE, skeleton["giant_component_nodes"] > 0, severity="warn",
              message=f"giant_component_nodes={skeleton['giant_component_nodes']}")
    return out
