"""GraphCompute default Adapter — in-memory NetworkX Phase 3."""

from __future__ import annotations

from graph_distiller.ports import GraphCompute
from graph_distiller.schema import GraphDistillerState


class NetworkXGraphCompute:
    """
    Default GraphCompute Adapter (NetworkX + community + PageRank).

    Neo4j GDS Adapter is deferred until the W3-A scale guard repeatedly fails.
    """

    def compute(self, state: GraphDistillerState) -> GraphDistillerState:
        from graph_distiller.graph.build import run_phase3_networkx

        return run_phase3_networkx(state)


def default_graph_compute() -> GraphCompute:
    return NetworkXGraphCompute()
