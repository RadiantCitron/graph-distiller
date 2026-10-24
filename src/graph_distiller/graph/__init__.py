"""Phase 2 (extraction) + Phase 3 (hierarchy / PageRank)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graph_distiller.schema import GraphDistillerState

__all__ = ["run_phase2", "run_phase3"]


def run_phase2(*args, **kwargs):
    from graph_distiller.graph.extraction import run_phase2 as _run

    return _run(*args, **kwargs)


def run_phase3(*args, **kwargs):
    from graph_distiller.graph.build import run_phase3 as _run

    return _run(*args, **kwargs)
