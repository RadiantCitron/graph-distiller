"""GraphStore default Adapter — JSON export + graph_distiller.* tent load."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from graph_distiller.ports import GraphStore


class GraphDistillerGraphStore:
    """
    Default GraphStore Adapter.

    persist_export → rag_export.export_skeleton (payload = keyword args).
    load_tent → rag_load.load_export_dir (allowlisted graph_distiller.* only).
    """

    def persist_export(
        self, export_dir: Path, payload: Mapping[str, Any]
    ) -> Mapping[str, str]:
        from graph_distiller.rag_export import export_skeleton

        return export_skeleton(export_dir, **dict(payload))

    def load_tent(
        self,
        uri: str,
        export_dir: Path,
        *,
        run_id: str | None = None,
    ) -> Any:
        from graph_distiller.rag_load import load_export_dir

        return load_export_dir(uri, export_dir, run_id=run_id)


def default_graph_store() -> GraphStore:
    return GraphDistillerGraphStore()
