"""Unit: graph_distiller load helpers — $0 (no DB)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_distiller.rag_load import DDL, SCHEMA, _assert_export_dir

pytestmark = pytest.mark.unit


def test_ddl_only_touches_graph_distiller_schema() -> None:
    assert f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}" in DDL
    assert "graph_distiller.nodes" in DDL
    assert "graph_distiller.edges" in DDL
    assert "graph_distiller.chapters" in DDL
    assert "graph_distiller.source_map" in DDL
    assert "graph_distiller.runs" in DDL
    # Must not mutate public labor tables
    assert "labor_law" not in DDL
    assert "labor_qna" not in DDL
    assert "INSERT INTO public" not in DDL.upper().replace(" ", "")


def test_assert_export_dir_requires_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _assert_export_dir(tmp_path)
    for name in ("nodes.json", "edges.json", "toc.json", "meta.json", "source_map.json"):
        (tmp_path / name).write_text("{}" if name != "nodes.json" else "[]", encoding="utf-8")
    # nodes/edges should be arrays for real load; presence check only here
    (tmp_path / "nodes.json").write_text("[]", encoding="utf-8")
    (tmp_path / "edges.json").write_text("[]", encoding="utf-8")
    (tmp_path / "toc.json").write_text("{}", encoding="utf-8")
    (tmp_path / "meta.json").write_text(json.dumps({"llm_mode": "cites_only"}), encoding="utf-8")
    (tmp_path / "source_map.json").write_text("{}", encoding="utf-8")
    assert _assert_export_dir(tmp_path) == tmp_path.resolve()
