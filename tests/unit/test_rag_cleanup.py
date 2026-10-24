"""Unit: Phase E3 Cleanup — $0 (no DB)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from graph_distiller.rag_cleanup import (
    DELETE_GRAPH_DISTILLER_RUN_SQL,
    DROP_GRAPH_DISTILLER_SCHEMA_SQL,
    cleanup_with_optional_exports,
    safe_remove_export_dir,
)
from graph_distiller.rag_load import SCHEMA
from graph_distiller.rag_write_guard import GraphDistillerWriteGuardError, assert_graph_distiller_write_sql

pytestmark = pytest.mark.unit


def test_drop_and_delete_sql_pass_guard() -> None:
    assert SCHEMA == "graph_distiller"
    assert_graph_distiller_write_sql(DROP_GRAPH_DISTILLER_SCHEMA_SQL)
    assert_graph_distiller_write_sql(DELETE_GRAPH_DISTILLER_RUN_SQL)
    assert "labor_" not in DROP_GRAPH_DISTILLER_SCHEMA_SQL.lower()
    assert "public" not in DROP_GRAPH_DISTILLER_SCHEMA_SQL.lower()
    assert DROP_GRAPH_DISTILLER_SCHEMA_SQL == "DROP SCHEMA IF EXISTS graph_distiller CASCADE"
    assert "graph_distiller.runs" in DELETE_GRAPH_DISTILLER_RUN_SQL


def test_forbidden_cleanup_sql_rejected() -> None:
    with pytest.raises(GraphDistillerWriteGuardError):
        assert_graph_distiller_write_sql("DROP SCHEMA IF EXISTS public CASCADE")
    with pytest.raises(GraphDistillerWriteGuardError):
        assert_graph_distiller_write_sql("DELETE FROM labor_law")


def test_safe_remove_export_dir_only_under_root(tmp_path: Path) -> None:
    root = tmp_path / "rag_rebuild"
    good = root / "rag_x"
    good.mkdir(parents=True)
    (good / "meta.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    assert safe_remove_export_dir(good, rebuild_root=root) == str(good.resolve())
    assert not good.exists()
    assert safe_remove_export_dir(outside, rebuild_root=root) is None
    assert outside.exists()


def test_cleanup_requires_exclusive_action() -> None:
    with pytest.raises(ValueError, match="either"):
        cleanup_with_optional_exports("uri", drop_schema=True, run_id="x")
    with pytest.raises(ValueError, match="required"):
        cleanup_with_optional_exports("uri")


def test_drop_schema_calls_guarded_sql() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch("graph_distiller.rag_cleanup.connect", return_value=conn):
        with patch("graph_distiller.rag_cleanup.apply_write_search_path") as sp:
            with patch("graph_distiller.rag_cleanup.execute_graph_distiller") as ex:
                from graph_distiller.rag_cleanup import drop_graph_distiller_schema

                result = drop_graph_distiller_schema("postgresql://example/db")
    assert result.action == "drop_schema"
    assert result.schema == "graph_distiller"
    sp.assert_called()
    ex.assert_called()
    sql = ex.call_args[0][1]
    assert sql == DROP_GRAPH_DISTILLER_SCHEMA_SQL
