"""Unit: Phase E2 graph_distiller write guard — $0."""

from __future__ import annotations

import pytest

from graph_distiller.rag_load import DDL
from graph_distiller.rag_write_guard import (
    ALLOWED_SCHEMA,
    GraphDistillerWriteGuardError,
    assert_graph_distiller_write_sql,
    split_sql_statements,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "sql",
    [
        "CREATE SCHEMA IF NOT EXISTS graph_distiller",
        "DROP SCHEMA IF EXISTS graph_distiller CASCADE",
        "DELETE FROM graph_distiller.runs WHERE run_id = %s",
        "INSERT INTO graph_distiller.nodes (run_id, node_id) VALUES (%s, %s)",
        "UPDATE graph_distiller.runs SET export_dir = %s WHERE run_id = %s",
        "CREATE INDEX IF NOT EXISTS graph_distiller_nodes_pagerank_idx ON graph_distiller.nodes (run_id)",
        "SET search_path TO graph_distiller",
        "SET search_path TO graph_distiller, pg_catalog",
        "SET LOCAL search_path TO graph_distiller, pg_catalog",
    ],
)
def test_allow_graph_distiller_writes(sql: str) -> None:
    assert_graph_distiller_write_sql(sql)


def test_load_ddl_passes_guard() -> None:
    assert_graph_distiller_write_sql(DDL)
    assert len(split_sql_statements(DDL)) >= 5


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM public.labor_law",
        "DELETE FROM labor_law",
        "INSERT INTO public.labor_qna (id) VALUES (1)",
        "UPDATE labor_qna SET content = 'x'",
        "DROP SCHEMA public CASCADE",
        "CREATE SCHEMA IF NOT EXISTS public",
        "CREATE SCHEMA evil",
        "DROP TABLE labor_law",
        "TRUNCATE graph_distiller.nodes; DELETE FROM labor_law",
        "SET search_path TO public",
        "SET search_path TO graph_distiller, public",
        "SELECT * FROM graph_distiller.nodes",
        "INSERT INTO nodes (run_id) VALUES ('x')",  # unqualified
    ],
)
def test_reject_forbidden_writes(sql: str) -> None:
    with pytest.raises(GraphDistillerWriteGuardError):
        assert_graph_distiller_write_sql(sql)


def test_allowed_schema_constant() -> None:
    assert ALLOWED_SCHEMA == "graph_distiller"
