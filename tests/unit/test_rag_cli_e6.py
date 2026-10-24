"""Unit: Phase E6 CLI bridges / parser flags — $0."""

from __future__ import annotations

import pytest

from graph_distiller.cli_rag_skeleton import build_parser
from graph_distiller.rag_load import SCHEMA as LOAD_SCHEMA
from graph_distiller.rag_write_guard import assert_graph_distiller_write_sql

pytestmark = pytest.mark.unit


def test_skeleton_parser_has_e6_flags() -> None:
    p = build_parser()
    args = p.parse_args(["--status"])
    assert args.status is True
    args = p.parse_args(["--cleanup-schema", "-y"])
    assert args.cleanup_schema is True
    assert args.yes is True
    args = p.parse_args(["--interactive"])
    assert args.interactive is True


def test_cleanup_sql_still_graph_distiller_only() -> None:
    assert LOAD_SCHEMA == "graph_distiller"
    assert_graph_distiller_write_sql("DROP SCHEMA IF EXISTS graph_distiller CASCADE")
