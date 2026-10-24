"""Unit: Phase E4 tent PLAN / execute abort — $0."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from graph_distiller.rag_load import SCHEMA
from graph_distiller.rag_status import EXPORT_REQUIRED_FILES
from graph_distiller.rag_tent import (
    OPTION_ABORT,
    OPTION_LOAD,
    execute_plan,
    format_menu,
    plan_abort,
    plan_cleanup_schema,
    plan_load,
    plan_rework,
    plan_status,
)

pytestmark = pytest.mark.unit


def _touch_complete(export: Path) -> None:
    export.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_REQUIRED_FILES:
        (export / name).write_text("{}", encoding="utf-8")
    (export / "nodes.json").write_text(
        json.dumps([{"id": "a"}, {"id": "b"}]), encoding="utf-8"
    )
    (export / "edges.json").write_text(
        json.dumps([{"source": "a", "target": "b"}]), encoding="utf-8"
    )
    (export / "toc.json").write_text(json.dumps({"0": ["a"]}), encoding="utf-8")


def test_menu_lists_all_options() -> None:
    text = format_menu()
    for key in ("[S]", "[L]", "[R]", "[C]", "[I]", "[X]"):
        assert key in text


def test_plan_strings_no_public_drop(tmp_path: Path) -> None:
    _touch_complete(tmp_path / "rag_x")
    plans = [
        plan_status(),
        plan_abort(),
        plan_load(tmp_path / "rag_x"),
        plan_cleanup_schema(),
        plan_rework(cleanup_mode="schema", export_dir=tmp_path / "rag_x"),
    ]
    for plan in plans:
        text = plan.format_text()
        assert "labor_law" in text
        assert "DROP SCHEMA public" not in text
        if plan.drop_target:
            assert plan.drop_target == SCHEMA
        if plan.option == OPTION_LOAD:
            assert plan.estimated.get("nodes") == 2
            assert plan.estimated.get("edges") == 1


def test_execute_abort_no_db_calls(tmp_path: Path) -> None:
    plan = plan_abort()
    with patch("graph_distiller.rag_tent.load_export_dir") as load:
        with patch("graph_distiller.rag_tent.cleanup_with_optional_exports") as clean:
            out = execute_plan(plan, uri="postgresql://x", rebuild_dir=tmp_path)
    assert out["action"] == "abort"
    assert out["db_changed"] is False
    load.assert_not_called()
    clean.assert_not_called()
    assert plan.option == OPTION_ABORT
    assert plan.mutates_db is False


def test_execute_load_delegates(tmp_path: Path) -> None:
    export = tmp_path / "rag_x"
    _touch_complete(export)
    plan = plan_load(export)

    class Fake:
        run_id = "rag_x"
        export_dir = str(export)
        nodes = 2
        edges = 1
        chapters = 1
        source_map_rows = 0

    with patch("graph_distiller.rag_tent.load_export_dir", return_value=Fake()) as load:
        with patch(
            "graph_distiller.rag_tent.count_graph_distiller_rows", return_value={"nodes": 2}
        ):
            out = execute_plan(plan, uri="postgresql://x", rebuild_dir=tmp_path)
    load.assert_called_once()
    assert out["action"] == "load"
    assert out["db_changed"] is True
    assert out["load"]["nodes"] == 2
