"""Unit: Phase E5 resume / interrupted work — $0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_distiller.rag_load import SCHEMA
from graph_distiller.rag_resume import (
    HEALTH_INCOMPLETE_EXPORT,
    HEALTH_OK,
    HEALTH_PARTIAL_LOAD,
    RESOLVE_RESUME,
    RESOLVE_WIPE_SCHEMA,
    assess_export_vs_db,
    interrupted_only,
    plan_from_resolution,
)
from graph_distiller.rag_status import EXPORT_REQUIRED_FILES, GraphDistillerInventory, GraphDistillerRunStatus
from graph_distiller.rag_write_guard import assert_graph_distiller_write_sql

pytestmark = pytest.mark.unit


def _touch_complete(export: Path, *, nodes: int = 2, edges: int = 1) -> None:
    export.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_REQUIRED_FILES:
        (export / name).write_text("{}", encoding="utf-8")
    node_rows = [{"id": f"n{i}"} for i in range(nodes)]
    edge_rows = [
        {"source": "n0", "target": "n1"} for _ in range(edges)
    ]
    (export / "nodes.json").write_text(json.dumps(node_rows), encoding="utf-8")
    (export / "edges.json").write_text(json.dumps(edge_rows), encoding="utf-8")
    (export / "toc.json").write_text(json.dumps({"0": ["n0"]}), encoding="utf-8")
    (export / "source_map.json").write_text(json.dumps({"jo": []}), encoding="utf-8")


def test_incomplete_export_no_resume(tmp_path: Path) -> None:
    bad = tmp_path / "rag_bad"
    bad.mkdir()
    (bad / "nodes.json").write_text("[]", encoding="utf-8")
    a = assess_export_vs_db(bad, inventory=None)
    assert a.health == HEALTH_INCOMPLETE_EXPORT
    assert a.resume_possible is False
    with pytest.raises(ValueError, match="Resume not possible"):
        plan_from_resolution(a, RESOLVE_RESUME)


def test_partial_load_resume_and_wipe_plans(tmp_path: Path) -> None:
    export = tmp_path / "rag_partial"
    _touch_complete(export, nodes=5, edges=3)
    inv = GraphDistillerInventory(
        schema_exists=True,
        tables=["runs", "nodes", "edges", "chapters", "source_map"],
        missing_tables=[],
        runs=[
            GraphDistillerRunStatus(
                run_id="rag_partial",
                export_dir=str(export),
                llm_mode="cites_only",
                sample_n_jo=1,
                sample_m_qna=1,
                loaded_at_utc=None,
                nodes=1,
                edges=0,
                chapters=1,
                source_map=0,
            )
        ],
    )
    a = assess_export_vs_db(
        export,
        inventory=inv,
        db_counts={"nodes": 1, "edges": 0, "chapters": 1, "source_map": 0},
    )
    assert a.health == HEALTH_PARTIAL_LOAD
    assert a.resume_possible is True
    resume = plan_from_resolution(a, RESOLVE_RESUME)
    assert resume.mutates_db is True
    assert resume.replace_load is True
    assert resume.export_dir
    wipe = plan_from_resolution(a, RESOLVE_WIPE_SCHEMA)
    assert wipe.drop_target == SCHEMA
    assert wipe.cleanup_mode == "schema"
    assert wipe.mutates_db is True
    # wipe SQL path still graph_distiller-only
    assert_graph_distiller_write_sql(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")


def test_ok_when_counts_match(tmp_path: Path) -> None:
    export = tmp_path / "rag_ok"
    _touch_complete(export, nodes=2, edges=1)
    inv = GraphDistillerInventory(
        schema_exists=True,
        tables=["runs", "nodes", "edges", "chapters", "source_map"],
        missing_tables=[],
        runs=[
            GraphDistillerRunStatus(
                run_id="rag_ok",
                export_dir=str(export),
                llm_mode="",
                sample_n_jo=None,
                sample_m_qna=None,
                loaded_at_utc=None,
                nodes=2,
                edges=1,
                chapters=1,
                source_map=0,
            )
        ],
    )
    a = assess_export_vs_db(
        export,
        inventory=inv,
        db_counts={"nodes": 2, "edges": 1, "chapters": 1, "source_map": 0},
    )
    assert a.health == HEALTH_OK
    assert interrupted_only([a]) == []
    abort = plan_from_resolution(a, "none")
    assert abort.mutates_db is False
