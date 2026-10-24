"""Unit: Phase E1 STATUS helpers — $0 (no DB)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from graph_distiller.rag_status import (
    EXPORT_REQUIRED_FILES,
    collect_status,
    fetch_graph_distiller_inventory,
    format_status_text,
    scan_export_dir,
    scan_rebuild_root,
    status_to_dict,
)

pytestmark = pytest.mark.unit


def _touch_complete(export: Path) -> None:
    export.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_REQUIRED_FILES:
        (export / name).write_text("{}", encoding="utf-8")
    (export / "nodes.json").write_text("[]", encoding="utf-8")
    (export / "edges.json").write_text("[]", encoding="utf-8")


def test_scan_export_complete_and_incomplete(tmp_path: Path) -> None:
    good = tmp_path / "rag_good"
    _touch_complete(good)
    st = scan_export_dir(good)
    assert st.status == "complete"
    assert st.missing == []
    assert st.run_id == "rag_good"

    bad = tmp_path / "rag_bad"
    bad.mkdir()
    (bad / "nodes.json").write_text("[]", encoding="utf-8")
    st2 = scan_export_dir(bad)
    assert st2.status == "incomplete"
    assert "meta.json" in st2.missing


def test_scan_rebuild_root(tmp_path: Path) -> None:
    _touch_complete(tmp_path / "a")
    (tmp_path / "b").mkdir()
    (tmp_path / ".skip").mkdir()
    rows = scan_rebuild_root(tmp_path)
    ids = {r.run_id for r in rows}
    assert ids == {"a", "b"}
    by = {r.run_id: r for r in rows}
    assert by["a"].status == "complete"
    assert by["b"].status == "incomplete"


def test_collect_status_offline_no_uri(tmp_path: Path) -> None:
    _touch_complete(tmp_path / "rag_x")
    report = collect_status(uri=None, rebuild_dir=tmp_path, skip_db=True)
    assert report.db_queried is False
    assert report.public is None
    assert report.graph_distiller is None
    assert len(report.exports) == 1
    payload = status_to_dict(report)
    dumped = json.dumps(payload)
    assert "password" not in dumped.lower()
    assert "postgres://" not in dumped
    text = format_status_text(report)
    assert "rag_x" in text
    assert "db_queried=False" in text


def test_fetch_graph_distiller_inventory_schema_absent() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    cur.fetchone.return_value = (False,)
    inv = fetch_graph_distiller_inventory(conn)
    assert inv.schema_exists is False
    assert inv.runs == []
    assert "runs" in inv.missing_tables


def test_fetch_graph_distiller_inventory_with_run() -> None:
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur

    cur.fetchone.side_effect = [
        (True,),
        (3,),
        (4,),
        (5,),
        (6,),
    ]
    cur.fetchall.side_effect = [
        [("chapters",), ("edges",), ("nodes",), ("runs",), ("source_map",)],
        [
            (
                "rag_1",
                "/tmp/rag_1",
                "cites_only",
                4,
                8,
                None,
            )
        ],
    ]
    inv = fetch_graph_distiller_inventory(conn)
    assert inv.schema_exists is True
    assert inv.missing_tables == []
    assert len(inv.runs) == 1
    assert inv.runs[0].run_id == "rag_1"
    assert inv.runs[0].nodes == 3
    assert inv.runs[0].edges == 4
    assert inv.runs[0].chapters == 5
    assert inv.runs[0].source_map == 6


def test_status_module_has_no_write_sql_literals() -> None:
    """E1 source must not contain WRITE/DDL verbs."""
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "graph_distiller"
        / "rag_status.py"
    )
    text = src.read_text(encoding="utf-8").upper()
    for banned in (
        "INSERT INTO",
        "UPDATE ",
        "DELETE FROM",
        "DROP SCHEMA",
        "DROP TABLE",
        "TRUNCATE",
        "CREATE TABLE",
        "CREATE SCHEMA",
    ):
        assert banned not in text, f"E1 must stay read-only; found {banned!r}"
