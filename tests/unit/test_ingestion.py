"""Unit: flatten_json + source_doc_id sanitize (ADR-056) — no network / $0."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.ingestion.ingestion import (
    flatten_json,
    load_raw_memos,
    sanitize_source_doc_id,
)

pytestmark = pytest.mark.unit


def test_flatten_nested_dict_and_list() -> None:
    data = {
        "user": {"name": "Ada", "tags": ["a", "b"]},
        "count": 2,
    }
    lines = flatten_json(data)
    assert "user.name: Ada" in lines
    assert "user.tags[0]: a" in lines
    assert "user.tags[1]: b" in lines
    assert "count: 2" in lines


def test_flatten_bool_and_null() -> None:
    lines = flatten_json({"ok": True, "missing": None})
    assert "ok: true" in lines
    assert "missing: null" in lines


def test_flatten_empty_containers_with_prefix() -> None:
    assert flatten_json({}, "empty") == ["empty: {}"]
    assert flatten_json([], "items") == ["items: []"]


def test_flatten_scalar_without_prefix() -> None:
    assert flatten_json("hello") == ["hello"]


def test_sanitize_source_doc_id_collapses_whitespace() -> None:
    assert sanitize_source_doc_id("인공지능 발전 방향") == "인공지능_발전_방향"
    assert sanitize_source_doc_id("a\tb\nc") == "a_b_c"
    assert sanitize_source_doc_id("  spaced  name  ") == "_spaced_name_"


def test_sanitize_source_doc_id_leaves_no_space_stem() -> None:
    assert sanitize_source_doc_id("인공지능개론") == "인공지능개론"
    assert sanitize_source_doc_id("memo_20231012") == "memo_20231012"


def test_load_raw_memos_sanitizes_stem(tmp_path: Path) -> None:
    path = tmp_path / "인공지능 발전 방향.txt"
    path.write_text("hello body", encoding="utf-8")
    docs = load_raw_memos(tmp_path)
    assert len(docs) == 1
    assert docs[0]["source_doc_id"] == "인공지능_발전_방향"
    assert docs[0]["content"] == "hello body"