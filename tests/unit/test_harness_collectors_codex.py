"""Unit: codex re-verify collector against a synthetic ledger, $0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.collectors import codex as codex_collector

pytestmark = pytest.mark.unit


def _metric(collected, name, subject=None):
    hits = [m for m in collected.metrics if m.name == name and (subject is None or m.subject == subject)]
    assert len(hits) == 1, f"{name}/{subject}: {hits}"
    return hits[0].value


def _check(collected, check_id, subject=None):
    hits = [c for c in collected.checks if c.check_id == check_id and (subject is None or c.subject == subject)]
    assert len(hits) == 1, f"{check_id}/{subject}: {hits}"
    return hits[0]


def _write_ledger(runs: Path, run_id: str, codex_dir: Path, chapters: list[dict]) -> None:
    d = runs / run_id
    d.mkdir(parents=True)
    ledger = {
        "run_id": run_id,
        "artifacts": {"codex_output_dir": str(codex_dir)},
        "chapters": chapters,
    }
    (d / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")


GOOD_BODY = (
    "# T\n\n제목 문단입니다. 실제로 근거를 인용합니다 [DOC_ID: memo1].\n\n"
    "두 번째 문단도 인용을 포함합니다 [DOC_ID: memo1].\n"
)


def test_codex_missing_dir_is_error_check(tmp_path: Path) -> None:
    out = codex_collector.collect(tmp_path / "nope", tmp_path / "runs")
    assert _check(out, "codex.input_present").severity == "error"


def test_codex_no_matching_ledger_is_warn(tmp_path: Path) -> None:
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    out = codex_collector.collect(codex_dir, tmp_path / "runs")
    chk = _check(out, "codex.matching_ledger_found")
    assert chk.passed is False and chk.severity == "warn"


def test_codex_verifies_clean_chapter_against_ledger_citations(tmp_path: Path) -> None:
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / "chapter_01_t.md").write_text(GOOD_BODY, encoding="utf-8")
    _write_ledger(
        tmp_path / "runs", "codex_1", codex_dir,
        [{"chapter_id": 0, "title": "T", "drafted": True, "poisoned": False, "cited_doc_ids": ["memo1"]}],
    )
    out = codex_collector.collect(codex_dir, tmp_path / "runs")

    assert _metric(out, "chapters_expected") == 1
    assert _metric(out, "chapters_on_disk") == 1
    assert _check(out, "codex.chapter_count_matches_ledger").passed is True
    assert _check(out, "codex.matches_last_verified_citations", "chapter_01").passed is True
    assert _check(out, "codex.no_text_corruption", "chapter_01").passed is True
    assert _check(out, "codex.verify_against_last_known_valid_ids", "chapter_01").passed is True
    assert _check(out, "codex.verify_against_last_known_valid_ids", "chapter_01").severity == "error"


def test_codex_detects_hand_edit_that_breaks_citations(tmp_path: Path) -> None:
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    edited = "# T\n\n누군가 인용을 지우고 손으로 고쳤다.\n"
    (codex_dir / "chapter_01_t.md").write_text(edited, encoding="utf-8")
    _write_ledger(
        tmp_path / "runs", "codex_1", codex_dir,
        [{"chapter_id": 0, "title": "T", "drafted": True, "poisoned": False, "cited_doc_ids": ["memo1"]}],
    )
    out = codex_collector.collect(codex_dir, tmp_path / "runs")

    assert _check(out, "codex.matches_last_verified_citations", "chapter_01").passed is False
    assert _check(out, "codex.verify_against_last_known_valid_ids", "chapter_01").passed is False


def test_codex_marks_stale_when_chapter_count_diverges(tmp_path: Path) -> None:
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    (codex_dir / "chapter_01_t.md").write_text(GOOD_BODY, encoding="utf-8")
    _write_ledger(
        tmp_path / "runs", "codex_1", codex_dir,
        [
            {"chapter_id": 0, "title": "T", "drafted": True, "poisoned": False, "cited_doc_ids": ["memo1"]},
            {"chapter_id": 1, "title": "U", "drafted": True, "poisoned": False, "cited_doc_ids": ["memo2"]},
        ],
    )
    out = codex_collector.collect(codex_dir, tmp_path / "runs")

    assert _check(out, "codex.chapter_count_matches_ledger").passed is False
    assert _metric(out, "file_present", "chapter_01") is True
    chk = _check(out, "codex.matches_last_verified_citations", "chapter_01")
    assert chk.severity == "warn"  # stale run → downgraded from error
    missing_chk = _check(out, "codex.chapter_file_present", "chapter_02")
    assert missing_chk.passed is False
