"""Unit: 프롬프트 통제실 로더 (ADR-059) — $0."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.prompts import load_prompt

pytestmark = pytest.mark.unit


def test_load_prompt_reads_file_content(tmp_path: Path) -> None:
    (tmp_path / "phase2_extraction.md").write_text("  실제 프롬프트 내용  \n", encoding="utf-8")
    result = load_prompt(tmp_path, "phase2_extraction.md", fallback="FALLBACK")
    assert result == "실제 프롬프트 내용"


def test_load_prompt_falls_back_when_file_missing(tmp_path: Path) -> None:
    result = load_prompt(tmp_path, "does_not_exist.md", fallback="FALLBACK")
    assert result == "FALLBACK"


def test_load_prompt_falls_back_when_file_empty(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    result = load_prompt(tmp_path, "empty.md", fallback="FALLBACK")
    assert result == "FALLBACK"


def test_load_prompt_warns_on_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        load_prompt(tmp_path, "missing.md", fallback="FALLBACK")
    assert "missing.md" in caplog.text
