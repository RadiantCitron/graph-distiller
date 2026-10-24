"""Unit: compiler argparse + db-mode gates ($0, no Postgres)."""

from __future__ import annotations

import pytest

from graph_distiller.cli_compiler import build_parser, validate_args
from graph_distiller.rag_gates import fatal_if_db_mode_ambiguous, fatal_if_full_db_default

pytestmark = pytest.mark.unit

_MODELS = [
    "--extraction-model",
    "openai/gpt-4o-mini",
    "--synthesis-model",
    "google/gemini-2.5-flash-lite",
]


def test_parser_requires_model_args() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--source", "db", "--sample", "--n-jo", "1", "--m-qna", "1"])


def test_validate_db_requires_sample_or_full() -> None:
    args = build_parser().parse_args([*_MODELS, "--source", "db"])
    reason = validate_args(args)
    assert reason is not None
    assert "sample" in reason.lower() or "full-db" in reason.lower()


def test_validate_sample_requires_sizes() -> None:
    args = build_parser().parse_args([*_MODELS, "--source", "db", "--sample"])
    reason = validate_args(args)
    assert reason is not None
    assert "n-jo" in reason.lower() or "sample" in reason.lower()


def test_validate_sample_with_sizes_ok() -> None:
    args = build_parser().parse_args(
        [*_MODELS, "--source", "db", "--sample", "--n-jo", "2", "--m-qna", "2"]
    )
    assert validate_args(args) is None


def test_validate_full_db_ok_without_sizes() -> None:
    args = build_parser().parse_args([*_MODELS, "--source", "db", "--full-db"])
    assert validate_args(args) is None


def test_validate_sample_and_full_conflict() -> None:
    args = build_parser().parse_args(
        [*_MODELS, "--source", "db", "--sample", "--full-db", "--n-jo", "1", "--m-qna", "0"]
    )
    reason = validate_args(args)
    assert reason is not None
    assert "exactly one" in reason.lower() or "not both" in reason.lower()


def test_validate_local_rejects_sample_flags() -> None:
    args = build_parser().parse_args([*_MODELS, "--source", "local", "--sample"])
    reason = validate_args(args)
    assert reason is not None


def test_validate_empty_model_strings() -> None:
    args = build_parser().parse_args(
        ["--extraction-model", "  ", "--synthesis-model", "x", "--source", "local"]
    )
    reason = validate_args(args)
    assert reason is not None
    assert "extraction-model" in reason


def test_settings_from_cli_ignores_env_model_slugs(monkeypatch) -> None:
    from graph_distiller.cli_compiler import settings_from_cli
    from graph_distiller.config import get_settings, set_runtime_settings

    monkeypatch.setenv("EXTRACTION_MODEL", "env/must-not-win")
    monkeypatch.setenv("SYNTHESIS_MODEL", "env/must-not-win-synth")
    prev = get_settings()
    try:
        args = build_parser().parse_args([*_MODELS, "--source", "local"])
        cfg = settings_from_cli(args)
        assert cfg.extraction_model == "openai/gpt-4o-mini"
        assert cfg.synthesis_model == "google/gemini-2.5-flash-lite"
        assert cfg.extraction_model != "env/must-not-win"
    finally:
        set_runtime_settings(prev)


def test_silent_full_db_still_fatal() -> None:
    assert fatal_if_full_db_default(n_jo=None, m_qna=None, llm=False)
    assert fatal_if_db_mode_ambiguous(sample=False, full_db=False) is not None
    assert fatal_if_db_mode_ambiguous(sample=False, full_db=True) is None
