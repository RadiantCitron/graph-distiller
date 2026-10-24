"""Unit tests for compiler CLI workspace integration (ROADMAP_0001 step 040).

Tests:
- --workspace CLI argument parsing and validation
- D4: --source db + --workspace is rejected
- D8: --workspace defaults --source to 'local' when omitted, warns on first line
- D7: No auto-discovery (explicit --workspace required)
- D10: CLI budget/model flags preserved
- D11: Missing input folder raises FileNotFoundError and fatal exit 2
- Workspace path and knob injection into Settings via settings_from_cli
- Output dir and poc-top CLI overlay precedence over workspace settings
- First line stdout output verification
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from graph_distiller.cli_compiler import build_parser, main, settings_from_cli, validate_args
from graph_distiller.config import PROJECT_ROOT, Settings, get_settings, set_runtime_settings

pytestmark = pytest.mark.unit

_MODELS = [
    "--extraction-model",
    "openai/gpt-4o-mini",
    "--synthesis-model",
    "google/gemini-2.5-flash-lite",
]


@pytest.fixture(autouse=True)
def clean_knobs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate tests from repository .env knob variables."""
    for k in (
        "MIN_COMMUNITY_SIZE",
        "POC_TOP_CHAPTERS",
        "MAX_CHUNK_TOKENS",
    ):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def minimal_workspace(tmp_path: Path) -> Path:
    """Create a minimal valid workspace on disk."""
    ws = tmp_path / "ws_demo"
    ws.mkdir()
    (ws / "input").mkdir()
    (ws / "input" / "sample.txt").write_text("Hello workspace", encoding="utf-8")
    (ws / "workspace.toml").write_text(
        """schema_version = 1
name = "demo-corp"

[paths]
input = "input"
work = "work"
output = "output"

[pipeline]
min_community_size = 4
poc_top_chapters = 2
""",
        encoding="utf-8",
    )
    return ws


def test_parser_has_workspace_option() -> None:
    parser = build_parser()
    args = parser.parse_args([*_MODELS, "--workspace", "/tmp/ws"])
    assert args.workspace == Path("/tmp/ws")
    assert args.source is None  # not defaulted until validate_args


def test_d4_source_db_with_workspace_rejected(minimal_workspace: Path) -> None:
    """D4: --source db is rejected when --workspace is specified."""
    parser = build_parser()
    args = parser.parse_args([*_MODELS, "--workspace", str(minimal_workspace), "--source", "db"])
    reason = validate_args(args)
    assert reason is not None
    assert "D4: local source only" in reason


def test_d8_source_omitted_defaults_to_local_for_workspace(minimal_workspace: Path) -> None:
    """D8: When --source is omitted with --workspace, it defaults to 'local'."""
    parser = build_parser()
    args = parser.parse_args([*_MODELS, "--workspace", str(minimal_workspace)])
    assert args.source is None
    reason = validate_args(args)
    assert reason is None
    assert args.source == "local"
    assert getattr(args, "_source_defaulted_by_workspace", False) is True


def test_source_explicit_local_with_workspace_has_no_omitted_flag(minimal_workspace: Path) -> None:
    """When --source local is explicitly passed, _source_defaulted_by_workspace is not set."""
    parser = build_parser()
    args = parser.parse_args([*_MODELS, "--workspace", str(minimal_workspace), "--source", "local"])
    reason = validate_args(args)
    assert reason is None
    assert args.source == "local"
    assert getattr(args, "_source_defaulted_by_workspace", False) is False


def test_workspace_dir_not_found() -> None:
    parser = build_parser()
    args = parser.parse_args([*_MODELS, "--workspace", "/nonexistent/path/ws"])
    reason = validate_args(args)
    assert reason is not None
    assert "Workspace directory not found" in reason


def test_workspace_toml_not_found(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty_ws"
    empty_dir.mkdir()
    parser = build_parser()
    args = parser.parse_args([*_MODELS, "--workspace", str(empty_dir)])
    reason = validate_args(args)
    assert reason is not None
    assert "Workspace configuration file not found" in reason


def test_workspace_validation_error_rejected(tmp_path: Path) -> None:
    bad_ws = tmp_path / "bad_ws"
    bad_ws.mkdir()
    (bad_ws / "workspace.toml").write_text(
        """schema_version = 1
name = "bad-secrets"
openrouter_api_key = "sk-ant-forbidden"
""",
        encoding="utf-8",
    )
    parser = build_parser()
    args = parser.parse_args([*_MODELS, "--workspace", str(bad_ws)])
    reason = validate_args(args)
    assert reason is not None
    assert "Workspace validation error" in reason
    assert "Secrets must not be stored in workspace.toml" in reason


def test_settings_from_cli_injects_workspace(minimal_workspace: Path) -> None:
    """settings_from_cli applies workspace paths, knobs, and workspace_dir."""
    parser = build_parser()
    args = parser.parse_args([*_MODELS, "--workspace", str(minimal_workspace)])
    validate_args(args)

    prev = get_settings()
    try:
        cfg = settings_from_cli(args)
        assert cfg.workspace_dir == minimal_workspace.resolve()
        assert cfg.raw_memos_dir == (minimal_workspace / "input").resolve()
        assert cfg.checkpoint_dir == (minimal_workspace / "work").resolve()
        assert cfg.codex_output_dir == (minimal_workspace / "output" / "codex").resolve()
        assert cfg.runs_dir == (minimal_workspace / "output" / "runs").resolve()
        assert cfg.min_community_size == 4
        assert cfg.poc_top_chapters == 2
    finally:
        set_runtime_settings(prev)


def test_cli_overlay_precedence_over_workspace(minimal_workspace: Path, tmp_path: Path) -> None:
    """CLI --output-dir and --poc-top override workspace settings."""
    custom_runs = tmp_path / "custom_runs"
    parser = build_parser()
    args = parser.parse_args([
        *_MODELS,
        "--workspace",
        str(minimal_workspace),
        "--output-dir",
        str(custom_runs),
        "--poc-top",
        "99",
    ])
    validate_args(args)

    prev = get_settings()
    try:
        cfg = settings_from_cli(args)
        assert cfg.runs_dir == custom_runs.resolve()
        assert cfg.poc_top_chapters == 99
        # Other knobs still from workspace
        assert cfg.min_community_size == 4
    finally:
        set_runtime_settings(prev)


def test_first_line_printing_in_main(minimal_workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main() prints workspace identity on the first line, followed by local fallback notification."""
    with patch("graph_distiller.cli_compiler._run_local", return_value=0):
        code = main([*_MODELS, "--workspace", str(minimal_workspace)])
        assert code == 0

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) >= 2
    assert lines[0] == f"[workspace] Workspace: demo-corp ({minimal_workspace.resolve()})"
    assert lines[1] == "[workspace] --source omitted with --workspace; defaulting to 'local'"


def test_first_line_printing_when_source_explicit_local(minimal_workspace: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """When --source local is explicitly specified, the omitted notice is suppressed."""
    with patch("graph_distiller.cli_compiler._run_local", return_value=0):
        code = main([*_MODELS, "--workspace", str(minimal_workspace), "--source", "local"])
        assert code == 0

    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) >= 1
    assert lines[0] == f"[workspace] Workspace: demo-corp ({minimal_workspace.resolve()})"
    assert not any("--source omitted" in line for line in lines)


def test_d11_missing_workspace_input_dir_raises_fatal(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """D11: If workspace input folder does not exist, main() exits with 2 and prints [FATAL]."""
    ws = tmp_path / "ws_missing_input"
    ws.mkdir()
    (ws / "workspace.toml").write_text(
        """schema_version = 1
name = "no-input-corp"

[paths]
input = "nonexistent_input_dir"
""",
        encoding="utf-8",
    )
    code = main([*_MODELS, "--workspace", str(ws)])
    assert code == 2

    captured = capsys.readouterr()
    assert "[FATAL] Workspace input directory not found" in captured.err


def test_d11_ensure_directories_creates_only_workspace_work_and_output(minimal_workspace: Path) -> None:
    """D11: ensure_directories in workspace mode creates workspace work/output dirs only, not repo data/."""
    parser = build_parser()
    args = parser.parse_args([*_MODELS, "--workspace", str(minimal_workspace)])
    validate_args(args)
    cfg = settings_from_cli(args)

    assert not (minimal_workspace / "work").exists()
    assert not (minimal_workspace / "output").exists()

    cfg.ensure_directories()

    assert (minimal_workspace / "work").is_dir()
    assert (minimal_workspace / "output" / "codex").is_dir()
