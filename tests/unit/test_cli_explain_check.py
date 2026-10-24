"""Unit tests for --explain and --check-config flags (ROADMAP_0001 Step 050)."""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from graph_distiller.cli_compiler import main
from graph_distiller.workspace import reset_deprecated_knobs_warnings

pytestmark = pytest.mark.unit


@pytest.fixture
def clean_knobs_env(monkeypatch):
    """Ensure baseline repository .env knobs do not leak into test environment."""
    from graph_distiller.workspace import KNOB_ENV_TO_FIELD

    for env_k in KNOB_ENV_TO_FIELD:
        monkeypatch.delenv(env_k, raising=False)
    reset_deprecated_knobs_warnings()
    yield monkeypatch
    reset_deprecated_knobs_warnings()


def test_check_config_workspace_success(tmp_path: Path, clean_knobs_env, capsys):
    ws_dir = tmp_path / "valid_ws"
    ws_dir.mkdir()
    (ws_dir / "input").mkdir()
    (ws_dir / "workspace.toml").write_text(
        'schema_version = 1\nname = "test_valid"\n',
        encoding="utf-8",
    )

    rc = main(["--workspace", str(ws_dir), "--check-config"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "[check-config] OK" in captured.out
    assert "test_valid" in captured.out


def test_check_config_workspace_missing_dir(tmp_path: Path, capsys):
    nonexistent = tmp_path / "nonexistent_ws"
    rc = main(["--workspace", str(nonexistent), "--check-config"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Workspace directory not found" in captured.err
    assert str(nonexistent) in captured.err


def test_check_config_workspace_missing_toml(tmp_path: Path, capsys):
    ws_dir = tmp_path / "empty_ws"
    ws_dir.mkdir()
    rc = main(["--workspace", str(ws_dir), "--check-config"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Workspace configuration file not found" in captured.err


def test_check_config_workspace_invalid_toml(tmp_path: Path, capsys):
    ws_dir = tmp_path / "broken_ws"
    ws_dir.mkdir()
    (ws_dir / "workspace.toml").write_text("invalid toml syntax [[[\n", encoding="utf-8")
    rc = main(["--workspace", str(ws_dir), "--check-config"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "[FATAL]" in captured.err or "Workspace validation error" in captured.err


def test_check_config_workspace_missing_input_dir(tmp_path: Path, clean_knobs_env, capsys):
    ws_dir = tmp_path / "no_input_ws"
    ws_dir.mkdir()
    (ws_dir / "workspace.toml").write_text(
        'schema_version = 1\nname = "no_input"\n',
        encoding="utf-8",
    )
    rc = main(["--workspace", str(ws_dir), "--check-config"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "Input directory not found" in captured.err


def test_check_config_workspace_invalid_rule_schema(tmp_path: Path, clean_knobs_env, capsys):
    ws_dir = tmp_path / "bad_rule_ws"
    ws_dir.mkdir()
    (ws_dir / "input").mkdir()
    rules_dir = ws_dir / "rules"
    rules_dir.mkdir()
    # Out of range top_fraction (> 1.0)
    (rules_dir / "phase4_repomap.toml").write_text("top_fraction = 2.5\n", encoding="utf-8")
    (ws_dir / "workspace.toml").write_text(
        'schema_version = 1\nname = "bad_rule"\n',
        encoding="utf-8",
    )
    rc = main(["--workspace", str(ws_dir), "--check-config"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "[FATAL]" in captured.err
    assert "phase4_repomap.toml" in captured.err


def test_check_config_workspace_invalid_rule_syntax(tmp_path: Path, clean_knobs_env, capsys):
    ws_dir = tmp_path / "bad_syntax_ws"
    ws_dir.mkdir()
    (ws_dir / "input").mkdir()
    rules_dir = ws_dir / "rules"
    rules_dir.mkdir()
    (rules_dir / "phase2_canonicalize.toml").write_text("bad toml [[[\n", encoding="utf-8")
    (ws_dir / "workspace.toml").write_text(
        'schema_version = 1\nname = "bad_syntax"\n',
        encoding="utf-8",
    )
    rc = main(["--workspace", str(ws_dir), "--check-config"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "[FATAL]" in captured.err
    assert "phase2_canonicalize.toml" in captured.err


def test_check_config_default_mode(capsys):
    rc = main(["--check-config"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "[check-config] OK: default configuration and schemas are valid." in captured.out


def test_check_config_does_not_create_dirs_or_logs(tmp_path: Path, clean_knobs_env):
    ws_dir = tmp_path / "pure_ws"
    ws_dir.mkdir()
    (ws_dir / "input").mkdir()
    (ws_dir / "workspace.toml").write_text(
        'schema_version = 1\nname = "pure"\n',
        encoding="utf-8",
    )

    work_dir = ws_dir / "work"
    output_dir = ws_dir / "output"
    assert not work_dir.exists()
    assert not output_dir.exists()

    rc = main(["--workspace", str(ws_dir), "--check-config"])
    assert rc == 0
    assert not work_dir.exists()
    assert not output_dir.exists()


def test_explain_workspace_layers_accuracy(tmp_path: Path, clean_knobs_env, monkeypatch, capsys):
    # Set up profile
    prof_dir = tmp_path / "test_prof"
    prof_dir.mkdir()
    (prof_dir / "rules").mkdir()
    (prof_dir / "rules" / "phase4_planner.toml").write_text(
        'max_title_words = 7\n', encoding="utf-8"
    )
    (prof_dir / "profile.toml").write_text(
        'schema_version = 1\nname = "custom_prof"\n[pipeline]\nlouvain_resolution = 0.75\n',
        encoding="utf-8",
    )

    # Set up workspace inheriting profile
    ws_dir = tmp_path / "explain_ws"
    ws_dir.mkdir()
    (ws_dir / "input").mkdir()
    (ws_dir / "rules").mkdir()
    (ws_dir / "rules" / "phase4_repomap.toml").write_text(
        'top_fraction = 0.15\n', encoding="utf-8"
    )
    (ws_dir / "workspace.toml").write_text(
        f'schema_version = 1\nname = "explain_workspace"\n'
        f'[profile]\npath = "{prof_dir.as_posix()}"\n'
        f'[pipeline]\nmax_chunk_tokens = 750\n',
        encoding="utf-8",
    )

    # Deprecated env knob
    monkeypatch.setenv("MIN_COMMUNITY_SIZE", "12")

    rc = main(["--workspace", str(ws_dir), "--poc-top", "4", "--explain"])
    captured = capsys.readouterr()
    assert rc == 0

    out = captured.out
    # Check general info
    assert "explain_workspace" in out
    # Check policy file provenance (D5)
    assert "phase4_repomap.toml" in out
    assert "phase4_planner.toml" in out
    # Check knobs layers
    assert "max_chunk_tokens" in out
    assert "750" in out
    assert "workspace" in out

    assert "louvain_resolution" in out
    assert "0.75" in out
    assert "profile" in out

    assert "min_community_size" in out
    assert "12" in out
    assert "env (deprecated)" in out
    assert "MIN_COMMUNITY_SIZE" in out

    assert "poc_top_chapters" in out
    assert "4" in out
    assert "cli" in out


def test_explain_secrets_strictly_masked(clean_knobs_env, monkeypatch, capsys):
    secret_key = "sk-super-secret-key-xyz-12345"
    secret_uri = "postgresql://user:password@localhost:5432/memos"
    monkeypatch.setenv("OPENROUTER_API_KEY", secret_key)
    monkeypatch.setenv("RAG_DB_URI", secret_uri)

    rc = main(["--explain"])
    captured = capsys.readouterr()
    assert rc == 0

    out = captured.out
    err = captured.err
    # [set] must be displayed
    assert "[set]" in out
    # Actual secrets MUST NEVER appear anywhere
    assert secret_key not in out
    assert secret_key not in err
    assert "db.example.com" not in out
    assert "db.example.com" not in err


def test_explain_no_llm_or_network_calls(tmp_path: Path, clean_knobs_env):
    ws_dir = tmp_path / "offline_ws"
    ws_dir.mkdir()
    (ws_dir / "input").mkdir()
    (ws_dir / "workspace.toml").write_text(
        'schema_version = 1\nname = "offline"\n',
        encoding="utf-8",
    )

    with patch("graph_distiller.cli_compiler.fetch_openrouter_rate_book") as mock_fetch, \
         patch("graph_distiller.local_pipeline.run_pipeline") as mock_pipe, \
         patch("graph_distiller.cli_compiler._run_db") as mock_db:
        rc = main(["--workspace", str(ws_dir), "--explain"])
        assert rc == 0
        mock_fetch.assert_not_called()
        mock_pipe.assert_not_called()
        mock_db.assert_not_called()


def test_check_config_and_explain_together(tmp_path: Path, clean_knobs_env, capsys):
    ws_dir = tmp_path / "combo_ws"
    ws_dir.mkdir()
    (ws_dir / "input").mkdir()
    (ws_dir / "workspace.toml").write_text(
        'schema_version = 1\nname = "combo"\n',
        encoding="utf-8",
    )

    rc = main(["--workspace", str(ws_dir), "--check-config", "--explain"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "[check-config] OK" in captured.out
    assert "Configuration Overview" in captured.out
    assert "combo" in captured.out


def test_check_config_d4_rejection(tmp_path: Path, capsys):
    ws_dir = tmp_path / "d4_ws"
    ws_dir.mkdir()
    (ws_dir / "workspace.toml").write_text('schema_version = 1\n', encoding="utf-8")
    rc = main(["--workspace", str(ws_dir), "--source", "db", "--check-config"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "D4: local source only" in captured.err
