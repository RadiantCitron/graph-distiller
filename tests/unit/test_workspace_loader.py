"""Unit tests for workspace.py loader (ROADMAP_0001 step 020).

Verifies:
- Unsupported schema_version rejection
- Unknown key rejection (extra="forbid")
- Forbidden secret / model slug / budget key rejection
- Relative path resolution relative to the file defining them
- 5-layer precedence: engine defaults < profile < workspace.toml < env < CLI
- File-by-file policy override (D5)
- Profile single-level inheritance constraint
- Pipeline knobs range and pattern validation
- Read-only behavior (no disk mutations)
- UTF-8 and Korean character support
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.workspace import (
    ENGINE_DEFAULT_KNOBS,
    LEXICON_FILENAME,
    PROMPT_FILES,
    RULE_FILES,
    WorkspaceValidationError,
    load_profile_file,
    load_workspace_file,
    resolve_workspace,
)

pytestmark = pytest.mark.unit


def test_missing_workspace_file_raises_error(tmp_path: Path) -> None:
    non_existent = tmp_path / "workspace.toml"
    with pytest.raises(WorkspaceValidationError, match="not found"):
        load_workspace_file(non_existent)


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    ws_file = tmp_path / "workspace.toml"
    ws_file.write_text("schema_version = 2\n", encoding="utf-8")
    with pytest.raises(WorkspaceValidationError, match="Unsupported schema_version"):
        load_workspace_file(ws_file)


def test_extra_unknown_keys_forbidden(tmp_path: Path) -> None:
    ws_file = tmp_path / "workspace.toml"
    ws_file.write_text("schema_version = 1\nunknown_top_level = 'value'\n", encoding="utf-8")
    with pytest.raises(WorkspaceValidationError, match="extra"):
        load_workspace_file(ws_file)


def test_forbidden_secret_keys_rejected(tmp_path: Path) -> None:
    ws_file = tmp_path / "workspace.toml"
    ws_file.write_text("schema_version = 1\nopenrouter_api_key = 'sk-test'\n", encoding="utf-8")
    with pytest.raises(WorkspaceValidationError, match="Secrets must not be stored"):
        load_workspace_file(ws_file)


def test_forbidden_model_slug_keys_rejected(tmp_path: Path) -> None:
    ws_file = tmp_path / "workspace.toml"
    ws_file.write_text("schema_version = 1\nextraction_model = 'gpt-4o'\n", encoding="utf-8")
    with pytest.raises(WorkspaceValidationError, match="Model slugs must be specified via CLI"):
        load_workspace_file(ws_file)


def test_forbidden_budget_section_rejected(tmp_path: Path) -> None:
    ws_file = tmp_path / "workspace.toml"
    ws_file.write_text("schema_version = 1\n[budget]\nmax_usd = 1.0\n", encoding="utf-8")
    with pytest.raises(WorkspaceValidationError, match=r"\[budget\] section is not allowed"):
        load_workspace_file(ws_file)


def test_profile_nested_inheritance_forbidden(tmp_path: Path) -> None:
    prof_file = tmp_path / "profile.toml"
    prof_file.write_text(
        "schema_version = 1\n[profile]\npath = '../parent_profile'\n",
        encoding="utf-8",
    )
    with pytest.raises(WorkspaceValidationError, match="only 1 level of inheritance"):
        load_profile_file(prof_file)


def test_knobs_range_validation(tmp_path: Path) -> None:
    ws_file = tmp_path / "workspace.toml"
    # max_chunk_tokens has ge=100
    ws_file.write_text("schema_version = 1\n[pipeline]\nmax_chunk_tokens = 50\n", encoding="utf-8")
    with pytest.raises(WorkspaceValidationError, match="greater than or equal to 100"):
        load_workspace_file(ws_file)

    # graph_scale_on_exceed must be 'abort' or 'slice'
    ws_file.write_text("schema_version = 1\n[pipeline]\ngraph_scale_on_exceed = 'ignore'\n", encoding="utf-8")
    with pytest.raises(WorkspaceValidationError, match="String should match pattern"):
        load_workspace_file(ws_file)


def test_relative_paths_resolved_against_file_location(tmp_path: Path) -> None:
    ws_dir = tmp_path / "workspaces" / "legal"
    ws_dir.mkdir(parents=True)
    ws_file = ws_dir / "workspace.toml"
    ws_file.write_text(
        """
schema_version = 1
name = "legal-docs"

[paths]
input = "raw_input"
work = "scratch"
output = "dist"
""",
        encoding="utf-8",
    )

    resolved = resolve_workspace(ws_dir)
    assert resolved.raw_memos_dir == (ws_dir / "raw_input").resolve()
    assert resolved.checkpoint_dir == (ws_dir / "scratch").resolve()
    assert resolved.codex_output_dir == (ws_dir / "dist" / "codex").resolve()
    assert resolved.errors_log_path == (ws_dir / "dist" / "errors.log").resolve()
    assert resolved.runs_dir == (ws_dir / "dist" / "runs").resolve()


def test_five_layer_precedence_and_provenance(tmp_path: Path) -> None:
    # 1. Profile with custom max_chunk_tokens and min_community_size
    prof_dir = tmp_path / "profiles" / "test_prof"
    prof_dir.mkdir(parents=True)
    (prof_dir / "profile.toml").write_text(
        """
schema_version = 1
[pipeline]
max_chunk_tokens = 1500
min_community_size = 12
""",
        encoding="utf-8",
    )

    # 2. Workspace referencing profile and overriding max_chunk_tokens
    ws_dir = tmp_path / "workspaces" / "my_ws"
    ws_dir.mkdir(parents=True)
    (ws_dir / "workspace.toml").write_text(
        """
schema_version = 1
[profile]
path = "../../profiles/test_prof"

[pipeline]
max_chunk_tokens = 2500
""",
        encoding="utf-8",
    )

    # 3. Environment overlay overriding raw_memos_dir
    custom_env_memos = tmp_path / "env_memos"
    env_overlay = {"RAW_MEMOS_DIR": str(custom_env_memos)}

    # 4. CLI overlay overriding poc_top_chapters and output_dir
    custom_cli_runs = tmp_path / "cli_runs"
    cli_overlay = {"poc_top_chapters": 7, "output_dir": str(custom_cli_runs)}

    resolved = resolve_workspace(
        ws_dir,
        env_overlay=env_overlay,
        cli_overlay=cli_overlay,
    )

    # Checks:
    # - max_chunk_tokens: workspace overrides profile and engine default
    assert resolved.knobs["max_chunk_tokens"] == 2500
    assert resolved.provenance["max_chunk_tokens"].source == "workspace"

    # - min_community_size: profile overrides engine default (8 -> 12)
    assert resolved.knobs["min_community_size"] == 12
    assert resolved.provenance["min_community_size"].source == "profile"

    # - llm_concurrency: engine default untouched (20)
    assert resolved.knobs["llm_concurrency"] == ENGINE_DEFAULT_KNOBS["llm_concurrency"]
    assert resolved.provenance["llm_concurrency"].source == "engine_default"

    # - raw_memos_dir: env overlay overrides workspace
    assert resolved.raw_memos_dir == custom_env_memos.resolve()
    assert resolved.provenance["raw_memos_dir"].source == "env"

    # - runs_dir: CLI overlay overrides workspace paths.output
    assert resolved.runs_dir == custom_cli_runs.resolve()
    assert resolved.provenance["runs_dir"].source == "cli"

    # - poc_top_chapters: CLI overlay overrides engine default
    assert resolved.knobs["poc_top_chapters"] == 7
    assert resolved.provenance["poc_top_chapters"].source == "cli"


def test_file_by_file_policy_overrides_d5(tmp_path: Path) -> None:
    """Only policy files explicitly present in workspace/profile override; others inherit."""
    # Profile provides only phase4_planner.md
    prof_dir = tmp_path / "prof"
    prof_prompts = prof_dir / "prompts"
    prof_prompts.mkdir(parents=True)
    custom_planner_prompt = prof_prompts / "phase4_planner.md"
    custom_planner_prompt.write_text("# Custom Planner Prompt", encoding="utf-8")
    (prof_dir / "profile.toml").write_text("schema_version = 1\n", encoding="utf-8")

    # Workspace provides only phase4_synthesizer.md and lexicon.json
    ws_dir = tmp_path / "ws"
    ws_prompts = ws_dir / "prompts"
    ws_prompts.mkdir(parents=True)
    custom_synth_prompt = ws_prompts / "phase4_synthesizer.md"
    custom_synth_prompt.write_text("# Custom Synthesizer Prompt", encoding="utf-8")
    custom_lexicon = ws_dir / "lexicon.json"
    custom_lexicon.write_text("{}", encoding="utf-8")

    (ws_dir / "workspace.toml").write_text(
        """
schema_version = 1
[profile]
path = "../prof"
""",
        encoding="utf-8",
    )

    resolved = resolve_workspace(ws_dir)

    # 1. Synthesizer prompt came from workspace
    assert resolved.policy_files["phase4_synthesizer.md"].source == "workspace"
    assert resolved.policy_files["phase4_synthesizer.md"].resolved_path == custom_synth_prompt.resolve()

    # 2. Planner prompt came from profile
    assert resolved.policy_files["phase4_planner.md"].source == "profile"
    assert resolved.policy_files["phase4_planner.md"].resolved_path == custom_planner_prompt.resolve()

    # 3. Extraction prompt inherited from engine_default
    assert resolved.policy_files["phase2_extraction.md"].source == "engine_default"

    # 4. Lexicon came from workspace
    assert resolved.policy_files[LEXICON_FILENAME].source == "workspace"
    assert resolved.policy_files[LEXICON_FILENAME].resolved_path == custom_lexicon.resolve()

    # 5. Rules all inherited from engine_default
    for r_name in RULE_FILES:
        assert resolved.policy_files[r_name].source == "engine_default"


def test_loader_is_read_only_and_does_not_mutate_disk(tmp_path: Path) -> None:
    ws_dir = tmp_path / "clean_ws"
    ws_dir.mkdir()
    ws_file = ws_dir / "workspace.toml"
    ws_file.write_text("schema_version = 1\nname = 'read-only-test'\n", encoding="utf-8")

    before_entries = set(ws_dir.iterdir())
    resolve_workspace(ws_dir)
    after_entries = set(ws_dir.iterdir())

    # No input/, work/, output/ or any subdirectories should have been created
    assert before_entries == after_entries == {ws_file}


def test_utf8_and_korean_paths_support(tmp_path: Path) -> None:
    ws_dir = tmp_path / "한글_코퍼스_워크스페이스"
    ws_dir.mkdir()
    ws_file = ws_dir / "workspace.toml"
    ws_file.write_text(
        """
schema_version = 1
name = "한국어_테스트"

[paths]
input = "원문_폴더"
work = "작업_폴더"
output = "산출물_폴더"
""",
        encoding="utf-8",
    )

    resolved = resolve_workspace(ws_dir)
    assert resolved.name == "한국어_테스트"
    assert resolved.raw_memos_dir == (ws_dir / "원문_폴더").resolve()
    assert resolved.checkpoint_dir == (ws_dir / "작업_폴더").resolve()
    assert resolved.codex_output_dir == (ws_dir / "산출물_폴더" / "codex").resolve()
