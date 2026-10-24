"""Unit tests for knob TOML migration and deprecated env vars (ROADMAP_0001 step 045).

Tests:
- TOML knob values applied when env vars are clear ($0)
- Deprecated env vars override TOML knobs (layer 4 > layer 3)
- Deprecated env vars emit warning when set, no warning when unset
- Path and secret env vars do NOT emit deprecation warnings
- Invalid TOML knob values fail fast
- Invalid env knob values fail in workspace mode
- All 19 numerical & boolean knobs are tracked and covered
- 5-layer hierarchy: engine defaults < profile < workspace < env < CLI
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from graph_distiller.config import Settings, reset_deprecated_knobs_warnings
from graph_distiller.workspace import (
    ENGINE_DEFAULT_KNOBS,
    KNOB_ENV_TO_FIELD,
    WorkspaceValidationError,
    load_workspace_file,
    resolve_workspace,
)

pytestmark = pytest.mark.unit

_ENV_KEYS_TO_CLEAR = [
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "EMBEDDING_MODEL",
    "EXTRACTION_MODEL",
    "SYNTHESIS_MODEL",
    "RAW_MEMOS_DIR",
    "CHECKPOINT_DIR",
    "CODEX_OUTPUT_DIR",
    "ERRORS_LOG_PATH",
    "PROMPTS_DIR",
    "RULES_DIR",
    "RAG_DB_URI",
    "DB_URI",
    "RAG_REBUILD_DIR",
    "RUNS_DIR",
    "REPORTS_DIR",
    "CANON_ALIAS_LEXICON_PATH",
    *KNOB_ENV_TO_FIELD.keys(),
]


@pytest.fixture(autouse=True)
def clean_env_and_warnings(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for k in _ENV_KEYS_TO_CLEAR:
        monkeypatch.delenv(k, raising=False)
    reset_deprecated_knobs_warnings()
    yield
    reset_deprecated_knobs_warnings()


@pytest.fixture
def workspace_with_custom_knobs(tmp_path: Path) -> Path:
    ws = tmp_path / "ws_custom_knobs"
    ws.mkdir()
    (ws / "input").mkdir()
    (ws / "workspace.toml").write_text(
        """schema_version = 1
name = "custom-knobs-corp"

[paths]
input = "input"

[pipeline]
max_chunk_tokens = 750
llm_concurrency = 12
poc_top_chapters = 5
louvain_resolution = 0.8
min_community_size = 4
min_chapter_grounding_chunks = 3
repomap_small_keep_all_below = 8
repomap_snippet_chars = 1500
verify_citation_paragraph_ratio = 0.9
verify_citation_sentence_ratio = 0.3
canon_embed_merge = true
canon_embed_sim_threshold = 0.95
canon_embed_max_nodes = 1500
graph_max_nodes = 50000
graph_max_edges = 200000
graph_max_est_mb = 512.0
graph_scale_on_exceed = "slice"
""",
        encoding="utf-8",
    )
    return ws


def test_all_19_knobs_accounted_for() -> None:
    """Ensure KNOB_ENV_TO_FIELD contains exactly the 19 numerical & boolean knobs."""
    assert len(KNOB_ENV_TO_FIELD) == 19
    assert len(ENGINE_DEFAULT_KNOBS) == 19
    for env_k, field_name in KNOB_ENV_TO_FIELD.items():
        assert field_name in ENGINE_DEFAULT_KNOBS


def test_toml_knobs_applied_when_env_clear(workspace_with_custom_knobs: Path) -> None:
    """When knob env vars are clear, Settings loads knob values from workspace.toml with NO warnings."""
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        s = Settings.from_env(models_required=False, workspace_dir=workspace_with_custom_knobs)

    # All custom knobs from workspace.toml applied
    assert s.max_chunk_tokens == 750
    assert s.llm_concurrency == 12
    assert s.poc_top_chapters == 5
    assert s.louvain_resolution == 0.8
    assert s.min_community_size == 4
    assert s.min_chapter_grounding_chunks == 3
    assert s.repomap_small_keep_all_below == 8
    assert s.repomap_snippet_chars == 1500
    assert s.verify_citation_paragraph_ratio == 0.9
    assert s.verify_citation_sentence_ratio == 0.3
    assert s.canon_embed_merge is True
    assert s.canon_embed_sim_threshold == 0.95
    assert s.canon_embed_max_nodes == 1500
    assert s.graph_max_nodes == 50000
    assert s.graph_max_edges == 200000
    assert s.graph_max_est_mb == 512.0
    assert s.graph_scale_on_exceed == "slice"

    # Zero deprecation warnings
    dep_warnings = [w for w in recorded if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 0


def test_deprecated_env_knob_overrides_toml_and_warns(
    workspace_with_custom_knobs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deprecated environment variable overrides workspace.toml and emits a warning (layer 4 > layer 3)."""
    monkeypatch.setenv("MAX_CHUNK_TOKENS", "1250")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        s = Settings.from_env(models_required=False, workspace_dir=workspace_with_custom_knobs)

    # Env value wins over workspace.toml (1250 > 750)
    assert s.max_chunk_tokens == 1250
    # Other knobs still come from workspace.toml
    assert s.llm_concurrency == 12

    # Deprecation warning emitted for MAX_CHUNK_TOKENS
    dep_warnings = [
        w for w in recorded
        if issubclass(w.category, DeprecationWarning) and "MAX_CHUNK_TOKENS" in str(w.message)
    ]
    assert len(dep_warnings) == 1
    assert "Move this knob to [pipeline] in workspace.toml" in str(dep_warnings[0].message)


def test_provenance_records_env_source_for_overridden_knob(
    workspace_with_custom_knobs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve_workspace records source='env' for knobs overridden by environment variables."""
    monkeypatch.setenv("MAX_CHUNK_TOKENS", "1250")
    resolved = resolve_workspace(workspace_dir=workspace_with_custom_knobs)

    assert resolved.knobs["max_chunk_tokens"] == 1250
    assert resolved.provenance["max_chunk_tokens"].source == "env"
    assert resolved.provenance["max_chunk_tokens"].value == 1250

    # Non-overridden knob has source='workspace'
    assert resolved.knobs["llm_concurrency"] == 12
    assert resolved.provenance["llm_concurrency"].source == "workspace"


def test_secret_and_path_env_vars_do_not_warn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Non-knob env vars (secrets, models, paths) must NOT emit deprecation warnings."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-mock-key")
    monkeypatch.setenv("RAW_MEMOS_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("CODEX_OUTPUT_DIR", str(tmp_path / "codex"))
    monkeypatch.setenv("ERRORS_LOG_PATH", str(tmp_path / "errors.log"))
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path / "prompts"))
    monkeypatch.setenv("RULES_DIR", str(tmp_path / "rules"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        Settings.from_env(models_required=False)

    dep_warnings = [w for w in recorded if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) == 0


def test_invalid_toml_knob_fails_fast(tmp_path: Path) -> None:
    """Invalid knob in workspace.toml raises WorkspaceValidationError immediately."""
    ws = tmp_path / "ws_invalid_knob"
    ws.mkdir()
    (ws / "workspace.toml").write_text(
        """schema_version = 1
name = "invalid-corp"
[pipeline]
max_chunk_tokens = 99999999
""",
        encoding="utf-8",
    )
    with pytest.raises(WorkspaceValidationError, match="max_chunk_tokens"):
        load_workspace_file(ws / "workspace.toml")


def test_invalid_env_knob_fails_in_workspace(
    workspace_with_custom_knobs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid knob in environment raises WorkspaceValidationError in resolve_workspace."""
    monkeypatch.setenv("GRAPH_SCALE_ON_EXCEED", "explode")
    with pytest.raises(WorkspaceValidationError, match="GRAPH_SCALE_ON_EXCEED"):
        resolve_workspace(workspace_dir=workspace_with_custom_knobs)


def test_all_19_knobs_warn_when_set_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each of the 19 knobs emits a deprecation warning when set in environment."""
    test_values = {
        "MAX_CHUNK_TOKENS": "1200",
        "SEMANTIC_BREAKPOINT_PERCENTILE": "85",
        "LLM_CONCURRENCY": "25",
        "MAX_CHAPTER_RETRIES": "4",
        "POC_TOP_CHAPTERS": "3",
        "LOUVAIN_RESOLUTION": "0.6",
        "MIN_COMMUNITY_SIZE": "10",
        "MIN_CHAPTER_GROUNDING_CHUNKS": "3",
        "REPOMAP_SMALL_KEEP_ALL_BELOW": "15",
        "REPOMAP_SNIPPET_CHARS": "2500",
        "VERIFY_CITATION_PARAGRAPH_RATIO": "0.8",
        "VERIFY_CITATION_SENTENCE_RATIO": "0.3",
        "CANON_EMBED_MERGE": "true",
        "CANON_EMBED_SIM_THRESHOLD": "0.92",
        "CANON_EMBED_MAX_NODES": "3000",
        "GRAPH_MAX_NODES": "120000",
        "GRAPH_MAX_EDGES": "600000",
        "GRAPH_MAX_EST_MB": "2048.0",
        "GRAPH_SCALE_ON_EXCEED": "slice",
    }

    for env_k, val in test_values.items():
        reset_deprecated_knobs_warnings()
        monkeypatch.setenv(env_k, val)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            Settings.from_env(models_required=False)
        monkeypatch.delenv(env_k, raising=False)

        dep_warnings = [
            w for w in recorded
            if issubclass(w.category, DeprecationWarning) and env_k in str(w.message)
        ]
        assert len(dep_warnings) == 1, f"Expected warning for {env_k}"


def test_five_layer_hierarchy_full_chain(tmp_path: Path) -> None:
    """Full 5-layer hierarchy test: engine default < profile < workspace < env < CLI."""
    # 1. Profile sets poc_top_chapters = 10, min_community_size = 5
    prof_dir = tmp_path / "profiles" / "p1"
    prof_dir.mkdir(parents=True)
    (prof_dir / "profile.toml").write_text(
        """schema_version = 1
[pipeline]
poc_top_chapters = 10
min_community_size = 5
""",
        encoding="utf-8",
    )

    # 2. Workspace overrides poc_top_chapters = 20, sets max_chunk_tokens = 600
    ws_dir = tmp_path / "workspaces" / "w1"
    ws_dir.mkdir(parents=True)
    (ws_dir / "input").mkdir()
    (ws_dir / "workspace.toml").write_text(
        """schema_version = 1
name = "chain-test"
[paths]
input = "input"
[profile]
path = "../../profiles/p1"
[pipeline]
poc_top_chapters = 20
max_chunk_tokens = 600
""",
        encoding="utf-8",
    )

    # 3. Env overrides poc_top_chapters = 30 and max_chunk_tokens = 700
    env_overlay = {"POC_TOP_CHAPTERS": "30", "MAX_CHUNK_TOKENS": "700"}

    # 4. CLI overrides poc_top_chapters = 40
    cli_overlay = {"poc_top_chapters": 40}

    resolved = resolve_workspace(
        workspace_dir=ws_dir,
        env_overlay=env_overlay,
        cli_overlay=cli_overlay,
    )

    # poc_top_chapters: CLI (40) > Env (30) > Workspace (20) > Profile (10) > Default (0)
    assert resolved.knobs["poc_top_chapters"] == 40
    assert resolved.provenance["poc_top_chapters"].source == "cli"

    # max_chunk_tokens: Env (700) > Workspace (600) > Default (1000)
    assert resolved.knobs["max_chunk_tokens"] == 700
    assert resolved.provenance["max_chunk_tokens"].source == "env"

    # min_community_size: Profile (5) > Default (8)
    assert resolved.knobs["min_community_size"] == 5
    assert resolved.provenance["min_community_size"].source == "profile"

    # repomap_small_keep_all_below: Default (10)
    assert resolved.knobs["repomap_small_keep_all_below"] == 10
    assert resolved.provenance["repomap_small_keep_all_below"].source == "engine_default"
