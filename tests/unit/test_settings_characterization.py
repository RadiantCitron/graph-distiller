"""Unit: Settings characterization tests (ROADMAP_0001 step 010 & 030).

Freezes default values of Settings fields, hardcoded path constants,
environment variable override behavior, explicit directory creation (D11),
and verifies Settings-routed runs_dir / reports_dir ($0, no network).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from graph_distiller.config import (
    DEFAULT_EMBEDDING_MODEL,
    OPENROUTER_BASE_URL,
    PROJECT_ROOT,
    Settings,
    get_settings,
    set_runtime_settings,
)
from graph_distiller.rag_export import DEFAULT_RAG_REBUILD_ROOT, create_export_dir
from graph_distiller.run_ledger import DEFAULT_RUNS_ROOT, create_run_directory

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
    "MAX_CHUNK_TOKENS",
    "SEMANTIC_BREAKPOINT_PERCENTILE",
    "LLM_CONCURRENCY",
    "MAX_CHAPTER_RETRIES",
    "POC_TOP_CHAPTERS",
    "LOUVAIN_RESOLUTION",
    "MIN_COMMUNITY_SIZE",
    "MIN_CHAPTER_GROUNDING_CHUNKS",
    "REPOMAP_SMALL_KEEP_ALL_BELOW",
    "REPOMAP_SNIPPET_CHARS",
    "VERIFY_CITATION_PARAGRAPH_RATIO",
    "VERIFY_CITATION_SENTENCE_RATIO",
    "CANON_EMBED_MERGE",
    "CANON_EMBED_SIM_THRESHOLD",
    "CANON_EMBED_MAX_NODES",
    "GRAPH_MAX_NODES",
    "GRAPH_MAX_EDGES",
    "GRAPH_MAX_EST_MB",
    "GRAPH_SCALE_ON_EXCEED",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Clear all Settings-related environment variables for test isolation."""
    for key in _ENV_KEYS_TO_CLEAR:
        monkeypatch.delenv(key, raising=False)
    yield


def test_settings_total_field_count() -> None:
    """Settings has exactly 36 fields: 2 credentials/endpoint + 3 models + 10 paths + 1 db + 19 knobs + 1 workspace_dir."""
    from dataclasses import fields

    all_fields = fields(Settings)
    assert len(all_fields) == 36


def test_default_path_fields_relative_to_project_root(
    clean_env: None,
) -> None:
    """1. All 10 path fields match current defaults relative to PROJECT_ROOT when env is clear."""
    s = Settings.from_env(models_required=False)

    assert s.raw_memos_dir == PROJECT_ROOT / "data" / "raw_memos"
    assert s.checkpoint_dir == PROJECT_ROOT / "data" / "checkpoints"
    assert s.codex_output_dir == PROJECT_ROOT / "output" / "codex"
    assert s.errors_log_path == PROJECT_ROOT / "output" / "errors.log"
    assert s.prompts_dir == PROJECT_ROOT / "data" / "prompts"
    assert s.rules_dir == PROJECT_ROOT / "data" / "rules"
    assert s.rag_rebuild_dir == PROJECT_ROOT / "output" / "rag_rebuild"
    assert s.runs_dir == PROJECT_ROOT / "output" / "runs"
    assert s.reports_dir == PROJECT_ROOT / "tests" / "reports"
    assert s.canon_alias_lexicon_path == (
        PROJECT_ROOT / "data" / "lexicon" / "entity_aliases.json"
    )


def test_default_numerical_and_boolean_knobs(
    clean_env: None,
) -> None:
    """2. All 19 numerical and boolean knobs freeze at their exact default values."""
    s = Settings.from_env(models_required=False)

    assert s.max_chunk_tokens == 1000
    assert s.semantic_breakpoint_percentile == 80
    assert s.llm_concurrency == 20
    assert s.max_chapter_retries == 3
    assert s.poc_top_chapters == 0
    assert s.louvain_resolution == 0.5
    assert s.min_community_size == 8
    assert s.min_chapter_grounding_chunks == 2
    assert s.repomap_small_keep_all_below == 10
    assert s.repomap_snippet_chars == 2000
    assert s.verify_citation_paragraph_ratio == 1.0
    assert s.verify_citation_sentence_ratio == 0.25
    assert s.canon_embed_merge is False
    assert s.canon_embed_sim_threshold == 0.90
    assert s.canon_embed_max_nodes == 2000
    assert s.graph_max_nodes == 100000
    assert s.graph_max_edges == 500000
    assert s.graph_max_est_mb == 1024.0
    assert s.graph_scale_on_exceed == "abort"

    # Other non-path, non-knob baseline fields
    assert s.openrouter_base_url == OPENROUTER_BASE_URL
    assert s.embedding_model == DEFAULT_EMBEDDING_MODEL
    assert s.rag_db_uri == ""


def test_path_env_var_overrides(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """3. Pointing path environment variables to custom paths overrides those fields."""
    custom_memos = tmp_path / "custom_memos"
    custom_checkpoints = tmp_path / "custom_checkpoints"
    custom_codex = tmp_path / "custom_codex"
    custom_errors = tmp_path / "logs" / "custom_errors.log"
    custom_prompts = tmp_path / "custom_prompts"
    custom_rules = tmp_path / "custom_rules"
    custom_rebuild = tmp_path / "custom_rebuild"
    custom_runs = tmp_path / "custom_runs"
    custom_reports = tmp_path / "custom_reports"
    custom_lexicon = tmp_path / "lexicon" / "custom_aliases.json"

    monkeypatch.setenv("RAW_MEMOS_DIR", str(custom_memos))
    monkeypatch.setenv("CHECKPOINT_DIR", str(custom_checkpoints))
    monkeypatch.setenv("CODEX_OUTPUT_DIR", str(custom_codex))
    monkeypatch.setenv("ERRORS_LOG_PATH", str(custom_errors))
    monkeypatch.setenv("PROMPTS_DIR", str(custom_prompts))
    monkeypatch.setenv("RULES_DIR", str(custom_rules))
    monkeypatch.setenv("RAG_REBUILD_DIR", str(custom_rebuild))
    monkeypatch.setenv("RUNS_DIR", str(custom_runs))
    monkeypatch.setenv("REPORTS_DIR", str(custom_reports))
    monkeypatch.setenv("CANON_ALIAS_LEXICON_PATH", str(custom_lexicon))

    s = Settings.from_env(models_required=False)

    assert s.raw_memos_dir == custom_memos
    assert s.checkpoint_dir == custom_checkpoints
    assert s.codex_output_dir == custom_codex
    assert s.errors_log_path == custom_errors
    assert s.prompts_dir == custom_prompts
    assert s.rules_dir == custom_rules
    assert s.rag_rebuild_dir == custom_rebuild
    assert s.runs_dir == custom_runs
    assert s.reports_dir == custom_reports
    assert s.canon_alias_lexicon_path == custom_lexicon


def test_hardcoded_path_constants_match_expected() -> None:
    """4. Hardcoded path constants in run_ledger and rag_export match expected defaults."""
    assert DEFAULT_RUNS_ROOT == PROJECT_ROOT / "output" / "runs"
    assert DEFAULT_RAG_REBUILD_ROOT == PROJECT_ROOT / "output" / "rag_rebuild"


def test_d11_ensure_directories_not_called_eagerly_on_from_env(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """5. D11: Settings.from_env() does NOT eagerly create directories on import/creation.

    Directories are only created when ensure_directories() is explicitly invoked.
    """
    custom_memos = tmp_path / "isolated_memos"
    custom_checkpoints = tmp_path / "isolated_checkpoints"
    custom_codex = tmp_path / "isolated_codex"
    custom_errors = tmp_path / "logs" / "isolated_errors.log"
    custom_rebuild = tmp_path / "isolated_rebuild"
    custom_lexicon = tmp_path / "lexicon" / "isolated_aliases.json"

    monkeypatch.setenv("RAW_MEMOS_DIR", str(custom_memos))
    monkeypatch.setenv("CHECKPOINT_DIR", str(custom_checkpoints))
    monkeypatch.setenv("CODEX_OUTPUT_DIR", str(custom_codex))
    monkeypatch.setenv("ERRORS_LOG_PATH", str(custom_errors))
    monkeypatch.setenv("RAG_REBUILD_DIR", str(custom_rebuild))
    monkeypatch.setenv("CANON_ALIAS_LEXICON_PATH", str(custom_lexicon))

    # from_env should NOT create directories (D11 verified)
    s = Settings.from_env(models_required=False)
    assert not s.raw_memos_dir.exists()
    assert not s.checkpoint_dir.exists()
    assert not s.codex_output_dir.exists()

    # Explicit call must create the expected set
    s.ensure_directories()
    assert s.raw_memos_dir.is_dir()
    assert s.checkpoint_dir.is_dir()
    assert s.codex_output_dir.is_dir()
    assert s.errors_log_path.parent.is_dir()
    assert s.rag_rebuild_dir.is_dir()
    assert s.canon_alias_lexicon_path.parent.is_dir()


def test_hardcoded_paths_routed_through_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Step 030 verification: create_run_directory and create_export_dir use Settings."""
    custom_runs = tmp_path / "my_custom_runs"
    custom_rebuild = tmp_path / "my_custom_rebuild"
    custom_reports = tmp_path / "my_custom_reports"

    monkeypatch.setenv("RUNS_DIR", str(custom_runs))
    monkeypatch.setenv("RAG_REBUILD_DIR", str(custom_rebuild))
    monkeypatch.setenv("REPORTS_DIR", str(custom_reports))

    test_settings = Settings.from_env(models_required=False)
    set_runtime_settings(test_settings)

    # 1. create_run_directory without explicit runs_root uses Settings.runs_dir
    run_dir = create_run_directory()
    assert run_dir.parent == custom_runs.resolve()

    # 2. create_export_dir without explicit root uses Settings.rag_rebuild_dir
    export_dir = create_export_dir()
    assert export_dir.parent == custom_rebuild.resolve()

    # 3. get_settings().reports_dir is available
    assert get_settings().reports_dir == custom_reports.resolve()
