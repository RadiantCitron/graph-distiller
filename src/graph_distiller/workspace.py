"""Workspace configuration loader (ROADMAP_0001 step 020).

Pure, read-only loader for workspace.toml and profile configurations.
Validates schemas, forbids secrets and model slugs in configuration files,
resolves relative paths relative to the file defining them, and implements
the 5-layer precedence hierarchy:
  engine defaults < profile < workspace.toml < env < CLI

Does NOT depend on Settings, pipeline orchestration, CLI, or network.
"""

from __future__ import annotations

import logging
import os
import tomllib
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

SUPPORTED_SCHEMA_VERSIONS = {1}

# Engine default knob values (mirroring config.py / ADR-077 baseline)
ENGINE_DEFAULT_KNOBS: Mapping[str, Any] = {
    "max_chunk_tokens": 1000,
    "semantic_breakpoint_percentile": 80,
    "llm_concurrency": 20,
    "max_chapter_retries": 3,
    "poc_top_chapters": 0,
    "louvain_resolution": 0.5,
    "min_community_size": 8,
    "min_chapter_grounding_chunks": 2,
    "repomap_small_keep_all_below": 10,
    "repomap_snippet_chars": 2000,
    "verify_citation_paragraph_ratio": 1.0,
    "verify_citation_sentence_ratio": 0.25,
    "canon_embed_merge": False,
    "canon_embed_sim_threshold": 0.90,
    "canon_embed_max_nodes": 2000,
    "graph_max_nodes": 100000,
    "graph_max_edges": 500000,
    "graph_max_est_mb": 1024.0,
    "graph_scale_on_exceed": "abort",
}

# 19 numerical & boolean knobs: environment variable -> Settings field name (D2, ADR-077)
KNOB_ENV_TO_FIELD: Mapping[str, str] = {
    "MAX_CHUNK_TOKENS": "max_chunk_tokens",
    "SEMANTIC_BREAKPOINT_PERCENTILE": "semantic_breakpoint_percentile",
    "LLM_CONCURRENCY": "llm_concurrency",
    "MAX_CHAPTER_RETRIES": "max_chapter_retries",
    "POC_TOP_CHAPTERS": "poc_top_chapters",
    "LOUVAIN_RESOLUTION": "louvain_resolution",
    "MIN_COMMUNITY_SIZE": "min_community_size",
    "MIN_CHAPTER_GROUNDING_CHUNKS": "min_chapter_grounding_chunks",
    "REPOMAP_SMALL_KEEP_ALL_BELOW": "repomap_small_keep_all_below",
    "REPOMAP_SNIPPET_CHARS": "repomap_snippet_chars",
    "VERIFY_CITATION_PARAGRAPH_RATIO": "verify_citation_paragraph_ratio",
    "VERIFY_CITATION_SENTENCE_RATIO": "verify_citation_sentence_ratio",
    "CANON_EMBED_MERGE": "canon_embed_merge",
    "CANON_EMBED_SIM_THRESHOLD": "canon_embed_sim_threshold",
    "CANON_EMBED_MAX_NODES": "canon_embed_max_nodes",
    "GRAPH_MAX_NODES": "graph_max_nodes",
    "GRAPH_MAX_EDGES": "graph_max_edges",
    "GRAPH_MAX_EST_MB": "graph_max_est_mb",
    "GRAPH_SCALE_ON_EXCEED": "graph_scale_on_exceed",
}

_WARNED_DEPRECATED_KNOBS: Set[str] = set()


def reset_deprecated_knobs_warnings() -> None:
    """Clear the set of warned deprecated knob environment variables (for tests)."""
    _WARNED_DEPRECATED_KNOBS.clear()


def _warn_deprecated_knob(name: str) -> None:
    if name in _WARNED_DEPRECATED_KNOBS:
        return
    _WARNED_DEPRECATED_KNOBS.add(name)
    msg = (
        f"Environment variable '{name}' is deprecated. "
        "Move this knob to [pipeline] in workspace.toml."
    )
    warnings.warn(msg, DeprecationWarning, stacklevel=3)
    logger.warning(msg)


# Standard policy file names expected in control rooms
PROMPT_FILES = (
    "phase2_extraction.md",
    "phase4_planner.md",
    "phase4_synthesizer.md",
    "eval_judge.md",
)

RULE_FILES = (
    "phase2_canonicalize.toml",
    "phase4_repomap.toml",
    "phase4_planner.toml",
    "phase4_verifier.toml",
    "eval_scoring.toml",
)

LEXICON_FILENAME = "entity_aliases.json"

FORBIDDEN_KEY_MESSAGES: Mapping[str, str] = {
    "openrouter_api_key": "Secrets must not be stored in workspace.toml (use environment variables or .env).",
    "api_key": "Secrets must not be stored in workspace.toml (use environment variables or .env).",
    "openrouter_base_url": "Endpoints must not be stored in workspace.toml (use environment variables or .env).",
    "rag_db_uri": "Database connection URIs must not be stored in workspace.toml (use environment variables or .env).",
    "db_uri": "Database connection URIs must not be stored in workspace.toml (use environment variables or .env).",
    "extraction_model": "Model slugs must be specified via CLI (--extraction-model), not in workspace.toml (ADR-053).",
    "synthesis_model": "Model slugs must be specified via CLI (--synthesis-model), not in workspace.toml (ADR-053).",
    "model": "Model configuration is not allowed in workspace.toml (use CLI flags).",
    "budget": "[budget] section is not allowed in workspace.toml (use CLI flags --budget-usd, --token-limit, D10).",
    "budget_usd": "Budget must be specified via CLI (--budget-usd), not in workspace.toml (D10).",
    "token_limit": "Token limit must be specified via CLI (--token-limit), not in workspace.toml (D10).",
    "max_usd": "Budget must be specified via CLI (--budget-usd), not in workspace.toml (D10).",
}


class WorkspaceValidationError(Exception):
    """Raised when a workspace or profile configuration file fails validation."""


def _check_forbidden_keys(raw: Any, file_path: Path, prefix: str = "") -> None:
    """Recursively inspect raw dictionary for forbidden keys before schema parsing."""
    if isinstance(raw, dict):
        for k, v in raw.items():
            k_lower = k.lower()
            if k_lower in FORBIDDEN_KEY_MESSAGES:
                msg = FORBIDDEN_KEY_MESSAGES[k_lower]
                raise WorkspaceValidationError(
                    f"Forbidden key '{prefix}{k}' in {file_path}: {msg}"
                )
            _check_forbidden_keys(v, file_path, prefix=f"{prefix}{k}.")


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: Optional[str] = None
    work: Optional[str] = None
    output: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    codex_output_dir: Optional[str] = None
    errors_log_path: Optional[str] = None
    rag_rebuild_dir: Optional[str] = None
    runs_dir: Optional[str] = None


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompts: Optional[str] = None
    rules: Optional[str] = None
    lexicon: Optional[str] = None


class CorpusRefConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class ProfileRefConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str


class PipelineKnobsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chunk_tokens: Optional[int] = Field(default=None, ge=100, le=10000)
    semantic_breakpoint_percentile: Optional[int] = Field(default=None, ge=1, le=100)
    llm_concurrency: Optional[int] = Field(default=None, ge=1, le=100)
    max_chapter_retries: Optional[int] = Field(default=None, ge=0, le=20)
    poc_top_chapters: Optional[int] = Field(default=None, ge=0, le=1000)
    louvain_resolution: Optional[float] = Field(default=None, ge=0.01, le=10.0)
    min_community_size: Optional[int] = Field(default=None, ge=1, le=100)
    min_chapter_grounding_chunks: Optional[int] = Field(default=None, ge=1, le=50)
    repomap_small_keep_all_below: Optional[int] = Field(default=None, ge=1, le=100)
    repomap_snippet_chars: Optional[int] = Field(default=None, ge=100, le=50000)
    verify_citation_paragraph_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    verify_citation_sentence_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    canon_embed_merge: Optional[bool] = Field(default=None)
    canon_embed_sim_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    canon_embed_max_nodes: Optional[int] = Field(default=None, ge=10, le=100000)
    graph_max_nodes: Optional[int] = Field(default=None, ge=100, le=10000000)
    graph_max_edges: Optional[int] = Field(default=None, ge=100, le=50000000)
    graph_max_est_mb: Optional[float] = Field(default=None, ge=10.0, le=100000.0)
    graph_scale_on_exceed: Optional[str] = Field(default=None, pattern="^(abort|slice)$")


class WorkspaceFileSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    name: Optional[str] = None
    corpus: Optional[CorpusRefConfig] = None
    profile: Optional[ProfileRefConfig] = None
    paths: PathsConfig = Field(default_factory=PathsConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    pipeline: PipelineKnobsConfig = Field(default_factory=PipelineKnobsConfig)


class ProfileFileSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    name: Optional[str] = None
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    pipeline: PipelineKnobsConfig = Field(default_factory=PipelineKnobsConfig)


@dataclass(frozen=True)
class ValueProvenance:
    value: Any
    source: str  # "engine_default" | "profile" | "workspace" | "env" | "cli"
    defined_at: Optional[Path] = None


@dataclass(frozen=True)
class PolicyFileProvenance:
    file_name: str
    resolved_path: Path
    source: str  # "engine_default" | "profile" | "workspace"
    defined_at: Optional[Path] = None


@dataclass(frozen=True)
class ResolvedWorkspace:
    workspace_dir: Optional[Path]
    name: str
    raw_memos_dir: Path
    checkpoint_dir: Path
    codex_output_dir: Path
    errors_log_path: Path
    rag_rebuild_dir: Path
    runs_dir: Path
    prompts_dir: Path
    rules_dir: Path
    canon_alias_lexicon_path: Path
    policy_files: Dict[str, PolicyFileProvenance]
    knobs: Dict[str, Any]
    provenance: Dict[str, ValueProvenance]


def load_workspace_file(file_path: Path) -> WorkspaceFileSchema:
    """Load and validate a workspace.toml file. Does not mutate disk."""
    if not file_path.is_file():
        raise WorkspaceValidationError(f"Workspace file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        raise WorkspaceValidationError(f"Failed to parse TOML in {file_path}: {exc}") from exc

    _check_forbidden_keys(data, file_path)

    ver = data.get("schema_version")
    if ver not in SUPPORTED_SCHEMA_VERSIONS:
        raise WorkspaceValidationError(
            f"Unsupported schema_version {ver!r} in {file_path}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    try:
        return WorkspaceFileSchema.model_validate(data)
    except ValidationError as exc:
        raise WorkspaceValidationError(f"Invalid workspace schema in {file_path}: {exc}") from exc


def load_profile_file(file_path: Path) -> ProfileFileSchema:
    """Load and validate a profile.toml file. Does not mutate disk."""
    if not file_path.is_file():
        raise WorkspaceValidationError(f"Profile file not found: {file_path}")

    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
    except Exception as exc:
        raise WorkspaceValidationError(f"Failed to parse TOML in {file_path}: {exc}") from exc

    _check_forbidden_keys(data, file_path)

    # 1-level inheritance constraint: profile cannot reference another profile or corpus
    if "profile" in data or "corpus" in data:
        raise WorkspaceValidationError(
            f"Profile at {file_path} cannot contain [profile] or [corpus] (only 1 level of inheritance supported)"
        )

    ver = data.get("schema_version", 1)
    if ver not in SUPPORTED_SCHEMA_VERSIONS:
        raise WorkspaceValidationError(
            f"Unsupported schema_version {ver!r} in {file_path}; supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )

    try:
        return ProfileFileSchema.model_validate(data)
    except ValidationError as exc:
        raise WorkspaceValidationError(f"Invalid profile schema in {file_path}: {exc}") from exc


def _resolve_relative(base_dir: Path, target: str | Path | None) -> Optional[Path]:
    if target is None:
        return None
    p = Path(target)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def resolve_workspace(
    workspace_dir: Optional[Path] = None,
    *,
    project_root: Optional[Path] = None,
    env_overlay: Optional[Mapping[str, str]] = None,
    cli_overlay: Optional[Mapping[str, Any]] = None,
) -> ResolvedWorkspace:
    """Resolve complete workspace configuration applying the 5-layer hierarchy.

    Hierarchy: engine defaults < profile < workspace.toml < env < CLI

    This is a pure function. It reads filesystem paths for policy file presence
    but writes NOTHING to disk.
    """
    root = project_root or Path(__file__).resolve().parents[2]
    env_map = os.environ if env_overlay is None else env_overlay
    cli_map = cli_overlay or {}

    provenance: Dict[str, ValueProvenance] = {}
    knobs: Dict[str, Any] = {}

    # Layer 1: Engine defaults
    for k, v in ENGINE_DEFAULT_KNOBS.items():
        knobs[k] = v
        provenance[k] = ValueProvenance(value=v, source="engine_default", defined_at=None)

    default_raw_memos = root / "data" / "raw_memos"
    default_checkpoints = root / "data" / "checkpoints"
    default_codex = root / "output" / "codex"
    default_errors = root / "output" / "errors.log"
    default_rag_rebuild = root / "output" / "rag_rebuild"
    default_runs = root / "output" / "runs"
    default_prompts = root / "data" / "prompts"
    default_rules = root / "data" / "rules"
    default_lexicon = root / "data" / "lexicon" / "entity_aliases.json"

    raw_memos_dir = default_raw_memos
    checkpoint_dir = default_checkpoints
    codex_output_dir = default_codex
    errors_log_path = default_errors
    rag_rebuild_dir = default_rag_rebuild
    runs_dir = default_runs

    provenance["raw_memos_dir"] = ValueProvenance(raw_memos_dir, "engine_default")
    provenance["checkpoint_dir"] = ValueProvenance(checkpoint_dir, "engine_default")
    provenance["codex_output_dir"] = ValueProvenance(codex_output_dir, "engine_default")
    provenance["errors_log_path"] = ValueProvenance(errors_log_path, "engine_default")
    provenance["rag_rebuild_dir"] = ValueProvenance(rag_rebuild_dir, "engine_default")
    provenance["runs_dir"] = ValueProvenance(runs_dir, "engine_default")

    # Policy files tracking: file_name -> PolicyFileProvenance
    policy_files: Dict[str, PolicyFileProvenance] = {}
    for p_name in PROMPT_FILES:
        p_path = default_prompts / p_name
        policy_files[p_name] = PolicyFileProvenance(p_name, p_path, "engine_default", p_path)
    for r_name in RULE_FILES:
        r_path = default_rules / r_name
        policy_files[r_name] = PolicyFileProvenance(r_name, r_path, "engine_default", r_path)
    policy_files[LEXICON_FILENAME] = PolicyFileProvenance(
        LEXICON_FILENAME, default_lexicon, "engine_default", default_lexicon
    )

    ws_schema: Optional[WorkspaceFileSchema] = None
    ws_file: Optional[Path] = None
    workspace_name = "default"

    profile_dir: Optional[Path] = None
    profile_schema: Optional[ProfileFileSchema] = None

    if workspace_dir is not None:
        ws_dir = workspace_dir.resolve()
        ws_file = ws_dir / "workspace.toml"
        ws_schema = load_workspace_file(ws_file)
        workspace_name = ws_schema.name or ws_dir.name

        # Layer 2: Profile (if specified by workspace)
        if ws_schema.profile is not None and ws_schema.profile.path:
            profile_dir = _resolve_relative(ws_dir, ws_schema.profile.path)
            if profile_dir is not None and profile_dir.is_dir():
                prof_file = profile_dir / "profile.toml"
                if prof_file.is_file():
                    profile_schema = load_profile_file(prof_file)
                    # Apply profile knobs
                    for k, v in profile_schema.pipeline.model_dump(exclude_none=True).items():
                        knobs[k] = v
                        provenance[k] = ValueProvenance(value=v, source="profile", defined_at=prof_file)

                # Check policy files in profile
                prof_prompts_dir = profile_dir / "prompts"
                if profile_schema and profile_schema.policy.prompts:
                    prof_prompts_dir = _resolve_relative(profile_dir, profile_schema.policy.prompts) or prof_prompts_dir
                for p_name in PROMPT_FILES:
                    candidate = prof_prompts_dir / p_name
                    if candidate.is_file():
                        policy_files[p_name] = PolicyFileProvenance(p_name, candidate, "profile", candidate)

                prof_rules_dir = profile_dir / "rules"
                if profile_schema and profile_schema.policy.rules:
                    prof_rules_dir = _resolve_relative(profile_dir, profile_schema.policy.rules) or prof_rules_dir
                for r_name in RULE_FILES:
                    candidate = prof_rules_dir / r_name
                    if candidate.is_file():
                        policy_files[r_name] = PolicyFileProvenance(r_name, candidate, "profile", candidate)

                prof_lexicon_file = None
                if profile_schema and profile_schema.policy.lexicon:
                    prof_lexicon_file = _resolve_relative(profile_dir, profile_schema.policy.lexicon)
                if prof_lexicon_file is None:
                    for cand in (
                        profile_dir / "lexicon.json",
                        profile_dir / "entity_aliases.json",
                        profile_dir / "lexicon" / "entity_aliases.json",
                    ):
                        if cand.is_file():
                            prof_lexicon_file = cand
                            break
                if prof_lexicon_file and prof_lexicon_file.is_file():
                    policy_files[LEXICON_FILENAME] = PolicyFileProvenance(
                        LEXICON_FILENAME, prof_lexicon_file, "profile", prof_lexicon_file
                    )

        # Layer 3: workspace.toml
        # Resolve paths
        paths_cfg = ws_schema.paths
        base_input = _resolve_relative(ws_dir, paths_cfg.input) if paths_cfg.input else None
        if base_input is None and ws_schema.corpus is not None and ws_schema.corpus.path:
            corpus_dir = _resolve_relative(ws_dir, ws_schema.corpus.path)
            if corpus_dir is not None:
                corpus_input = corpus_dir / "input"
                base_input = corpus_input if corpus_input.is_dir() else corpus_dir

        raw_memos_dir = base_input or (ws_dir / "input")
        provenance["raw_memos_dir"] = ValueProvenance(raw_memos_dir, "workspace", ws_file)

        base_work = _resolve_relative(ws_dir, paths_cfg.work) if paths_cfg.work else (ws_dir / "work")
        checkpoint_dir = _resolve_relative(ws_dir, paths_cfg.checkpoint_dir) if paths_cfg.checkpoint_dir else base_work
        provenance["checkpoint_dir"] = ValueProvenance(checkpoint_dir, "workspace", ws_file)

        base_output = _resolve_relative(ws_dir, paths_cfg.output) if paths_cfg.output else (ws_dir / "output")
        codex_output_dir = (
            _resolve_relative(ws_dir, paths_cfg.codex_output_dir)
            if paths_cfg.codex_output_dir
            else (base_output / "codex")
        )
        errors_log_path = (
            _resolve_relative(ws_dir, paths_cfg.errors_log_path)
            if paths_cfg.errors_log_path
            else (base_output / "errors.log")
        )
        rag_rebuild_dir = (
            _resolve_relative(ws_dir, paths_cfg.rag_rebuild_dir)
            if paths_cfg.rag_rebuild_dir
            else (base_output / "rag_rebuild")
        )
        runs_dir = (
            _resolve_relative(ws_dir, paths_cfg.runs_dir)
            if paths_cfg.runs_dir
            else (base_output / "runs")
        )

        provenance["codex_output_dir"] = ValueProvenance(codex_output_dir, "workspace", ws_file)
        provenance["errors_log_path"] = ValueProvenance(errors_log_path, "workspace", ws_file)
        provenance["rag_rebuild_dir"] = ValueProvenance(rag_rebuild_dir, "workspace", ws_file)
        provenance["runs_dir"] = ValueProvenance(runs_dir, "workspace", ws_file)

        # Apply workspace policy file overrides (file-by-file)
        ws_prompts_dir = _resolve_relative(ws_dir, ws_schema.policy.prompts) if ws_schema.policy.prompts else (ws_dir / "prompts")
        for p_name in PROMPT_FILES:
            candidate = ws_prompts_dir / p_name
            if candidate.is_file():
                policy_files[p_name] = PolicyFileProvenance(p_name, candidate, "workspace", candidate)

        ws_rules_dir = _resolve_relative(ws_dir, ws_schema.policy.rules) if ws_schema.policy.rules else (ws_dir / "rules")
        for r_name in RULE_FILES:
            candidate = ws_rules_dir / r_name
            if candidate.is_file():
                policy_files[r_name] = PolicyFileProvenance(r_name, candidate, "workspace", candidate)

        ws_lexicon_file = None
        if ws_schema.policy.lexicon:
            ws_lexicon_file = _resolve_relative(ws_dir, ws_schema.policy.lexicon)
        if ws_lexicon_file is None:
            for cand in (
                ws_dir / "lexicon.json",
                ws_dir / "entity_aliases.json",
                ws_dir / "lexicon" / "entity_aliases.json",
            ):
                if cand.is_file():
                    ws_lexicon_file = cand
                    break
        if ws_lexicon_file and ws_lexicon_file.is_file():
            policy_files[LEXICON_FILENAME] = PolicyFileProvenance(
                LEXICON_FILENAME, ws_lexicon_file, "workspace", ws_lexicon_file
            )

        # Apply workspace knobs
        for k, v in ws_schema.pipeline.model_dump(exclude_none=True).items():
            knobs[k] = v
            provenance[k] = ValueProvenance(value=v, source="workspace", defined_at=ws_file)

    # Layer 4: Environment overlay
    env_path_mappings = {
        "RAW_MEMOS_DIR": "raw_memos_dir",
        "CHECKPOINT_DIR": "checkpoint_dir",
        "CODEX_OUTPUT_DIR": "codex_output_dir",
        "ERRORS_LOG_PATH": "errors_log_path",
        "RAG_REBUILD_DIR": "rag_rebuild_dir",
    }
    for env_k, prop_name in env_path_mappings.items():
        if env_k in env_map and env_map[env_k]:
            resolved_p = Path(env_map[env_k]).resolve()
            if prop_name == "raw_memos_dir":
                raw_memos_dir = resolved_p
            elif prop_name == "checkpoint_dir":
                checkpoint_dir = resolved_p
            elif prop_name == "codex_output_dir":
                codex_output_dir = resolved_p
            elif prop_name == "errors_log_path":
                errors_log_path = resolved_p
            elif prop_name == "rag_rebuild_dir":
                rag_rebuild_dir = resolved_p
            provenance[prop_name] = ValueProvenance(resolved_p, "env")

    for env_k, prop_name in KNOB_ENV_TO_FIELD.items():
        if env_k in env_map and env_map[env_k] is not None and str(env_map[env_k]).strip() != "":
            raw_v = str(env_map[env_k]).strip().strip('"').strip("'")
            _warn_deprecated_knob(env_k)
            try:
                parsed = PipelineKnobsConfig(**{prop_name: raw_v})
                v = getattr(parsed, prop_name)
                knobs[prop_name] = v
                provenance[prop_name] = ValueProvenance(value=v, source="env")
            except ValidationError as exc:
                raise WorkspaceValidationError(
                    f"Invalid environment variable value {env_k}={raw_v!r}: {exc}"
                ) from exc

    # Layer 5: CLI overlay
    if "output_dir" in cli_map and cli_map["output_dir"]:
        cli_runs_dir = Path(cli_map["output_dir"]).resolve()
        runs_dir = cli_runs_dir
        provenance["runs_dir"] = ValueProvenance(cli_runs_dir, "cli")

    if "poc_top_chapters" in cli_map and cli_map["poc_top_chapters"] is not None:
        knobs["poc_top_chapters"] = int(cli_map["poc_top_chapters"])
        provenance["poc_top_chapters"] = ValueProvenance(
            value=knobs["poc_top_chapters"], source="cli"
        )

    # Derived policy directories from policy files (or primary directory)
    prompts_dir = (
        _resolve_relative(workspace_dir, ws_schema.policy.prompts)
        if (workspace_dir and ws_schema and ws_schema.policy.prompts)
        else (workspace_dir / "prompts" if workspace_dir and (workspace_dir / "prompts").is_dir() else default_prompts)
    )
    rules_dir = (
        _resolve_relative(workspace_dir, ws_schema.policy.rules)
        if (workspace_dir and ws_schema and ws_schema.policy.rules)
        else (workspace_dir / "rules" if workspace_dir and (workspace_dir / "rules").is_dir() else default_rules)
    )
    canon_alias_lexicon_path = policy_files[LEXICON_FILENAME].resolved_path

    return ResolvedWorkspace(
        workspace_dir=workspace_dir.resolve() if workspace_dir else None,
        name=workspace_name,
        raw_memos_dir=raw_memos_dir,
        checkpoint_dir=checkpoint_dir,
        codex_output_dir=codex_output_dir,
        errors_log_path=errors_log_path,
        rag_rebuild_dir=rag_rebuild_dir,
        runs_dir=runs_dir,
        prompts_dir=prompts_dir,
        rules_dir=rules_dir,
        canon_alias_lexicon_path=canon_alias_lexicon_path,
        policy_files=policy_files,
        knobs=knobs,
        provenance=provenance,
    )
