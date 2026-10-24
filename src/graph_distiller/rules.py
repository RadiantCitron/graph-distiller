"""Rule control room loader (ADR-060): data/rules/*.toml → validated schemas.

Mirrors the separation already used for prompts (prompts.py) and the alias
lexicon (graph/lexicon.py): this module only turns files into validated
Python objects. It does not know about Settings, LLM calls, or the pipeline.

Loading policy (deliberately asymmetric, see ADR-060 Decision 4):
- File missing              -> in-code default, logged at WARNING. Safe no-op,
                                 same as the prompt/lexicon loaders.
- File present but invalid  -> raises RulesValidationError immediately. A
                                 wrong prompt sentence still means *something*;
                                 a wrong regex or an out-of-range ratio can
                                 silently corrupt every chapter, so this must
                                 surface before any LLM call is made.
"""

from __future__ import annotations

import logging
import re
import tomllib
from pathlib import Path
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# Must equal agents.repomap.REPOMAP_TOKEN_BUDGET (W4-B hard ceiling). Kept as a
# literal here, not an import, to avoid rules.py <-> agents.repomap circularity;
# tests/unit/test_rules.py asserts the two stay equal.
REPOMAP_HARD_TOKEN_CEILING = 80_000


class RulesValidationError(Exception):
    """A data/rules/*.toml file exists but fails its schema. Fail fast, before any LLM call."""


def _compiled(pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
    return pattern


class CanonicalizeRules(BaseModel):
    """graph/extraction.py::_are_alias_ids — same-concept merge thresholds."""

    model_config = ConfigDict(extra="forbid")

    min_fingerprint_len: int = Field(default=4, ge=1, le=64)
    min_token_len: int = Field(default=2, ge=1, le=64)
    jaccard_threshold: float = Field(default=0.8, ge=0.0, le=1.0)


class RepomapRules(BaseModel):
    """agents/repomap.py — Part 1/Part 2 sizing. token_budget may only be lowered (ADR-060)."""

    model_config = ConfigDict(extra="forbid")

    top_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    grounding_budget_fraction: float = Field(default=0.90, gt=0.0, le=1.0)
    token_budget: int = Field(default=REPOMAP_HARD_TOKEN_CEILING, ge=1000, le=REPOMAP_HARD_TOKEN_CEILING)


class PlannerRules(BaseModel):
    """agents/planner.py — schema-leak detection + title length cap."""

    model_config = ConfigDict(extra="forbid")

    leak_markers: List[str] = Field(default_factory=lambda: ["chaptertitleresult", "basemodel", "field(description"])
    max_title_words: int = Field(default=5, ge=1, le=20)

    @field_validator("leak_markers")
    @classmethod
    def _non_empty_markers(cls, v: List[str]) -> List[str]:
        if any(not m.strip() for m in v):
            raise ValueError("leak_markers entries must be non-empty strings")
        return v


class VerifierRules(BaseModel):
    """agents/verifier.py — citation density gate + corruption detectors."""

    model_config = ConfigDict(extra="forbid")

    min_citations: int = Field(default=1, ge=0, le=100)
    sentence_split_pattern: str = Field(default=r"(?<=[.!?。])\s+|(?<=다\.)\s+")
    check_markdown_fences_balanced: bool = True

    @field_validator("sentence_split_pattern")
    @classmethod
    def _valid_regex(cls, v: str) -> str:
        return _compiled(v)


class EvalScoringRules(BaseModel):
    """tests/e2e/run_e2e_eval.py + tests/harness/scoring.py — Prompt Diet judge
    post-processing: diet_score weights, next-loop target formula, truncation
    lengths. The judge's own instructions live in data/prompts/eval_judge.md
    (ADR-059); this file governs what the harness does with the judge's numbers."""

    model_config = ConfigDict(extra="forbid")

    compression_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    retention_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    next_ratio_step_pct: float = Field(default=2.0, ge=0.0, le=100.0)
    next_ratio_cap_pct: float = Field(default=99.0, ge=0.0, le=100.0)
    next_retention_target_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    next_score_step: int = Field(default=1, ge=0, le=10)
    feedback_max_lines: int = Field(default=3, ge=1, le=50)
    error_log_tail_lines: int = Field(default=40, ge=0, le=10_000)
    judge_input_max_chars: int = Field(default=40_000, ge=1_000, le=1_000_000)


class Rules(BaseModel):
    """One validated snapshot of data/rules/*.toml, loaded once per process."""

    model_config = ConfigDict(extra="forbid")

    canonicalize: CanonicalizeRules = Field(default_factory=CanonicalizeRules)
    repomap: RepomapRules = Field(default_factory=RepomapRules)
    planner: PlannerRules = Field(default_factory=PlannerRules)
    verifier: VerifierRules = Field(default_factory=VerifierRules)
    eval_scoring: EvalScoringRules = Field(default_factory=EvalScoringRules)


_FILES: dict = {
    "canonicalize": ("phase2_canonicalize.toml", CanonicalizeRules),
    "repomap": ("phase4_repomap.toml", RepomapRules),
    "planner": ("phase4_planner.toml", PlannerRules),
    "verifier": ("phase4_verifier.toml", VerifierRules),
    "eval_scoring": ("eval_scoring.toml", EvalScoringRules),
}


def _load_one(rules_dir: Path, field_name: str) -> BaseModel:
    filename, schema = _FILES[field_name]
    path = Path(rules_dir) / filename
    default = schema()
    if not path.is_file():
        logger.warning("Rule file not found: %s — using in-code default.", path)
        return default
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RulesValidationError(f"{path}: invalid TOML: {exc}") from exc
    try:
        return schema.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError, re-raised with the file path attached
        raise RulesValidationError(f"{path}: {exc}") from exc


def load_rules(rules_dir: Path) -> Rules:
    """Load+validate every data/rules/*.toml file. Raises RulesValidationError on the first bad file."""
    return Rules(**{name: _load_one(rules_dir, name) for name in _FILES})


_runtime_rules: Rules | None = None


def get_rules(rules_dir: Path | None = None) -> Rules:
    """Process-scoped Rules, loaded once and cached (mirrors config.get_settings()).

    rules_dir defaults to Settings.rules_dir so low-level helpers (e.g.
    agents.repomap.resolve_repomap_token_budget) can call this with no
    argument, exactly like they already call config.get_settings()."""
    global _runtime_rules
    if _runtime_rules is None:
        if rules_dir is None:
            from graph_distiller.config import get_settings

            rules_dir = get_settings().rules_dir
        _runtime_rules = load_rules(rules_dir)
    return _runtime_rules


def set_runtime_rules(rules: Rules) -> None:
    """Bind Rules for this process (tests / explicit reload). Does not write files."""
    global _runtime_rules
    _runtime_rules = rules
