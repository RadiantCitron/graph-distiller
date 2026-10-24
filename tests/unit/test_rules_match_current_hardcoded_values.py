"""Unit: data/rules/*.toml must encode the SAME values the code used before ADR-060.

This is the safety net for the externalization itself: it proves the
control-room files are a lossless copy, not an accidental behavior change,
independent of whether any product module has been wired to read them yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.agents.planner import _LEAK_MARKERS
from graph_distiller.agents.repomap import GROUNDING_BUDGET_FRACTION, REPOMAP_TOKEN_BUDGET, REPOMAP_TOP_FRACTION
from graph_distiller.agents.verifier import SENTENCE_RE
from graph_distiller.rules import Rules, load_rules

pytestmark = pytest.mark.unit

RULES_DIR = Path(__file__).resolve().parents[2] / "data" / "rules"


@pytest.fixture(scope="module")
def rules() -> Rules:
    return load_rules(RULES_DIR)


def test_rules_directory_has_all_five_files() -> None:
    for name in (
        "phase2_canonicalize.toml",
        "phase4_repomap.toml",
        "phase4_planner.toml",
        "phase4_verifier.toml",
        "eval_scoring.toml",
    ):
        assert (RULES_DIR / name).is_file(), name


def test_canonicalize_matches_hardcoded_are_alias_ids_thresholds(rules: Rules) -> None:
    # graph/extraction.py::_are_alias_ids literals: fingerprint len >= 4, token len >= 2, jaccard >= 0.8
    assert rules.canonicalize.min_fingerprint_len == 4
    assert rules.canonicalize.min_token_len == 2
    assert rules.canonicalize.jaccard_threshold == 0.8


def test_repomap_matches_hardcoded_module_constants(rules: Rules) -> None:
    assert rules.repomap.top_fraction == REPOMAP_TOP_FRACTION
    assert rules.repomap.grounding_budget_fraction == GROUNDING_BUDGET_FRACTION
    assert rules.repomap.token_budget == REPOMAP_TOKEN_BUDGET


def test_planner_matches_hardcoded_leak_markers_and_word_cap(rules: Rules) -> None:
    assert tuple(rules.planner.leak_markers) == _LEAK_MARKERS
    assert rules.planner.max_title_words == 5  # coerce_chapter_title: len(words) > 5


def test_verifier_matches_hardcoded_sentence_pattern_and_min_citations(rules: Rules) -> None:
    assert rules.verifier.sentence_split_pattern == SENTENCE_RE.pattern
    assert rules.verifier.min_citations == 1
    assert rules.verifier.check_markdown_fences_balanced is True


def test_eval_scoring_matches_hardcoded_run_e2e_eval_literals(rules: Rules) -> None:
    # tests/e2e/run_e2e_eval.py literals: _clip_feedback(max_lines=3), _tail_errors_log(max_lines=40),
    # repo_text[:40_000], next_loop_targets (+2.0 capped at 99.0, retention target 100.0, +1 score step),
    # diet_score = compression_score + information_retention (implicit 1:1 weight).
    s = rules.eval_scoring
    assert s.compression_weight == 1.0
    assert s.retention_weight == 1.0
    assert s.next_ratio_step_pct == 2.0
    assert s.next_ratio_cap_pct == 99.0
    assert s.next_retention_target_pct == 100.0
    assert s.next_score_step == 1
    assert s.feedback_max_lines == 3
    assert s.error_log_tail_lines == 40
    assert s.judge_input_max_chars == 40_000
