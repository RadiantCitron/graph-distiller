"""Unit: tests/harness/scoring.py — Prompt Diet post-processing, $0."""

from __future__ import annotations

import pytest

from harness.scoring import (
    clip_feedback,
    diet_score,
    diet_score_max,
    fallback_scores,
    next_loop_targets,
    tail_lines,
    truncate_judge_input,
)
from graph_distiller.rules import EvalScoringRules

pytestmark = pytest.mark.unit


def test_default_rules_reproduce_run_e2e_eval_defaults() -> None:
    rules = EvalScoringRules()
    assert diet_score(7, 9, rules) == 16.0
    assert diet_score_max(rules) == 20.0


def test_clip_feedback_truncates_at_configured_line_count() -> None:
    rules = EvalScoringRules(feedback_max_lines=2)
    assert clip_feedback("a\nb\nc\nd", rules) == "a\nb"


def test_clip_feedback_keeps_short_text_untouched() -> None:
    rules = EvalScoringRules()
    assert clip_feedback("only one line", rules) == "only one line"


def test_truncate_judge_input_respects_configured_char_limit() -> None:
    rules = EvalScoringRules(judge_input_max_chars=1000)
    text = "x" * 2000
    out = truncate_judge_input(text, rules)
    assert out.startswith("x" * 1000)
    assert "[truncated]" in out
    assert len(out) < len(text)


def test_truncate_judge_input_leaves_short_text_alone() -> None:
    rules = EvalScoringRules()
    assert truncate_judge_input("short", rules) == "short"


def test_tail_lines_respects_configured_count() -> None:
    rules = EvalScoringRules(error_log_tail_lines=2)
    assert tail_lines(["a", "b", "c"], rules) == ["b", "c"]


def test_tail_lines_zero_returns_empty() -> None:
    rules = EvalScoringRules(error_log_tail_lines=0)
    assert tail_lines(["a", "b"], rules) == []


def test_fallback_scores_matches_original_div_by_ten_rounding() -> None:
    assert fallback_scores(0.0, 0.0) == (1, 1)  # `or 1` floor for zero
    assert fallback_scores(35.0, 100.0) == (4, 10)
    assert fallback_scores(999.0, 999.0) == (10, 10)  # clamped


def test_next_loop_targets_pushes_ratio_when_retention_is_full() -> None:
    rules = EvalScoringRules()
    out = next_loop_targets(
        ratio_pct=50.0, retention_pct=100.0, compression_score=5, information_retention=10, rules=rules
    )
    assert out["min_compression_ratio_pct"] == 52.0
    assert out["min_core_retention_pct"] == 100.0
    assert out["min_information_retention"] == 10


def test_next_loop_targets_freezes_ratio_when_retention_regressed() -> None:
    rules = EvalScoringRules()
    out = next_loop_targets(
        ratio_pct=50.0, retention_pct=80.0, compression_score=5, information_retention=8, rules=rules
    )
    assert out["min_compression_ratio_pct"] == 50.0  # frozen, not pushed further
    assert out["min_information_retention"] == 9  # +1 step


def test_next_loop_targets_caps_ratio_at_configured_ceiling() -> None:
    rules = EvalScoringRules(next_ratio_cap_pct=99.0, next_ratio_step_pct=2.0)
    out = next_loop_targets(
        ratio_pct=98.5, retention_pct=100.0, compression_score=10, information_retention=10, rules=rules
    )
    assert out["min_compression_ratio_pct"] == 99.0


def test_custom_weights_change_diet_score() -> None:
    rules = EvalScoringRules(compression_weight=2.0, retention_weight=0.5)
    assert diet_score(5, 4, rules) == 12.0
    assert diet_score_max(rules) == 25.0
