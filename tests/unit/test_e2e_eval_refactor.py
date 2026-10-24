"""Unit: tests/e2e/run_e2e_eval.py post-ADR-061 refactor — $0, no pipeline/LLM calls.

Guards two things: (1) --skip-pipeline is now a true no-cost path (the judge
is only invoked through harness.collectors.judge, which short-circuits when
phase1_tokens<=0 — see test_harness_collectors_judge.py for that contract),
and (2) latest_agent_directives.json's schema (schema_version=1, top-level
keys) stays stable for anything already consuming that file.
"""

from __future__ import annotations

import pytest

from harness.judge_schema import EvaluationResult
from graph_distiller.rules import EvalScoringRules

pytestmark = pytest.mark.unit


def test_count_error_tags_delegates_to_harness_logs_collector() -> None:
    from e2e.run_e2e_eval import _count_error_tags

    errors = ["[DLQ] x", "[VERIFY_FAIL] Chapter 1", "[FATAL] Chapter 1 failed after 3 retries."]
    dlq, poison, verify = _count_error_tags(errors)
    assert (dlq, poison, verify) == (1, 1, 1)


def test_tail_errors_log_respects_rules_object(tmp_path) -> None:
    from e2e.run_e2e_eval import _tail_errors_log

    log = tmp_path / "errors.log"
    log.write_text("a\nb\nc\nd\n", encoding="utf-8")
    rules = EvalScoringRules(error_log_tail_lines=2)
    assert _tail_errors_log(log, rules) == ["c", "d"]


def test_tail_errors_log_missing_file_message(tmp_path) -> None:
    from e2e.run_e2e_eval import _tail_errors_log

    rules = EvalScoringRules()
    assert _tail_errors_log(tmp_path / "nope.log", rules) == ["(errors.log not found)"]


def test_build_agent_directives_schema_is_backward_compatible() -> None:
    from e2e.run_e2e_eval import build_agent_directives

    eval_result = EvaluationResult(compression_score=7, information_retention=9, feedback="a\nb")
    directives = build_agent_directives(
        generated_at="2026-09-17 00:00:00",
        diet={"compression_ratio_pct": 50.0, "core_retention_pct": 100.0, "phase1_tokens": 100, "repo_map_tokens": 50},
        eval_result=eval_result,
        prior=None,
        chapter_count=3,
        dlq_count=0,
        poison_count=0,
        scoring_rules=EvalScoringRules(),
    )
    assert directives["schema_version"] == 1
    assert set(directives) == {
        "schema_version", "goal", "generated_at", "controller", "baseline",
        "prior_baseline", "directives", "next_loop_targets", "loss_function",
    }
    assert directives["next_loop_targets"] == {
        "min_compression_ratio_pct": 52.0,
        "min_core_retention_pct": 100.0,
        "min_compression_score": 7,
        "min_information_retention": 10,
    }
    assert directives["baseline"]["compression_score"] == 7
