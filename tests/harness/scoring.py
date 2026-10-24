"""Prompt Diet scoring — post-processing of the LLM judge's numbers (ADR-060/061).

The judge's own instructions live in data/prompts/eval_judge.md (ADR-059) and
are sent verbatim to the LLM. Everything here is pure, deterministic, $0 code
that turns the judge's 1-10 scores (or a measured fallback when the judge
call itself fails) into a single diet_score and the next agent loop's
targets. Values come from data/rules/eval_scoring.toml via graph_distiller.rules —
see EvalScoringRules for what each field means.

Shared by tests/harness/collectors/judge.py and tests/e2e/run_e2e_eval.py so
both compute the same numbers the same way.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from graph_distiller.rules import EvalScoringRules

TRUNCATION_MARKER = "\n...[truncated]...\n"


def clip_feedback(text: str, rules: EvalScoringRules) -> str:
    lines = [ln.rstrip() for ln in (text or "").strip().splitlines() if ln.strip()]
    if len(lines) > rules.feedback_max_lines:
        lines = lines[: rules.feedback_max_lines]
    return "\n".join(lines) if lines else (text or "").strip()


def truncate_judge_input(text: str, rules: EvalScoringRules) -> str:
    if len(text) <= rules.judge_input_max_chars:
        return text
    return text[: rules.judge_input_max_chars] + TRUNCATION_MARKER


def tail_lines(lines: Sequence[str], rules: EvalScoringRules) -> List[str]:
    if rules.error_log_tail_lines <= 0:
        return []
    return list(lines[-rules.error_log_tail_lines :])


def diet_score(compression_score: int, information_retention: int, rules: EvalScoringRules) -> float:
    return compression_score * rules.compression_weight + information_retention * rules.retention_weight


def diet_score_max(rules: EvalScoringRules) -> float:
    return 10.0 * rules.compression_weight + 10.0 * rules.retention_weight


def fallback_scores(ratio_pct: float, retention_pct: float) -> tuple[int, int]:
    """When the judge call itself fails: convert measured percentages (0-100) to a 1-10 scale.

    Not governed by eval_scoring.toml — this is an emergency substitute for the
    judge's opinion, not a scoring-criteria choice, so the /10 conversion stays fixed.
    """
    compression_score = max(1, min(10, int(round(ratio_pct / 10.0)) or 1))
    information_retention = max(1, min(10, int(round(retention_pct / 10.0)) or 1))
    return compression_score, information_retention


def next_loop_targets(
    *,
    ratio_pct: float,
    retention_pct: float,
    compression_score: int,
    information_retention: int,
    rules: EvalScoringRules,
) -> Dict[str, Any]:
    target_ratio = min(rules.next_ratio_cap_pct, max(ratio_pct + rules.next_ratio_step_pct, ratio_pct))
    if retention_pct < 100.0:
        target_ratio = ratio_pct  # recover retention before pushing compression further

    if retention_pct >= 100.0:
        min_information_retention = 10
    else:
        min_information_retention = min(10, max(information_retention + rules.next_score_step, 1))

    return {
        "min_compression_ratio_pct": target_ratio,
        "min_core_retention_pct": rules.next_retention_target_pct,
        "min_compression_score": max(compression_score, 1),
        "min_information_retention": min_information_retention,
    }
