"""LLM Prompt Diet judge — the only paid collector in this harness.

Only runs when the caller explicitly asks for it (`tests/audit.py audit
--judge`). Every other collector in tests/harness/collectors/ is $0.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from graph_distiller.config import Settings, get_settings
from graph_distiller.prompts import load_prompt
from graph_distiller.rules import EvalScoringRules, get_rules

from harness.collectors import Collected
from harness.judge_schema import EvaluationResult
from harness.scoring import clip_feedback, diet_score, diet_score_max, fallback_scores, truncate_judge_input

SCOPE = "judge"

# Fallback only — the live prompt is data/prompts/eval_judge.md (ADR-059).
JUDGE_SYSTEM_FALLBACK = """\
You are a strict Prompt Diet evaluator for GraphDistiller.
Your ONLY goals: maximize token compression (Phase1 -> RepoMap) while retaining
PageRank-critical nodes. Do NOT score prose quality, style, or readability.

Rules:
1. compression_score (1-10): Reward extreme reduction. Use measured
   compression_ratio_pct first (higher % reduced => higher score).
2. information_retention (1-10): Check that listed PageRank core node ids appear
   in the RepoMap text. Any missing high-rank core is a CRITICAL defect
   (score must drop sharply; omit many => score near 1).
3. feedback: at most 3 short, machine-actionable lines for the next agent loop
   (e.g. raise top-% keep, tighten chunk attach, fix missing node X).
"""


def run_judge(
    *,
    diet: Dict[str, Any],
    settings: Optional[Settings] = None,
    scoring_rules: Optional[EvalScoringRules] = None,
) -> EvaluationResult:
    """Score Prompt Diet with SYNTHESIS_MODEL (structured EvaluationResult). Real $ cost."""
    cfg = settings or get_settings()
    rules = scoring_rules or get_rules(cfg.rules_dir).eval_scoring

    if int(diet.get("phase1_tokens", 0) or 0) <= 0:
        return EvaluationResult(
            compression_score=1,
            information_retention=1,
            feedback="No Phase1 tokens / no chapter artifacts.\n"
            "Re-run full pipeline without --skip-pipeline.\n"
            "Confirm raw_memos and TOC are non-empty.",
        )

    repo_text = truncate_judge_input(str(diet.get("repo_map_text") or ""), rules)
    missing = diet.get("missing_core_nodes") or []
    expected = diet.get("expected_core_nodes") or []
    metrics_block = (
        f"phase1_tokens={diet.get('phase1_tokens')}\n"
        f"repo_map_tokens={diet.get('repo_map_tokens')}\n"
        f"compression_ratio_pct={diet.get('compression_ratio_pct')}\n"
        f"core_retention_pct={diet.get('core_retention_pct')}\n"
        f"expected_core_nodes={expected}\n"
        f"missing_core_nodes={missing}\n"
    )

    system_prompt = load_prompt(cfg.prompts_dir, "eval_judge.md", fallback=JUDGE_SYSTEM_FALLBACK)
    llm = cfg.build_chat_model(cfg.synthesis_model)
    structured = llm.with_structured_output(EvaluationResult)
    result = structured.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    "Evaluate Prompt Diet using measured metrics and RepoMap.\n"
                    "Do NOT judge writing quality.\n\n"
                    f"=== MEASURED METRICS ===\n{metrics_block}\n"
                    f"=== REPOMAP (evidence) ===\n{repo_text}\n"
                )
            ),
        ]
    )
    if not isinstance(result, EvaluationResult):
        result = EvaluationResult.model_validate(result)
    result.feedback = clip_feedback(result.feedback, rules)
    return result


def run_judge_with_fallback(
    *,
    diet: Dict[str, Any],
    settings: Optional[Settings] = None,
    scoring_rules: Optional[EvalScoringRules] = None,
) -> tuple[EvaluationResult, Optional[str]]:
    """run_judge(), but a failed call degrades to a measured-percentage estimate
    instead of raising. Returns (result, error_message_or_None)."""
    try:
        return run_judge(diet=diet, settings=settings, scoring_rules=scoring_rules), None
    except Exception as exc:  # noqa: BLE001 — judge failure must not crash the audit/E2E run
        ratio = float(diet.get("compression_ratio_pct") or 0.0)
        retention = float(diet.get("core_retention_pct") or 0.0)
        compression_score, information_retention = fallback_scores(ratio, retention)
        result = EvaluationResult(
            compression_score=compression_score,
            information_retention=information_retention,
            feedback=f"Judge invocation failed: {exc}\n"
            "Use measured compression_ratio_pct / core_retention_pct as baseline.\n"
            "Check OPENROUTER_API_KEY / SYNTHESIS_MODEL.",
        )
        return result, str(exc)


def collect(diet: Dict[str, Any], *, settings: Optional[Settings] = None) -> Collected:
    out = Collected()
    cfg = settings or get_settings()
    rules = get_rules(cfg.rules_dir).eval_scoring

    result, error = run_judge_with_fallback(diet=diet, settings=cfg, scoring_rules=rules)
    out.metric(SCOPE, "compression_score", result.compression_score, threshold=1, unit="1-10")
    out.metric(SCOPE, "information_retention", result.information_retention, threshold=1, unit="1-10")
    out.metric(SCOPE, "diet_score", diet_score(result.compression_score, result.information_retention, rules),
               unit=f"/{diet_score_max(rules):g}")
    out.metric(SCOPE, "feedback", result.feedback)
    out.check(
        "judge.call_succeeded",
        SCOPE,
        error is None,
        severity="warn",
        message=error or "",
    )
    return out
