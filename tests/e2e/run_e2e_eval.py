"""
GraphDistiller E2E evaluation harness — Prompt Diet controller.

The product goal measured here is extreme token reduction (Phase1 source ->
Phase4 RepoMap), not prose quality. This harness is the control loop for the
agent: metrics + machine-readable directives become the next-loop baseline.

Usage (from repo root):

    python tests/e2e/run_e2e_eval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Import path: project_root/src  (harness must not require editing src/)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
LATEST_DIRECTIVES_PATH = REPORTS_DIR / "latest_agent_directives.json"

for _p in (SRC, TESTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from graph_distiller import local_pipeline as graph_distiller_main  # noqa: E402
from graph_distiller.budget import (  # noqa: E402
    DEFAULT_BUDGET_USD,
    DEFAULT_TOKEN_BUDGET,
    clamp_budget_usd,
    clamp_token_budget,
)
from graph_distiller.config import settings  # noqa: E402
from graph_distiller.cost import KRW_PER_USD  # noqa: E402
from graph_distiller.rules import get_rules  # noqa: E402
from graph_distiller.schema import GraphDistillerState, empty_state  # noqa: E402

from harness.collectors.judge import run_judge_with_fallback  # noqa: E402
from harness.collectors.repomap import (  # noqa: E402
    _collect_repomap_metrics,
    _phase1_source_tokens,
    compute_compression_ratio_pct,
)
from harness.judge_schema import EvaluationResult  # noqa: E402
from harness.model import MetricRecord  # noqa: E402
from harness.render.csv_out import metrics_to_csv, write_csv  # noqa: E402
from harness.scoring import diet_score as compute_diet_score  # noqa: E402
from harness.scoring import diet_score_max as compute_diet_score_max  # noqa: E402
from harness.scoring import tail_lines  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics helpers
#
# EvaluationResult, _phase1_source_tokens, _collect_repomap_metrics,
# _core_node_ids_for_chapter and compute_compression_ratio_pct now live in
# tests/harness/ (judge_schema.py, collectors/repomap.py) — imported above —
# so the paid E2E judge and the $0 audit harness measure Prompt Diet the same
# way from one implementation instead of two kept in sync by hand.
# ---------------------------------------------------------------------------


def _count_error_tags(errors: Sequence[str]) -> Tuple[int, int, int]:
    """Return (dlq_count, poison_count, verify_fail_count)."""
    from harness.collectors.logs import count_error_tags

    tags = count_error_tags(errors)
    return tags["dlq"], tags["poison"], tags["verify_fail"]


def _tail_errors_log(path: Path, rules) -> List[str]:
    if not path.is_file():
        return ["(errors.log not found)"]
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [f"(failed to read errors.log: {exc})"]
    if not lines:
        return ["(errors.log empty)"]
    return tail_lines(lines, rules) or ["(tail disabled: error_log_tail_lines=0)"]


def _load_chapter_markdown(codex_dir: Path) -> List[Tuple[str, str]]:
    """Return [(filename, content), ...] for chapter_*.md (skip README)."""
    chapters: List[Tuple[str, str]] = []
    for path in sorted(codex_dir.glob("chapter_*.md")):
        chapters.append((path.name, path.read_text(encoding="utf-8")))
    return chapters


def _load_prior_directives() -> Optional[Dict[str, Any]]:
    if not LATEST_DIRECTIVES_PATH.is_file():
        return None
    try:
        return json.loads(LATEST_DIRECTIVES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Pipeline via main.run_pipeline(auto_confirm=True) - never hangs on Y/n
#
# Budget enforcement moved to main.run_pipeline() itself (ADR-019): it slices
# chunks/chapters to the hard ceiling before any LLM call, so there is no
# need for a separate Phase1-only pre-check here anymore (that used to run
# Phase1 twice — once just to estimate, once for real).
# ---------------------------------------------------------------------------


def run_pipeline_phases(
    *, budget_usd: float, token_budget: int
) -> Tuple[GraphDistillerState, Dict[str, Any]]:
    """
    Call main.run_pipeline(auto_confirm=True, ...) so Pre-flight never blocks.
    The hard ceiling (ADR-019) is enforced inside main.run_pipeline itself.
    """
    t0 = time.perf_counter()
    print(
        f"\n[E2E] Invoking main.run_pipeline(auto_confirm=True, "
        f"budget_usd={budget_usd}, token_budget={token_budget}) ..."
    )
    exit_code = graph_distiller_main.run_pipeline(
        auto_confirm=True, budget_usd=budget_usd, token_budget=token_budget
    )
    elapsed = time.perf_counter() - t0
    outcome = graph_distiller_main.LAST_PIPELINE_OUTCOME

    if outcome is None or outcome.state is None:
        raise RuntimeError(
            f"main.run_pipeline returned {exit_code} without LAST_PIPELINE_OUTCOME"
        )
    if exit_code != 0 and not (outcome.state.get("drafts") or outcome.state.get("toc")):
        raise RuntimeError(
            f"Pipeline failed with exit_code={exit_code} "
            f"(TOC empty or early abort)"
        )

    usage2 = outcome.usage2
    usage4 = outcome.usage4
    meta = {
        "elapsed_sec": elapsed,
        "phase4_ran": usage4 is not None,
        "usage2": usage2,
        "usage4": usage4,
        "est2": outcome.est2,
        "est4": outcome.est4,
        "budget_usd": budget_usd,
        "token_budget": token_budget,
        "exit_code": exit_code,
        "written": [],
    }
    return outcome.state, meta


# ---------------------------------------------------------------------------
# Judge — tests/harness/collectors/judge.py::run_judge / run_judge_with_fallback
# (imported above). Kept out of this file so the $0 audit harness and this
# paid E2E script score Prompt Diet with the exact same code (ADR-061).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Agent directives (self-correction baseline)
# ---------------------------------------------------------------------------


def build_agent_directives(
    *,
    generated_at: str,
    diet: Dict[str, Any],
    eval_result: EvaluationResult,
    prior: Optional[Dict[str, Any]],
    chapter_count: int,
    dlq_count: int,
    poison_count: int,
    scoring_rules,
) -> Dict[str, Any]:
    """Machine-readable baseline for the next agent self-correction loop."""
    from harness.scoring import next_loop_targets

    ratio = float(diet.get("compression_ratio_pct") or 0.0)
    retention = float(diet.get("core_retention_pct") or 0.0)
    targets = next_loop_targets(
        ratio_pct=ratio,
        retention_pct=retention,
        compression_score=eval_result.compression_score,
        information_retention=eval_result.information_retention,
        rules=scoring_rules,
    )

    feedback_lines = [
        ln.strip()
        for ln in (eval_result.feedback or "").splitlines()
        if ln.strip()
    ]

    return {
        "schema_version": 1,
        "goal": "Prompt Diet",
        "generated_at": generated_at,
        "controller": "tests/run_e2e_eval.py",
        "baseline": {
            "phase1_tokens": int(diet.get("phase1_tokens") or 0),
            "repo_map_tokens": int(diet.get("repo_map_tokens") or 0),
            "compression_ratio_pct": ratio,
            "core_retention_pct": retention,
            "expected_core_nodes": list(diet.get("expected_core_nodes") or []),
            "missing_core_nodes": list(diet.get("missing_core_nodes") or []),
            "compression_score": eval_result.compression_score,
            "information_retention": eval_result.information_retention,
            "chapter_count": chapter_count,
            "dlq_count": dlq_count,
            "poison_count": poison_count,
        },
        "prior_baseline": (prior or {}).get("baseline") if prior else None,
        "directives": feedback_lines,
        "next_loop_targets": targets,
        "loss_function": {
            "primary": "maximize compression_ratio_pct",
            "constraint": "information_retention / core nodes must not regress",
            "forbidden_objectives": [
                "subjective prose quality",
                "citation_score",
                "structure_score",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report_markdown(
    *,
    generated_at: str,
    elapsed_sec: float,
    chapter_count: int,
    dlq_count: int,
    poison_count: int,
    verify_fail_count: int,
    poisoned_chapters: Sequence[int],
    state_error_count: int,
    phase4_ran: bool,
    eval_result: EvaluationResult,
    error_log_tail: Sequence[str],
    chapter_files: Sequence[str],
    diet_score: float,
    diet_score_max: float,
    diet: Dict[str, Any],
    agent_directives: Dict[str, Any],
    cost_meta: Optional[Dict[str, Any]] = None,
) -> str:
    err_block = "\n".join(f"    {line}" for line in error_log_tail) or "    (none)"
    feedback_block = "\n".join(
        f"> {line}" for line in eval_result.feedback.splitlines()
    ) or "> (no feedback)"

    files_list = (
        "\n".join(f"- `{name}`" for name in chapter_files)
        if chapter_files
        else "- _(none)_"
    )

    cost_section = _format_cost_analysis_section(cost_meta)
    directives_json = json.dumps(agent_directives, ensure_ascii=False, indent=2)

    return f"""# GraphDistiller E2E Evaluation Report (Prompt Diet)

- Generated: `{generated_at}`
- Judge model: `{settings.synthesis_model}`
- Codex dir: `{settings.codex_output_dir}`
- Controller goal: **extreme token reduction (Prompt Diet)**, not search/prose quality

## [실행 요약]

| Metric | Value |
|--------|------:|
| Total elapsed (sec) | {elapsed_sec:.2f} |
| Chapters generated | {chapter_count} |
| DLQ errors | {dlq_count} |
| Poison / FATAL errors | {poison_count} |
| Verify-fail events | {verify_fail_count} |
| Poisoned chapter ids | {list(poisoned_chapters) if poisoned_chapters else "none"} |
| State `errors` length | {state_error_count} |
| Phase 4 executed | {"yes" if phase4_ran else "no"} |
| Phase1 source tokens | {diet.get("phase1_tokens", 0)} |
| RepoMap tokens (Phase4) | {diet.get("repo_map_tokens", 0)} |
| **Compression ratio (%)** | **{float(diet.get("compression_ratio_pct") or 0.0):.2f}** |
| Core node retention (%) | {float(diet.get("core_retention_pct") or 0.0):.2f} |
| Missing core nodes | {len(diet.get("missing_core_nodes") or [])} |

### Chapter artifacts

{files_list}

## [Prompt Diet 평가]

| Score | Value (range) |
|-------|--------------:|
| compression_score | {eval_result.compression_score} / 10 |
| information_retention | {eval_result.information_retention} / 10 |
| **diet_score** (compression + retention) | **{diet_score:.1f}** / {diet_score_max:g} |

### Feedback (actionable)

{feedback_block}

## [비용 분석]

{cost_section}

## [에러 로그 요약]

Tail of `{settings.errors_log_path.name}` (last {len(error_log_tail)} lines):

```
{err_block}
```

## [에이전트 행동 지침]

다음 루프의 자가 교정(Self-Correction) **Baseline**. 기계가 파싱할 것.
동일 내용: `tests/reports/latest_agent_directives.json`

```json
{directives_json}
```

---
_Report written by `tests/run_e2e_eval.py` (Prompt Diet controller; `main.run_pipeline(auto_confirm=True)`)._
"""


def _format_cost_analysis_section(cost_meta: Optional[Dict[str, Any]]) -> str:
    if not cost_meta:
        return "_비용 메타 없음 (예: `--skip-pipeline`)._"

    budget_usd = float(cost_meta.get("budget_usd") or 0.0)
    token_budget = int(cost_meta.get("token_budget") or 0)
    usage2 = cost_meta.get("usage2")
    usage4 = cost_meta.get("usage4")
    est2 = cost_meta.get("est2") or {}
    est4 = cost_meta.get("est4") or {}

    actual_in = 0
    actual_out = 0
    actual_usd = 0.0
    if usage2 is not None:
        actual_in += int(getattr(usage2, "prompt_tokens", 0) or 0)
        actual_out += int(getattr(usage2, "completion_tokens", 0) or 0)
        actual_usd += float(getattr(usage2, "total_cost_usd", 0.0) or 0.0)
    if usage4 is not None:
        actual_in += int(getattr(usage4, "prompt_tokens", 0) or 0)
        actual_out += int(getattr(usage4, "completion_tokens", 0) or 0)
        actual_usd += float(getattr(usage4, "total_cost_usd", 0.0) or 0.0)

    gate2_usd = float(est2.get("cost_usd", 0.0) or 0.0) if isinstance(est2, dict) else 0.0
    gate4_usd = float(est4.get("cost_usd", 0.0) or 0.0) if isinstance(est4, dict) else 0.0
    gate_total = gate2_usd + gate4_usd

    return f"""| Item | Gate (main) estimate | Actual (callback) |
|------|---------------------:|------------------:|
| Phase 2 USD | ${gate2_usd:.6f} | - |
| Phase 4 USD | ${gate4_usd:.6f} | - |
| **Total USD** | **${gate_total:.6f}** | **${actual_usd:.6f}** |
| Total KRW (@{KRW_PER_USD}) | {gate_total * KRW_PER_USD:.2f} | {actual_usd * KRW_PER_USD:.2f} |
| Actual tokens in/out | - | {actual_in} / {actual_out} |
| Hard ceiling this run (ADR-019) | ${budget_usd:.2f} / {token_budget:,} tokens | |

- Hard ceiling is enforced inside `main.run_pipeline()` by slicing chunks/chapters
  before any LLM call — the run cannot exceed it (see `[Budget]` lines in stdout/log
  if anything was dropped).
- Actual USD comes from LangChain `get_openai_callback` (may differ from table rates).
"""


def save_report(markdown: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORTS_DIR / f"eval_{stamp}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def save_latest_directives(directives: Dict[str, Any]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(directives, ensure_ascii=False, indent=2) + "\n"
    LATEST_DIRECTIVES_PATH.write_text(payload, encoding="utf-8")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stamped = REPORTS_DIR / f"directives_{stamp}.json"
    stamped.write_text(payload, encoding="utf-8")
    return LATEST_DIRECTIVES_PATH


def save_metrics_csv(
    *,
    diet: Dict[str, Any],
    eval_result: EvaluationResult,
    diet_score: float,
    chapter_count: int,
    dlq_count: int,
    poison_count: int,
    verify_fail_count: int,
) -> Path:
    """Tidy CSV alongside the .md report / directives .json (ADR-061) — one row
    per metric, same schema tests/audit.py's harness.render.csv_out uses, so
    this run's numbers can be diffed against an audit run or loaded by pandas."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [
        MetricRecord(scope="e2e", run_id=run_id, name="chapters", value=chapter_count, unit="count"),
        MetricRecord(scope="e2e", run_id=run_id, name="dlq_count", value=dlq_count, unit="count"),
        MetricRecord(scope="e2e", run_id=run_id, name="poison_count", value=poison_count, unit="count"),
        MetricRecord(scope="e2e", run_id=run_id, name="verify_fail_count", value=verify_fail_count, unit="count"),
        MetricRecord(scope="e2e", run_id=run_id, name="phase1_tokens", value=int(diet.get("phase1_tokens") or 0), unit="tokens"),
        MetricRecord(scope="e2e", run_id=run_id, name="repo_map_tokens", value=int(diet.get("repo_map_tokens") or 0), unit="tokens"),
        MetricRecord(scope="e2e", run_id=run_id, name="compression_ratio_pct", value=float(diet.get("compression_ratio_pct") or 0.0), unit="pct"),
        MetricRecord(scope="e2e", run_id=run_id, name="core_retention_pct", value=float(diet.get("core_retention_pct") or 0.0), unit="pct", threshold=100.0),
        MetricRecord(scope="judge", run_id=run_id, name="compression_score", value=eval_result.compression_score, unit="1-10"),
        MetricRecord(scope="judge", run_id=run_id, name="information_retention", value=eval_result.information_retention, unit="1-10"),
        MetricRecord(scope="judge", run_id=run_id, name="diet_score", value=diet_score, unit="score"),
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return write_csv(metrics_to_csv(rows), REPORTS_DIR / f"eval_{run_id}.csv")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "GraphDistiller E2E Prompt Diet controller: "
            "compression ratio + retention judge + agent directives."
        )
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip Phase 1-4; measure diet from empty/partial state and "
        "existing codex only (compression may be N/A).",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=DEFAULT_BUDGET_USD,
        help=(
            f"Hard USD spend ceiling passed to main.run_pipeline "
            f"(default ${DEFAULT_BUDGET_USD:.2f}, clamped to $1.00 max; ADR-019)"
        ),
    )
    parser.add_argument(
        "--token-limit",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
        help=(
            f"Hard token ceiling passed to main.run_pipeline "
            f"(default {DEFAULT_TOKEN_BUDGET:,}, clamped to 10,000,000 max; ADR-019)"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    budget_usd = clamp_budget_usd(args.budget_usd)
    token_budget = clamp_token_budget(args.token_limit)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cost_meta: Optional[Dict[str, Any]] = None
    prior = _load_prior_directives()
    if prior:
        print(
            f"[E2E] Loaded prior agent baseline from {LATEST_DIRECTIVES_PATH.name}"
        )

    if args.skip_pipeline:
        print("[E2E] --skip-pipeline: judging existing codex / empty state diet")
        state = empty_state()
        elapsed_sec = 0.0
        phase4_ran = False
        poisoned: List[int] = []
        errors: List[str] = []
    else:
        try:
            state, meta = run_pipeline_phases(
                budget_usd=budget_usd, token_budget=token_budget
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[E2E] FATAL: {exc}", file=sys.stderr)
            return 1
        elapsed_sec = float(meta["elapsed_sec"])
        phase4_ran = bool(meta["phase4_ran"])
        poisoned = list(state.get("poisoned_chapters") or [])
        errors = list(state.get("errors") or [])
        cost_meta = meta

    chapters = _load_chapter_markdown(settings.codex_output_dir)
    chapter_count = len(chapters)
    if not args.skip_pipeline and state.get("drafts"):
        chapter_count = len(state.get("drafts") or [])

    phase1_tokens = _phase1_source_tokens(state)
    repomap_metrics = _collect_repomap_metrics(state)
    compression_ratio_pct = compute_compression_ratio_pct(
        phase1_tokens, int(repomap_metrics["repo_map_tokens"])
    )
    diet: Dict[str, Any] = {
        "phase1_tokens": phase1_tokens,
        "compression_ratio_pct": compression_ratio_pct,
        **repomap_metrics,
    }
    print(
        f"\n[E2E] Prompt Diet  phase1_tokens={phase1_tokens}  "
        f"repo_map_tokens={repomap_metrics['repo_map_tokens']}  "
        f"compression_ratio_pct={compression_ratio_pct:.2f}  "
        f"core_retention_pct={repomap_metrics['core_retention_pct']:.2f}"
    )

    dlq_count, poison_count, verify_fail_count = _count_error_tags(errors)
    if poisoned:
        poison_count = max(poison_count, len(poisoned))

    scoring_rules = get_rules(settings.rules_dir).eval_scoring
    eval_result, judge_error = run_judge_with_fallback(diet=diet, settings=settings, scoring_rules=scoring_rules)
    if judge_error:
        print(f"[E2E] Judge failed: {judge_error}", file=sys.stderr)

    diet_score = compute_diet_score(eval_result.compression_score, eval_result.information_retention, scoring_rules)
    error_tail = _tail_errors_log(settings.errors_log_path, scoring_rules)

    agent_directives = build_agent_directives(
        generated_at=generated_at,
        diet=diet,
        eval_result=eval_result,
        prior=prior,
        chapter_count=chapter_count,
        dlq_count=dlq_count,
        poison_count=poison_count,
        scoring_rules=scoring_rules,
    )
    directives_path = save_latest_directives(agent_directives)

    report = build_report_markdown(
        generated_at=generated_at,
        elapsed_sec=elapsed_sec,
        chapter_count=chapter_count,
        dlq_count=dlq_count,
        poison_count=poison_count,
        verify_fail_count=verify_fail_count,
        poisoned_chapters=poisoned,
        state_error_count=len(errors),
        phase4_ran=phase4_ran,
        eval_result=eval_result,
        error_log_tail=error_tail,
        chapter_files=[name for name, _ in chapters],
        diet_score=diet_score,
        diet_score_max=compute_diet_score_max(scoring_rules),
        diet=diet,
        agent_directives=agent_directives,
        cost_meta=cost_meta,
    )
    report_path = save_report(report)
    metrics_csv_path = save_metrics_csv(
        diet=diet,
        eval_result=eval_result,
        diet_score=diet_score,
        chapter_count=chapter_count,
        dlq_count=dlq_count,
        poison_count=poison_count,
        verify_fail_count=verify_fail_count,
    )

    print("\n" + "=" * 60)
    print("[E2E] Prompt Diet evaluation complete")
    print(f"  elapsed              : {elapsed_sec:.2f}s")
    print(f"  chapters             : {chapter_count}")
    print(f"  DLQ / poison         : {dlq_count} / {poison_count}")
    print(
        f"  compression_ratio_%  : {compression_ratio_pct:.2f}  "
        f"(phase1={phase1_tokens} -> repo={repomap_metrics['repo_map_tokens']})"
    )
    print(
        f"  scores               : compression={eval_result.compression_score} "
        f"retention={eval_result.information_retention} "
        f"diet={diet_score:.1f}/{compute_diet_score_max(scoring_rules):g}"
    )
    print(f"  report               : {report_path}")
    print(f"  agent_directives     : {directives_path}")
    print(f"  metrics_csv          : {metrics_csv_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
