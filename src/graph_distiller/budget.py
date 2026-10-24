"""Hard spend ceiling policy — pre-slices work so a run never exceeds the cap.

Not part of cost.py (ADR-014 keeps cost.py free of policy constants); this
module owns the defaults/ceilings and the slicing that enforces them, reused
by both graph_distiller.local_pipeline and tests/e2e/run_e2e_eval.py (ADR-019).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from graph_distiller.agents.repomap import build_repo_map
from graph_distiller.cost import calculate_cost, count_tokens
from graph_distiller.pricing import active_rate_for
from graph_distiller.schema import ChunkMeta, GraphNode

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_USD = 0.25
MAX_BUDGET_USD = 1.00

DEFAULT_TOKEN_BUDGET = 1_000_000
MAX_TOKEN_BUDGET = 10_000_000

PHASE2_WRAPPER_TOKENS_PER_CHUNK = 80  # mirrors cost.py's own heuristic


def clamp_budget_usd(value: float) -> float:
    """Clamp to [0, MAX_BUDGET_USD], warning if the input overshot the ceiling."""
    v = max(0.0, float(value))
    if v > MAX_BUDGET_USD:
        logger.warning(
            "--budget-usd %.6f exceeds ceiling $%.2f; clamped to $%.2f",
            v,
            MAX_BUDGET_USD,
            MAX_BUDGET_USD,
        )
        print(
            f"[Budget] --budget-usd {v:.6f} > ${MAX_BUDGET_USD:.2f} ceiling; "
            f"clamped to ${MAX_BUDGET_USD:.2f}."
        )
        return MAX_BUDGET_USD
    return v


def clamp_token_budget(value: int) -> int:
    """Clamp to [0, MAX_TOKEN_BUDGET], warning if the input overshot the ceiling."""
    v = max(0, int(value))
    if v > MAX_TOKEN_BUDGET:
        logger.warning(
            "--token-limit %d exceeds ceiling %d tokens; clamped",
            v,
            MAX_TOKEN_BUDGET,
        )
        print(
            f"[Budget] --token-limit {v} > {MAX_TOKEN_BUDGET} token ceiling; "
            f"clamped to {MAX_TOKEN_BUDGET}."
        )
        return MAX_TOKEN_BUDGET
    return v


def slice_chunks_by_budget(
    chunks: List[ChunkMeta],
    model: str,
    *,
    budget_usd: float,
    token_budget: int,
) -> Tuple[List[ChunkMeta], Dict[str, object]]:
    """
    Keep chunks (in existing order) while cumulative estimated Phase2 tokens/cost
    stay within both caps. Stops before the first chunk that would exceed either.
    """
    kept: List[ChunkMeta] = []
    cum_tokens = 0
    cum_cost = 0.0
    stop_reason = "none"

    for chunk in chunks:
        tokens = count_tokens(chunk.content or "") + PHASE2_WRAPPER_TOKENS_PER_CHUNK
        cost = calculate_cost(model, tokens, 0, rate=active_rate_for(model))
        if cum_tokens + tokens > token_budget:
            stop_reason = "token_budget"
            break
        if cum_cost + cost > budget_usd:
            stop_reason = "budget_usd"
            break
        kept.append(chunk)
        cum_tokens += tokens
        cum_cost += cost

    dropped = len(chunks) - len(kept)
    if dropped:
        logger.info(
            "Budget slice (chunks): kept=%d dropped=%d reason=%s "
            "cum_tokens=%d cum_cost_usd=%.6f",
            len(kept),
            dropped,
            stop_reason,
            cum_tokens,
            cum_cost,
        )
    return kept, {
        "kept": len(kept),
        "dropped": dropped,
        "stop_reason": stop_reason,
        "cum_tokens": cum_tokens,
        "cum_cost_usd": cum_cost,
    }


def slice_chapters_by_budget(
    toc: Dict[int, List[str]],
    nodes: List[GraphNode],
    chunks: List[ChunkMeta],
    *,
    synthesis_model: str,
    budget_usd: float,
    token_budget: int,
) -> Tuple[Dict[int, List[str]], Dict[str, object]]:
    """
    Keep chapters (ascending chapter_id, i.e. already top-N ranked order) while
    cumulative RepoMap tokens / estimated synthesis cost stay within both caps.
    """
    kept: Dict[int, List[str]] = {}
    cum_tokens = 0
    cum_cost = 0.0
    stop_reason = "none"

    for chapter_id, member_ids in sorted(toc.items()):
        repo_map = build_repo_map(
            chapter_id=chapter_id,
            member_ids=list(member_ids),
            nodes=nodes,
            chunks=chunks,
        )
        tokens = count_tokens(repo_map)
        cost = calculate_cost(
            synthesis_model, tokens, 0, rate=active_rate_for(synthesis_model)
        )
        if cum_tokens + tokens > token_budget:
            stop_reason = "token_budget"
            break
        if cum_cost + cost > budget_usd:
            stop_reason = "budget_usd"
            break
        kept[chapter_id] = member_ids
        cum_tokens += tokens
        cum_cost += cost

    dropped = len(toc) - len(kept)
    if dropped:
        logger.info(
            "Budget slice (chapters): kept=%d dropped=%d reason=%s "
            "cum_tokens=%d cum_cost_usd=%.6f",
            len(kept),
            dropped,
            stop_reason,
            cum_tokens,
            cum_cost,
        )
    return kept, {
        "kept": len(kept),
        "dropped": dropped,
        "stop_reason": stop_reason,
        "cum_tokens": cum_tokens,
        "cum_cost_usd": cum_cost,
    }
