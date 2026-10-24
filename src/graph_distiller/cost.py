"""Stateless cost calculator — pure functions only (no UI side effects or pipeline state).

Rates come from the caller (typically OpenRouter snapshot via ``pricing.py``,
ADR-031). This module does **not** maintain a live catalog and does **not**
guess prices for unknown models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import tiktoken

from graph_distiller.schema import ChunkMeta

CostStatus = Literal["known", "free", "unknown"]

KRW_PER_USD = 1350

# Heuristics (callers may document; calculator stays silent)
OUTPUT_TOKENS_PER_CHUNK = 250
PHASE2_WRAPPER_TOKENS_PER_CHUNK = 80
PHASE4_SYNTHESIS_OUTPUT_PER_CHAPTER = 1500
PHASE4_PLANNER_INPUT_PER_CHAPTER = 200
PHASE4_PLANNER_OUTPUT_PER_CHAPTER = 50
PHASE4_REPOMAP_UPPER_PER_CHAPTER = 80_000


@dataclass(frozen=True)
class ModelRate:
    """USD per 1M tokens, or an explicit non-numeric status."""

    status: CostStatus
    input_usd_per_1m: float = 0.0
    output_usd_per_1m: float = 0.0

    @staticmethod
    def unknown() -> "ModelRate":
        return ModelRate(status="unknown")

    @staticmethod
    def free() -> "ModelRate":
        return ModelRate(status="free", input_usd_per_1m=0.0, output_usd_per_1m=0.0)

    def label(self) -> str:
        if self.status == "free":
            return "FREE"
        if self.status == "unknown":
            return "UNKNOWN"
        return (
            f"${self.input_usd_per_1m:.4g}/${self.output_usd_per_1m:.4g} per 1M "
            f"(in/out)"
        )


def rate_from_per_1m(input_usd_per_1m: float, output_usd_per_1m: float) -> ModelRate:
    """Build a rate; both sides ~0 → free."""
    inn = max(0.0, float(input_usd_per_1m))
    out = max(0.0, float(output_usd_per_1m))
    if inn == 0.0 and out == 0.0:
        return ModelRate.free()
    return ModelRate(status="known", input_usd_per_1m=inn, output_usd_per_1m=out)


def merge_cost_status(*statuses: CostStatus) -> CostStatus:
    if any(s == "unknown" for s in statuses):
        return "unknown"
    if statuses and all(s == "free" for s in statuses):
        return "free"
    return "known"


def format_cost_status(cost_usd: float, status: CostStatus) -> str:
    """Human label for Pre-flight / tqdm (FREE / UNKNOWN / $…)."""
    if status == "free":
        return "FREE"
    if status == "unknown":
        return "UNKNOWN"
    return f"${float(cost_usd):.6f}"


def count_tokens(text: str) -> int:
    """Return tiktoken cl100k_base token count for ``text``."""
    return len(tiktoken.get_encoding("cl100k_base").encode(text or ""))


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
    *,
    rate: Optional[ModelRate] = None,
) -> float:
    """
    Return estimated USD for token counts.

    ``model_name`` is unused for math (kept for call-site clarity). Without a
    ``rate``, or when status is unknown/free, returns ``0.0``. Use
    ``cost_status`` / ``format_cost_status`` so FREE vs UNKNOWN is visible.
    """
    del model_name  # pricing is entirely in ``rate``
    if rate is None or rate.status == "unknown":
        return 0.0
    if rate.status == "free":
        return 0.0
    return (max(0, input_tokens) / 1_000_000.0) * rate.input_usd_per_1m + (
        max(0, output_tokens) / 1_000_000.0
    ) * rate.output_usd_per_1m


def estimate_phase2_cost(
    chunks: List[ChunkMeta],
    model_name: str,
    *,
    rate: Optional[ModelRate] = None,
) -> dict:
    """
    Pure Phase 2 estimate.

    Returns dict with keys:
      chunk_count, input_tokens, output_tokens, total_tokens, cost_usd,
      cost_status, model_name
    """
    n = len(chunks)
    content_tokens = sum(count_tokens(c.content or "") for c in chunks)
    input_tokens = content_tokens + n * PHASE2_WRAPPER_TOKENS_PER_CHUNK
    output_tokens = n * OUTPUT_TOKENS_PER_CHUNK
    r = rate if rate is not None else ModelRate.unknown()
    cost_usd = calculate_cost(model_name, input_tokens, output_tokens, rate=r)
    return {
        "chunk_count": n,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost_usd,
        "cost_status": r.status,
        "model_name": model_name,
    }


def estimate_phase4_cost(
    *,
    n_chapters: int,
    repo_map_tokens: int,
    synthesis_model: str,
    planner_model: str,
    synthesis_rate: Optional[ModelRate] = None,
    planner_rate: Optional[ModelRate] = None,
) -> dict:
    """
    Pure Phase 4 estimate from chapter count and precomputed RepoMap tokens.

    Returns dict with keys:
      n_chapters, repo_map_tokens, input_tokens, output_tokens, total_tokens,
      cost_usd, cost_usd_synth, cost_usd_planner, cost_status,
      synthesis_model, planner_model
    """
    n = max(0, int(n_chapters))
    repo = max(0, int(repo_map_tokens))
    synth_in = repo
    synth_out = n * PHASE4_SYNTHESIS_OUTPUT_PER_CHAPTER
    planner_in = n * PHASE4_PLANNER_INPUT_PER_CHAPTER
    planner_out = n * PHASE4_PLANNER_OUTPUT_PER_CHAPTER
    sr = synthesis_rate if synthesis_rate is not None else ModelRate.unknown()
    pr = planner_rate if planner_rate is not None else ModelRate.unknown()
    cost_usd_synth = calculate_cost(synthesis_model, synth_in, synth_out, rate=sr)
    cost_usd_planner = calculate_cost(planner_model, planner_in, planner_out, rate=pr)
    input_tokens = synth_in + planner_in
    output_tokens = synth_out + planner_out
    status = merge_cost_status(sr.status, pr.status)
    return {
        "n_chapters": n,
        "repo_map_tokens": repo,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost_usd_synth + cost_usd_planner,
        "cost_usd_synth": cost_usd_synth,
        "cost_usd_planner": cost_usd_planner,
        "cost_status": status,
        "synthesis_model": synthesis_model,
        "planner_model": planner_model,
    }


def estimate_phase4_upper_cost(
    *,
    n_chapters: int,
    synthesis_model: str,
    planner_model: str,
    synthesis_rate: Optional[ModelRate] = None,
    planner_rate: Optional[ModelRate] = None,
) -> dict:
    """Phase 4 ceiling: ``n_chapters * PHASE4_REPOMAP_UPPER_PER_CHAPTER`` as input."""
    n = max(0, int(n_chapters))
    return estimate_phase4_cost(
        n_chapters=n,
        repo_map_tokens=n * PHASE4_REPOMAP_UPPER_PER_CHAPTER,
        synthesis_model=synthesis_model,
        planner_model=planner_model,
        synthesis_rate=synthesis_rate,
        planner_rate=planner_rate,
    )


# Back-compat alias: empty map — do not treat as a price catalog (ADR-031).
MODEL_RATES_USD_PER_1M: Dict[str, Tuple[float, float]] = {}
