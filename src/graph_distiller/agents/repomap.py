"""RepoMap Agent — natural-language narrative compression (ADR-050).

Blindfold v1 (2026-09-16) measured compression -1.56: node-centric repeated
snippets exploded tokens and shredded context. Natural-language chapters are
not code trees — the unit of meaning is the chunk, not the node.

Structure:
- Part 1 [Core Concepts]: PageRank desc top nodes, name + score only.
- Part 2 [Grounding Context]: deduplicated chunk bodies (full content),
  cited as [DOC_ID: ...]. Truncated at the 80k hard ceiling (ADR-044).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Set

import tiktoken

from graph_distiller.config import get_settings
from graph_distiller.contracts import DOC_ID_RE, PART2_HEADER, format_doc_id
from graph_distiller.rules import get_rules
from graph_distiller.schema import ChunkMeta, GraphNode, GraphDistillerState

logger = logging.getLogger(__name__)

# W4-B hard ceiling — raising without measured ADR is forbidden. Fallbacks only;
# the live values are data/rules/phase4_repomap.toml (ADR-060). Kept here so a
# missing/corrupt rules dir still renders a correct, bounded RepoMap.
REPOMAP_TOKEN_BUDGET = 80_000
REPOMAP_TOP_FRACTION = 0.20
# ADR-050 v3: Part 2 may use at most this fraction of the chapter's unique
# raw chunk tokens, so compression < 1.0 is structurally reachable. The
# Blindfold harness imports this constant — do not fork the value.
GROUNDING_BUDGET_FRACTION = 0.90

TRUNCATED_MARKER = "[TRUNCATED: token budget reached]"


def part2_doc_ids(repo_map: str) -> Set[str]:
    """
    DOC_IDs printed in RepoMap Part 2 only (ADR-055 / Verifier allowlist).

    Part 1 node names must never enter the set — they are concepts, not docs.
    """
    text = repo_map or ""
    idx = text.find(PART2_HEADER)
    if idx < 0:
        return set()
    part2 = text[idx + len(PART2_HEADER) :]
    return {m.strip() for m in DOC_ID_RE.findall(part2) if m.strip()}


def resolve_repomap_token_budget(requested: int | None = None) -> int:
    """
    Enforce Lost-in-the-Middle hard cap (W4-B).

    Values above REPOMAP_TOKEN_BUDGET are clamped down (never silently enlarged).
    """
    if requested is None:
        try:
            requested = get_rules().repomap.token_budget
        except Exception:  # noqa: BLE001 — resolve_repomap_token_budget must never itself crash a chapter
            requested = REPOMAP_TOKEN_BUDGET
    v = int(requested)
    if v < 0:
        return 0
    if v > REPOMAP_TOKEN_BUDGET:
        logger.warning(
            "RepoMap token_budget=%s exceeds hard ceiling %s; clamped (W4-B "
            "Lost-in-the-Middle — do not stuff more context)",
            v,
            REPOMAP_TOKEN_BUDGET,
        )
        return REPOMAP_TOKEN_BUDGET
    return v


def _count_tokens(text: str) -> int:
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _chunk_lookup(chunks: List[ChunkMeta]) -> Dict[str, ChunkMeta]:
    return {c.chunk_id: c for c in chunks}


def core_keep_count(
    n_nodes: int,
    *,
    small_keep_all_below: int | None = None,
) -> int:
    """
    How many PageRank-ranked chapter nodes to list in Part 1 (ADR-025 stage 3).

    Chapters with n <= small_keep_all_below keep every node (20% would collapse
    to 1). Larger chapters still keep ceil(n * 20%), at least 1.
    Core-first order is PageRank desc — do not reorder without measurement (W4-B).
    """
    if n_nodes <= 0:
        return 0
    threshold = (
        get_settings().repomap_small_keep_all_below
        if small_keep_all_below is None
        else small_keep_all_below
    )
    if n_nodes <= threshold:
        return n_nodes
    try:
        top_fraction = get_rules().repomap.top_fraction
    except Exception:  # noqa: BLE001 — core_keep_count must never itself crash a chapter
        top_fraction = REPOMAP_TOP_FRACTION
    return max(1, math.ceil(n_nodes * top_fraction))


def build_repo_map(
    chapter_id: int,
    member_ids: List[str],
    nodes: List[GraphNode],
    chunks: List[ChunkMeta],
    token_budget: int | None = None,
    *,
    snippet_chars: int | None = None,
    small_keep_all_below: int | None = None,
    grounding_budget: int | None = None,
) -> str:
    """
    Natural-language RepoMap (ADR-050):
    1. Part 1 — core concepts: PageRank desc, name + score only (no snippets)
    2. Part 2 — grounding context: dedup'd full chunk bodies, ranked by the
       summed PageRank of their citing core nodes (ties broken by chunk_id),
       truncated chunk-by-chunk at the token budget

    grounding_budget (ADR-050 v3): optional extra ceiling for Part 2 only.
    The Blindfold harness injects raw_tokens * 0.9 so compression < 1.0 is
    structurally reachable; the product path leaves it None (80k ceiling only).
    """
    if snippet_chars is not None:
        logger.warning(
            "snippet_chars is deprecated (ADR-050): Part 2 emits full chunk "
            "bodies; the 80k token budget is the only limiter"
        )
    token_budget = resolve_repomap_token_budget(token_budget)
    members = {m for m in member_ids}
    chapter_nodes = [n for n in nodes if n.id in members]
    chapter_nodes.sort(key=lambda n: n.pagerank_score, reverse=True)

    if not chapter_nodes:
        return f"[Chapter: {chapter_id}]\n(no nodes)\n"

    keep_n = core_keep_count(
        len(chapter_nodes), small_keep_all_below=small_keep_all_below
    )
    core_nodes = chapter_nodes[:keep_n]
    by_chunk = _chunk_lookup(chunks)

    lines: List[str] = [f"[Chapter: {chapter_id}]", "## Part 1: Core Concepts"]
    used_tokens = _count_tokens("\n".join(lines) + "\n")

    for node in core_nodes:
        entry = f"- {node.id} (PageRank: {node.pagerank_score:.4f})"
        entry_tokens = _count_tokens(entry + "\n")
        if used_tokens + entry_tokens > token_budget:
            lines.append(TRUNCATED_MARKER)
            return "\n".join(lines) + "\n"
        lines.append(entry)
        used_tokens += entry_tokens

    header = PART2_HEADER
    header_tokens = _count_tokens(header + "\n")
    if used_tokens + header_tokens > token_budget:
        lines.append(TRUNCATED_MARKER)
        return "\n".join(lines) + "\n"
    lines.append(header)
    used_tokens += header_tokens

    chunk_weight: Dict[str, float] = {}
    for n in core_nodes:
        for cid in n.source_chunk_ids:
            chunk_weight[cid] = chunk_weight.get(cid, 0.0) + n.pagerank_score
    cited_ids = sorted(chunk_weight, key=lambda c: (-chunk_weight[c], c))

    part2_budget = token_budget
    if grounding_budget is not None:
        part2_budget = min(token_budget, used_tokens + max(0, int(grounding_budget)))

    emitted = 0
    for cid in cited_ids:
        chunk = by_chunk.get(cid)
        if chunk is None or not (chunk.content or "").strip():
            continue
        block = f"{format_doc_id(chunk.source_doc_id)} {chunk.content.strip()}"
        block_tokens = _count_tokens(block + "\n")
        if used_tokens + block_tokens > part2_budget:
            # ADR-055: empty Part 2 ⇒ empty Verifier allowlist ⇒ every cite fails.
            # If grounding_budget alone would silence all chunks, still emit the
            # first block within the 80k hard ceiling (Lost-in-the-Middle clamp).
            if (
                emitted == 0
                and grounding_budget is not None
                and used_tokens + block_tokens <= token_budget
            ):
                lines.append(block)
                used_tokens += block_tokens
                emitted = 1
            lines.append(TRUNCATED_MARKER)
            break
        lines.append(block)
        used_tokens += block_tokens
        emitted += 1
    if emitted == 0 and TRUNCATED_MARKER not in lines:
        lines.append("(no linked chunks)")

    return "\n".join(lines) + "\n"


def chapter_grounding_budget(
    member_ids: List[str],
    nodes: List[GraphNode],
    chunks: List[ChunkMeta],
) -> int | None:
    """
    Production grounding_budget (ADR-050 v3): raw tokens of the chapter's
    unique cited chunks, scaled by GROUNDING_BUDGET_FRACTION.

    Returns None when no chunk bodies are available (raw == 0) so callers
    fall back to the plain 80k ceiling instead of a zero budget that would
    silence Part 2 entirely.
    """
    members = set(member_ids)
    by_chunk = _chunk_lookup(chunks)
    raw_tokens = 0
    seen: set[str] = set()
    for n in nodes:
        if n.id not in members:
            continue
        for cid in n.source_chunk_ids:
            if cid in seen:
                continue
            seen.add(cid)
            chunk = by_chunk.get(cid)
            if chunk is not None and (chunk.content or "").strip():
                raw_tokens += _count_tokens(chunk.content)
    if raw_tokens <= 0:
        return None
    try:
        grounding_fraction = get_rules().repomap.grounding_budget_fraction
    except Exception:  # noqa: BLE001 — chapter_grounding_budget must never itself crash a chapter
        grounding_fraction = GROUNDING_BUDGET_FRACTION
    return int(raw_tokens * grounding_fraction)


def repomap_node(state: GraphDistillerState) -> dict:
    chapter_id = state.get("current_chapter_id")
    if chapter_id is None:
        return {"repo_map": ""}

    member_ids = list((state.get("toc") or {}).get(chapter_id) or [])
    nodes = list(state.get("nodes") or [])
    chunks = list(state.get("chunks") or [])
    repo_map = build_repo_map(
        chapter_id=chapter_id,
        member_ids=member_ids,
        nodes=nodes,
        chunks=chunks,
        grounding_budget=chapter_grounding_budget(member_ids, nodes, chunks),
    )
    logger.debug(
        "RepoMap: chapter_id=%s tokens≈%s",
        chapter_id,
        _count_tokens(repo_map),
    )
    return {"repo_map": repo_map}
