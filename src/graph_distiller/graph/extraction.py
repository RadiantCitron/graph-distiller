"""Phase 2: LLM entity/relation extraction via OpenRouter ChatOpenAI."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import HumanMessage, SystemMessage
from openai import RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graph_distiller.config import Settings, get_settings
from graph_distiller.contracts import TAG_DLQ, TAG_FATAL
from graph_distiller.graph.checkpoint import save_extraction_checkpoint
from graph_distiller.ingestion.checkpoint import save_chunks_checkpoint
from graph_distiller.graph.lexicon import (
    lexicon_prompt_hint,
    load_entity_alias_lexicon,
    rewrite_ids_with_lexicon,
)
from graph_distiller.observability import (
    PhaseUsage,
    emit_loop_error,
    make_phase_progress,
    quiet_console_logs,
)
from graph_distiller.prompts import load_prompt
from graph_distiller.rules import get_rules
from graph_distiller.schema import (
    ChunkMeta,
    ExtractionResult,
    GraphEdge,
    GraphNode,
    GraphDistillerState,
    empty_state,
)

logger = logging.getLogger(__name__)

# Unmeasured placeholder (ADR-042). Not a quality gate. Override via Settings.
DEFAULT_CANON_EMBED_SIM_THRESHOLD = 0.90
DEFAULT_CANON_EMBED_MAX_NODES = 2000


class _Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]: ...

EXTRACTION_SYSTEM_PROMPT = """\
You extract domain-specific entities and relationships from knowledge memos.

MUST rules:
1. Never extract generic nouns (e.g. system, company, user, data) as nodes.
   Extract only domain-specific proper nouns / named concepts. (anti super-node)
2. Normalize every node id: replace spaces with underscore (_), UPPERCASE.
3. Node type must be exactly one of: CONCEPT, PERSON, SYSTEM.
4. Include the given chunk_id in each node's source_chunk_ids.
5. Edge source/target must reference node ids present in the nodes list.
6. One concept = one id. Do not emit near-duplicates that differ only by
   spacing, hyphen, or underscore (e.g. NA_JASIN vs NAJASIN).
7. Edges are as important as nodes. Densely connect the extracted nodes:
   every node MUST appear in at least one edge. An isolated node is a
   FAILURE — actively link each node to the chunk's core theme or to its
   sibling nodes via a meaningful relation (e.g. USES, CAUSES, PART_OF,
   CONTRASTS_WITH, RELATES_TO).
8. Prefer edge-rich output: target edge count >= node count. Do not stop
   after listing entities; always ask "how does this node relate to the
   others in this chunk?" before finishing.
"""


def _normalize_node_id(raw_id: str) -> str:
    return re.sub(r"\s+", "_", raw_id.strip()).upper()


def _id_fingerprint(node_id: str) -> str:
    """Alphanumeric-only key so NA_JASIN and NAJASIN collide."""
    return re.sub(r"[^A-Z0-9가-힣]", "", node_id.upper())


def _id_tokens(node_id: str) -> frozenset[str]:
    parts = re.split(r"[^A-Z0-9가-힣]+", node_id.upper())
    return frozenset(p for p in parts if p)


def _token_jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _are_alias_ids(
    left: str,
    right: str,
    *,
    min_fingerprint_len: int = 4,
    min_token_len: int = 2,
    jaccard_threshold: float = 0.8,
) -> bool:
    """
    Conservative alias test (ADR-025 stage 1).

    True only when ids are the same concept under spelling noise.
    Does not merge mere substrings (KOREA vs KOREA_AND_FRANCE).

    Keyword defaults are fallbacks — the live thresholds are
    data/rules/phase2_canonicalize.toml (ADR-060); canonicalize_extracted_graph
    passes them in explicitly.
    """
    if left == right:
        return True
    fa, fb = _id_fingerprint(left), _id_fingerprint(right)
    if fa and fa == fb and len(fa) >= min_fingerprint_len:
        return True
    ta, tb = _id_tokens(left), _id_tokens(right)
    if min(len(ta), len(tb)) >= min_token_len and _token_jaccard(ta, tb) >= jaccard_threshold:
        return True
    return False


def _is_proper_token_subset_pair(left: str, right: str) -> bool:
    """True when one id's tokens are a proper subset of the other's."""
    ta, tb = _id_tokens(left), _id_tokens(right)
    if not ta or not tb or ta == tb:
        return False
    return ta < tb or tb < ta


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for a, b in zip(left, right):
        fa, fb = float(a), float(b)
        dot += fa * fb
        na += fa * fa
        nb += fb * fb
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)


def _union_by_embedding(
    ids: List[str],
    node_map: Dict[str, GraphNode],
    find,
    union,
    *,
    embeddings: Optional[_Embedder],
    sim_threshold: float,
    max_embed_nodes: int,
) -> None:
    """Reduce 2: merge remaining roots by cosine similarity (ADR-042)."""
    if embeddings is None:
        logger.warning("embed canonicalize skipped: no embeddings client")
        return
    if len(ids) > max_embed_nodes:
        logger.warning(
            "embed canonicalize skipped: %s nodes > max_embed_nodes=%s",
            len(ids),
            max_embed_nodes,
        )
        return
    roots = sorted({find(i) for i in ids})
    if len(roots) < 2:
        return
    texts = [f"{nid} {node_map[nid].description}" for nid in roots]
    try:
        vectors = embeddings.embed_documents(texts)
    except Exception as exc:  # noqa: BLE001 — graceful degrade to Reduce 1
        logger.warning("embed canonicalize skipped: %s", exc)
        return
    if len(vectors) != len(roots):
        logger.warning(
            "embed canonicalize skipped: vector count %s != roots %s",
            len(vectors),
            len(roots),
        )
        return
    merged_pairs = 0
    for i, left in enumerate(roots):
        for j, right in enumerate(roots[i + 1 :], start=i + 1):
            if find(left) == find(right):
                continue
            n_left, n_right = node_map[left], node_map[right]
            if n_left.type != n_right.type:
                continue
            if _is_proper_token_subset_pair(left, right):
                continue
            if _cosine(vectors[i], vectors[j]) < sim_threshold:
                continue
            union(left, right)
            merged_pairs += 1
    if merged_pairs:
        logger.info(
            "Phase 2 embed canonicalize: unioned %d pairs (threshold=%s)",
            merged_pairs,
            sim_threshold,
        )


def _maybe_embedder(settings: Settings) -> Optional[_Embedder]:
    if not settings.canon_embed_merge:
        return None
    try:
        from graph_distiller.adapters.openrouter import default_embedder

        return default_embedder(settings)  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.warning("embed canonicalize skipped: cannot build embeddings (%s)", exc)
        return None


def _pick_canonical_id(ids: List[str], node_map: Dict[str, GraphNode]) -> str:
    """Prefer the id seen in the most chunks, then the longest spelling."""

    def key(nid: str) -> Tuple[int, int, str]:
        node = node_map[nid]
        return (len(node.source_chunk_ids), len(nid), nid)

    return max(ids, key=key)


def canonicalize_extracted_graph(
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    *,
    embed_merge: bool = False,
    embeddings: Optional[_Embedder] = None,
    sim_threshold: float = DEFAULT_CANON_EMBED_SIM_THRESHOLD,
    max_embed_nodes: int = DEFAULT_CANON_EMBED_MAX_NODES,
    alias_lexicon: Optional[Dict[str, str]] = None,
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    """
    Second pass after parallel per-chunk extraction (ADR-025 / ADR-042).

    Reduce 1: spelling aliases. Reduce 3 (W2-D): domain lexicon rewrite.
    Reduce 2: optional embedding merge (off by default).
    """
    if not nodes:
        return [], list(edges)

    if alias_lexicon:
        nodes, edges = rewrite_ids_with_lexicon(nodes, edges, alias_lexicon)

    node_map: Dict[str, GraphNode] = {}
    _merge_nodes(node_map, nodes)
    ids = list(node_map.keys())
    parent = {i: i for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    canon_rules = get_rules().canonicalize
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            if _are_alias_ids(
                left,
                right,
                min_fingerprint_len=canon_rules.min_fingerprint_len,
                min_token_len=canon_rules.min_token_len,
                jaccard_threshold=canon_rules.jaccard_threshold,
            ):
                union(left, right)

    if embed_merge:
        _union_by_embedding(
            ids,
            node_map,
            find,
            union,
            embeddings=embeddings,
            sim_threshold=sim_threshold,
            max_embed_nodes=max_embed_nodes,
        )

    clusters: Dict[str, List[str]] = {}
    for nid in ids:
        clusters.setdefault(find(nid), []).append(nid)

    alias_to_canon: Dict[str, str] = {}
    merged: Dict[str, GraphNode] = {}
    for members in clusters.values():
        canon = _pick_canonical_id(members, node_map)
        chunk_ids: List[str] = []
        description = node_map[canon].description
        node_type = node_map[canon].type
        for mid in members:
            alias_to_canon[mid] = canon
            prev = node_map[mid]
            chunk_ids.extend(prev.source_chunk_ids)
            if len(prev.description) > len(description):
                description = prev.description
                node_type = prev.type
        merged[canon] = node_map[canon].model_copy(
            update={
                "id": canon,
                "type": node_type,
                "description": description,
                "source_chunk_ids": list(dict.fromkeys(chunk_ids)),
            }
        )

    collapsed: Dict[Tuple[str, str], GraphEdge] = {}
    for edge in edges:
        orig_src = _normalize_node_id(edge.source)
        orig_tgt = _normalize_node_id(edge.target)
        src = alias_to_canon.get(orig_src, orig_src)
        tgt = alias_to_canon.get(orig_tgt, orig_tgt)
        if src not in merged or tgt not in merged:
            continue
        if src == tgt:
            continue
        remapped = src != orig_src or tgt != orig_tgt
        # Undirected (matches Phase 3 nx.Graph): A→B and B→A share a weight bucket.
        a, b = (src, tgt) if src <= tgt else (tgt, src)
        key = (a, b)
        if key in collapsed:
            prev = collapsed[key]
            collapsed[key] = prev.model_copy(
                update={
                    "weight": float(prev.weight) + float(edge.weight),
                    # Weight gravity from alias collapse → alias-merge channel (W2-B)
                    "provenance": "alias-merge",
                }
            )
        else:
            prov = (
                "alias-merge"
                if remapped
                else ((edge.provenance or "llm").strip() or "llm")
            )
            collapsed[key] = edge.model_copy(
                update={
                    "source": a,
                    "target": b,
                    "weight": float(edge.weight),
                    "provenance": prov,
                }
            )

    merged_alias_count = sum(1 for m in clusters.values() if len(m) > 1)
    if merged_alias_count:
        logger.info(
            "Phase 2 alias canonicalize: nodes %d -> %d (clusters_merged=%d)",
            len(node_map),
            len(merged),
            merged_alias_count,
        )
    return list(merged.values()), list(collapsed.values())


def _merge_nodes(existing: Dict[str, GraphNode], incoming: List[GraphNode]) -> None:
    for node in incoming:
        node.id = _normalize_node_id(node.id)
        if node.id in existing:
            prev = existing[node.id]
            merged_ids = list(
                dict.fromkeys(prev.source_chunk_ids + node.source_chunk_ids)
            )
            existing[node.id] = prev.model_copy(
                update={"source_chunk_ids": merged_ids}
            )
        else:
            existing[node.id] = node


def _dlq(
    errors: List[str],
    source_doc_id: str,
    chunk_id: str,
    reason: str,
) -> None:
    msg = f"{TAG_DLQ} source_doc_id={source_doc_id} chunk_id={chunk_id} reason={reason}"
    errors.append(msg)
    emit_loop_error(msg, level=logging.WARNING)


class GraphExtractor:
    """Async batch extractor: Semaphore(<=20) + tenacity backoff + OpenRouter."""

    def __init__(
        self,
        settings: Settings,
        *,
        llm: "LlmChat | None" = None,
    ) -> None:
        from graph_distiller.adapters.openrouter import default_llm_chat
        from graph_distiller.ports import LlmChat  # noqa: F401

        self.settings = settings
        self.llm = llm if llm is not None else default_llm_chat(
            settings, model=settings.extraction_model
        )
        self.structured = self.llm.with_structured_output(ExtractionResult)
        self._sem = asyncio.Semaphore(settings.llm_concurrency)
        self._alias_lexicon = load_entity_alias_lexicon(
            getattr(settings, "canon_alias_lexicon_path", None)
        )
        base_prompt = load_prompt(
            settings.prompts_dir,
            "phase2_extraction.md",
            fallback=EXTRACTION_SYSTEM_PROMPT,
        )
        self._system_prompt = base_prompt + lexicon_prompt_hint(self._alias_lexicon)

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(8),
        reraise=True,
    )
    async def _call_llm(self, chunk: ChunkMeta) -> ExtractionResult:
        user_prompt = (
            f"chunk_id: {chunk.chunk_id}\n"
            f"source_doc_id: {chunk.source_doc_id}\n\n"
            f"--- CONTENT ---\n{chunk.content}"
        )
        parsed = await self.structured.ainvoke(
            [
                SystemMessage(content=self._system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        if parsed is None:
            raise ValueError("Empty structured output from LLM")
        if not isinstance(parsed, ExtractionResult):
            parsed = ExtractionResult.model_validate(parsed)

        for node in parsed.nodes:
            node.id = _normalize_node_id(node.id)
            if chunk.chunk_id not in node.source_chunk_ids:
                node.source_chunk_ids.append(chunk.chunk_id)
        for edge in parsed.edges:
            edge.source = _normalize_node_id(edge.source)
            edge.target = _normalize_node_id(edge.target)
        return parsed

    async def _extract_one(
        self, chunk: ChunkMeta, errors: List[str]
    ) -> Optional[ExtractionResult]:
        async with self._sem:
            try:
                return await self._call_llm(chunk)
            except Exception as exc:  # noqa: BLE001 — DLQ path
                _dlq(errors, chunk.source_doc_id, chunk.chunk_id, str(exc))
                return None

    async def extract_all(
        self, chunks: List[ChunkMeta], errors: List[str]
    ) -> Tuple[List[GraphNode], List[GraphEdge], PhaseUsage]:
        """
        Concurrent chunk extraction with a live progress bar showing running
        USD cost as each chunk finishes (not just a total at the end).
        """
        results: List[Optional[ExtractionResult]] = []
        usage = PhaseUsage()
        with quiet_console_logs():
            with get_openai_callback() as cb:
                tasks = [
                    asyncio.ensure_future(self._extract_one(c, errors))
                    for c in chunks
                ]
                pbar = make_phase_progress(len(chunks), desc="Phase 2", unit="chunk")
                try:
                    for coro in asyncio.as_completed(tasks):
                        results.append(await coro)
                        usage = PhaseUsage.from_callback(
                            cb, model=self.settings.extraction_model
                        )
                        pbar.set_postfix_str(usage.cost_postfix)
                        pbar.update(1)
                finally:
                    pbar.close()

        logger.debug(
            "Phase 2 tokens: in=%s out=%s total=%s cost=$%s",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
            usage.total_cost_usd,
        )

        node_map: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []
        for result in results:
            if result is None:
                continue
            _merge_nodes(node_map, result.nodes)
            edges.extend(result.edges)
        nodes, edges = canonicalize_extracted_graph(
            list(node_map.values()),
            edges,
            embed_merge=self.settings.canon_embed_merge,
            embeddings=_maybe_embedder(self.settings),
            sim_threshold=self.settings.canon_embed_sim_threshold,
            max_embed_nodes=self.settings.canon_embed_max_nodes,
            alias_lexicon=self._alias_lexicon,
        )
        return nodes, edges, usage


async def run_phase2_async(
    state: GraphDistillerState,
    settings: Settings | None = None,
    persist_checkpoint: bool = True,
) -> Tuple[GraphDistillerState, PhaseUsage]:
    cfg = settings or get_settings()
    chunks = list(state.get("chunks") or [])
    errors = list(state.get("errors") or [])
    empty_usage = PhaseUsage()

    if not chunks:
        state["nodes"] = []
        state["edges"] = []
        state["errors"] = errors + ["[WARN] Phase 2 skipped: no chunks in state."]
        return state, empty_usage

    if not cfg.openrouter_api_key:
        state["errors"] = errors + [
            f"{TAG_FATAL} OPENROUTER_API_KEY is not set; Phase 2 aborted."
        ]
        return state, empty_usage

    nodes, edges, usage = await GraphExtractor(cfg).extract_all(chunks, errors)
    state["nodes"] = nodes
    state["edges"] = edges
    state["errors"] = errors

    if persist_checkpoint:
        path = save_extraction_checkpoint(nodes, edges, cfg.checkpoint_dir)
        logger.debug("Phase 2 checkpoint saved: %s", path)
        # Align evidence with this graph: post-slice chunks share uuid4 ids
        # with node.source_chunk_ids (ADR-049). Phase 2 file stays nodes/edges.
        chunks_path = save_chunks_checkpoint(chunks, cfg.checkpoint_dir)
        logger.debug("Phase 1 checkpoint aligned: %s (%d chunks)", chunks_path, len(chunks))

    return state, usage


def run_phase2(
    state: GraphDistillerState | None = None,
    settings: Settings | None = None,
    persist_checkpoint: bool = True,
) -> Tuple[GraphDistillerState, PhaseUsage]:
    """Phase 2 sync entrypoint → (state, token usage)."""
    out = state if state is not None else empty_state()
    return asyncio.run(
        run_phase2_async(
            out, settings=settings, persist_checkpoint=persist_checkpoint
        )
    )
