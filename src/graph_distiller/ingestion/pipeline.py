"""Phase 1 pipeline: load raw memos → semantic chunk → GraphDistillerState."""

from __future__ import annotations

import logging
from typing import Optional

from graph_distiller.adapters.source import default_source_pull
from graph_distiller.config import Settings, get_settings
from graph_distiller.ingestion.checkpoint import save_chunks_checkpoint
from graph_distiller.ingestion.chunker import chunk_documents
from graph_distiller.ports import Embedder, SourcePull
from graph_distiller.schema import GraphDistillerState, empty_state

logger = logging.getLogger(__name__)


def run_phase1(
    state: GraphDistillerState | None = None,
    settings: Settings | None = None,
    *,
    source: Optional[SourcePull] = None,
    embedder: Optional[Embedder] = None,
    persist_checkpoint: bool = True,
) -> GraphDistillerState:
    """
    Phase 1 (Ingestion & Semantic Chunking).

    - Read all .txt/.md/.docx/.json under data/raw_memos/ (SourcePull)
    - SemanticChunker (percentile=80) via Embedder
    - Force-split any chunk > 1000 tokens
    - Store ChunkMeta list on state["chunks"]
    - Persist bodies to phase1_chunks.json (ADR-049) unless persist_checkpoint=False
    """
    cfg = settings or get_settings()
    out: GraphDistillerState = state if state is not None else empty_state()

    pull = source if source is not None else default_source_pull()
    raw_documents = pull.load_documents(cfg.raw_memos_dir)
    chunks = chunk_documents(raw_documents, cfg, embedder=embedder)

    out["raw_documents"] = raw_documents
    out["chunks"] = chunks
    if persist_checkpoint:
        path = save_chunks_checkpoint(chunks, cfg.checkpoint_dir)
        logger.debug("Phase 1 checkpoint saved: %s (%d chunks)", path, len(chunks))
    return out
