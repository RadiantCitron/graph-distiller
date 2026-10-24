"""SemanticChunker + token-cap enforcement (SPEC Level 3 §2)."""

from __future__ import annotations

import itertools
from typing import Iterator, List, Optional
from uuid import uuid4

import tiktoken
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from graph_distiller.adapters.openrouter import default_embedder
from graph_distiller.config import Settings
from graph_distiller.ports import Embedder
from graph_distiller.schema import ChunkMeta


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    return len(tiktoken.get_encoding(encoding_name).encode(text))


def _build_chunker(
    settings: Settings,
    embedder: Optional[Embedder] = None,
) -> SemanticChunker:
    embeddings = embedder if embedder is not None else default_embedder(settings)
    return SemanticChunker(
        embeddings,  # OpenRouterEmbedder delegates LangChain Embeddings API
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=float(settings.semantic_breakpoint_percentile),
    )


def _next_chunk_source_doc_id(doc_id: str, seq: Iterator[int]) -> str:
    """
    ADR-057 조치4: local ingestion previously gave every chunk from the same
    file the identical file-level source_doc_id, so a citation like
    ``[DOC_ID: sample_document]`` couldn't tell which passage was actually
    used — any chunk from that file satisfied the Verifier's ID check. This
    keeps the file stem as a human-readable prefix but makes the id unique
    per chunk, so a citation now points at one specific passage.
    """
    return f"{doc_id}_{next(seq):04d}"


def _force_split(
    text: str, doc_id: str, seq: Iterator[int], max_tokens: int
) -> List[ChunkMeta]:
    """MUST: chunks > 1000 tokens → RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=50,
        length_function=count_tokens,
    )
    out: List[ChunkMeta] = []
    for piece in splitter.split_text(text):
        tokens = count_tokens(piece)
        if tokens > max_tokens:
            enc = tiktoken.get_encoding("cl100k_base")
            piece = enc.decode(enc.encode(piece)[:max_tokens])
            tokens = count_tokens(piece)
        source_doc_id = _next_chunk_source_doc_id(doc_id, seq)
        out.append(
            ChunkMeta(
                chunk_id=f"{source_doc_id}_{uuid4().hex[:8]}",
                source_doc_id=source_doc_id,
                content=piece,
                token_count=tokens,
            )
        )
    return out


def chunk_documents(
    documents: List[dict],
    settings: Settings,
    *,
    embedder: Optional[Embedder] = None,
) -> List[ChunkMeta]:
    if not documents:
        return []

    chunker = _build_chunker(settings, embedder=embedder)
    chunks: List[ChunkMeta] = []

    for doc in documents:
        doc_id = doc["source_doc_id"]
        seq = itertools.count()
        for part in chunker.split_text(doc["content"]):
            tokens = count_tokens(part)
            if tokens > settings.max_chunk_tokens:
                chunks.extend(_force_split(part, doc_id, seq, settings.max_chunk_tokens))
            else:
                source_doc_id = _next_chunk_source_doc_id(doc_id, seq)
                chunks.append(
                    ChunkMeta(
                        chunk_id=f"{source_doc_id}_{uuid4().hex[:8]}",
                        source_doc_id=source_doc_id,
                        content=part,
                        token_count=tokens,
                    )
                )
    return chunks
