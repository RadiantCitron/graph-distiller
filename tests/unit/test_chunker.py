"""Unit: chunk-level source_doc_id uniqueness (ADR-057 조치4) — no network / $0."""

from __future__ import annotations

import itertools

import pytest

from graph_distiller.config import get_settings
from graph_distiller.ingestion.chunker import (
    _force_split,
    _next_chunk_source_doc_id,
    chunk_documents,
)

pytestmark = pytest.mark.unit


class _ConstantEmbedder:
    """Every text maps to the same vector, so SemanticChunker keeps one segment."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_next_chunk_source_doc_id_increments_from_shared_counter() -> None:
    seq = itertools.count()
    assert _next_chunk_source_doc_id("book", seq) == "book_0000"
    assert _next_chunk_source_doc_id("book", seq) == "book_0001"
    assert _next_chunk_source_doc_id("book", seq) == "book_0002"


def test_force_split_gives_each_piece_a_distinct_source_doc_id() -> None:
    text = "가나다. " * 400  # long enough to force RecursiveCharacterTextSplitter to split
    seq = itertools.count()
    chunks = _force_split(text, "memo1", seq, max_tokens=50)

    assert len(chunks) > 1
    doc_ids = [c.source_doc_id for c in chunks]
    assert len(doc_ids) == len(set(doc_ids)), "each split piece must get a unique id"
    assert all(d.startswith("memo1_") for d in doc_ids)
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_chunk_documents_assigns_unique_source_doc_id_per_chunk_per_document() -> None:
    """
    ADR-057 조치4: before this, every chunk from the same file shared one
    file-level source_doc_id, so [DOC_ID: ...] couldn't tell which passage
    was actually cited. Now each chunk gets its own id, still traceable to
    the file via a shared prefix, and a second document restarts its own
    sequence independently.
    """
    documents = [
        {"source_doc_id": "book_a", "content": "문장 하나. 문장 둘. 문장 셋. 문장 넷."},
        {"source_doc_id": "book_b", "content": "다른 책의 문장. 또 다른 문장."},
    ]
    chunks = chunk_documents(documents, get_settings(), embedder=_ConstantEmbedder())

    assert len(chunks) >= 2
    doc_ids = [c.source_doc_id for c in chunks]
    assert len(doc_ids) == len(set(doc_ids)), "source_doc_id must be unique per chunk"

    book_a_ids = [d for d in doc_ids if d.startswith("book_a_")]
    book_b_ids = [d for d in doc_ids if d.startswith("book_b_")]
    assert book_a_ids and book_b_ids
    # Each document's own chunk sequence restarts at 0 independently.
    assert min(book_a_ids) == "book_a_0000"
    assert min(book_b_ids) == "book_b_0000"
