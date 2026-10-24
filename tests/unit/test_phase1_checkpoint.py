"""Phase 1 chunk checkpoint persist/load (ADR-049)."""

from __future__ import annotations

from pathlib import Path

from graph_distiller.ingestion.checkpoint import (
    PHASE1_CHUNKS_FILENAME,
    load_chunks_checkpoint,
    save_chunks_checkpoint,
)
from graph_distiller.schema import ChunkMeta


def test_roundtrip_chunk_bodies(tmp_path: Path) -> None:
    chunks = [
        ChunkMeta(
            chunk_id="doc_abcd1234",
            source_doc_id="doc",
            content="원문 본문 한 조각",
            token_count=5,
        )
    ]
    path = save_chunks_checkpoint(chunks, tmp_path)
    assert path.name == PHASE1_CHUNKS_FILENAME
    loaded = load_chunks_checkpoint(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].chunk_id == "doc_abcd1234"
    assert loaded[0].content == "원문 본문 한 조각"


def test_missing_file_raises(tmp_path: Path) -> None:
    try:
        load_chunks_checkpoint(tmp_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
