"""Persist Phase 1 chunk bodies (ADR-049).

Phase 2 checkpoint is nodes/edges only (Phase isolation). RepoMap / Blindfold
need the original ChunkMeta.content keyed by chunk_id. Without this file,
uuid4 ids make bodies unrecoverable after the process exits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from graph_distiller.schema import ChunkMeta

PHASE1_CHUNKS_FILENAME = "phase1_chunks.json"


def save_chunks_checkpoint(
    chunks: List[ChunkMeta],
    checkpoint_dir: Path,
    filename: str = PHASE1_CHUNKS_FILENAME,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    payload = {"chunks": [c.model_dump() for c in chunks]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_chunks_checkpoint(
    checkpoint_dir: Path,
    filename: str = PHASE1_CHUNKS_FILENAME,
) -> List[ChunkMeta]:
    path = checkpoint_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ChunkMeta.model_validate(c) for c in data.get("chunks", [])]
