"""Phase 1 checkpoint statistics (data/checkpoints/phase1_chunks.json)."""

from __future__ import annotations

from pathlib import Path

from graph_distiller.ingestion.checkpoint import load_chunks_checkpoint

from harness.collectors import Collected

SCOPE = "chunks"


def collect(checkpoint_dir: Path, *, filename: str = "phase1_chunks.json") -> Collected:
    out = Collected()
    path = Path(checkpoint_dir) / filename
    if not out.require_input(SCOPE, path):
        return out

    chunks = load_chunks_checkpoint(checkpoint_dir, filename=filename)
    docs = {c.source_doc_id for c in chunks}
    tokens = [c.token_count for c in chunks]
    oversized = sum(1 for t in tokens if t > 1000)

    out.metric(SCOPE, "chunks", len(chunks), unit="count")
    out.metric(SCOPE, "source_docs", len(docs), unit="count")
    out.metric(SCOPE, "tokens_total", sum(tokens), unit="tokens")
    out.metric(SCOPE, "tokens_max", max(tokens) if tokens else 0, unit="tokens")
    out.metric(SCOPE, "tokens_mean", (sum(tokens) / len(tokens)) if tokens else 0.0, unit="tokens")
    out.metric(SCOPE, "empty_content", sum(1 for c in chunks if not c.content.strip()), unit="count")
    out.metric(SCOPE, "oversized_over_1000_tokens", oversized, unit="count")

    out.check(
        "chunks.no_oversized_chunks",
        SCOPE,
        oversized == 0,
        severity="error",
        message="" if oversized == 0 else f"{oversized} chunk(s) exceed the 1000-token hard cap (SPEC Phase 1 MUST)",
    )
    ids = [c.chunk_id for c in chunks]
    dupes = len(ids) - len(set(ids))
    out.check("chunks.unique_chunk_ids", SCOPE, dupes == 0, severity="error",
              message="" if dupes == 0 else f"{dupes} duplicate chunk_id(s)")
    return out
