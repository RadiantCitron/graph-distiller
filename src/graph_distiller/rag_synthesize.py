"""Phase D2a: run Phase 4 synthesis on a rag_rebuild export (sample only).

Writes Living Codex under ``<export_dir>/codex/`` — does not touch the main
``output/codex/`` book path unless the caller passes that directory.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from graph_distiller.agents.codex_writer import save_codex
from graph_distiller.agents.workflow import run_phase4
from graph_distiller.graph.build import select_top_communities
from graph_distiller.schema import (
    ChunkMeta,
    GraphEdge,
    GraphNode,
    GraphDistillerState,
    empty_state,
)

logger = logging.getLogger(__name__)


@dataclass
class SynthesizeResult:
    export_dir: Path
    codex_dir: Path
    written: List[str] = field(default_factory=list)
    drafts: int = 0
    poisoned: List[int] = field(default_factory=list)
    phase4_usage: Dict[str, Any] = field(default_factory=dict)
    chapters_kept: int = 0


def load_state_from_export(export_dir: Path | str) -> GraphDistillerState:
    """Rebuild GraphDistillerState from rag_rebuild JSON (nodes/edges/toc/chunks)."""
    root = Path(export_dir)
    nodes_raw = json.loads((root / "nodes.json").read_text(encoding="utf-8"))
    edges_raw = json.loads((root / "edges.json").read_text(encoding="utf-8"))
    toc_raw = json.loads((root / "toc.json").read_text(encoding="utf-8"))
    chunks_raw = json.loads((root / "chunks.json").read_text(encoding="utf-8"))

    state = empty_state()
    state["nodes"] = [GraphNode.model_validate(n) for n in nodes_raw]
    state["edges"] = [GraphEdge.model_validate(e) for e in edges_raw]
    state["chunks"] = [ChunkMeta.model_validate(c) for c in chunks_raw]
    state["toc"] = {int(k): list(v) for k, v in (toc_raw or {}).items()}
    return state


def cap_toc_chapters(
    state: GraphDistillerState,
    max_chapters: int,
) -> GraphDistillerState:
    """Keep only top-N communities by size (cost control for sample Phase 4)."""
    if max_chapters <= 0:
        raise ValueError("max_chapters must be >= 1")
    toc = dict(state.get("toc") or {})
    if len(toc) <= max_chapters:
        return state
    errors: List[str] = list(state.get("errors") or [])
    state["toc"] = select_top_communities(toc, top_k=max_chapters, errors=errors)
    state["errors"] = errors
    # Drop drafts/titles that no longer apply
    state["drafts"] = []
    state["chapter_titles"] = {}
    state["poisoned_chapters"] = []
    state["retry_counts"] = {}
    state["current_chapter_id"] = None
    return state


def run_rag_synthesize(
    export_dir: Path | str,
    *,
    max_chapters: int = 2,
    codex_subdir: str = "codex",
) -> SynthesizeResult:
    """
    Phase 4 on export state → ``<export_dir>/codex/``.

    Costs money (SYNTHESIS_MODEL + EXTRACTION_MODEL titles). Sample-sized TOC only.
    """
    root = Path(export_dir).resolve()
    if not (root / "nodes.json").is_file():
        raise FileNotFoundError(f"Not a rag_rebuild export: {root}")

    state = load_state_from_export(root)
    state = cap_toc_chapters(state, max_chapters=max_chapters)
    kept = len(state.get("toc") or {})
    if kept == 0:
        raise RuntimeError("No communities in TOC — cannot synthesize.")

    logger.info(
        "RAG synthesize: export=%s chapters=%s nodes=%s",
        root.name,
        kept,
        len(state.get("nodes") or []),
    )
    state, usage = run_phase4(state)
    codex_dir = root / codex_subdir
    written_paths, _archive = save_codex(state, codex_dir)

    usage_dict = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "total_cost_usd": usage.total_cost_usd,
    }
    # Append synthesize meta next to export
    meta_path = root / "synthesize_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "max_chapters": max_chapters,
                "chapters_kept": kept,
                "drafts": len(state.get("drafts") or []),
                "poisoned": list(state.get("poisoned_chapters") or []),
                "phase4_usage": usage_dict,
                "codex_dir": str(codex_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return SynthesizeResult(
        export_dir=root,
        codex_dir=codex_dir,
        written=[str(p) for p in written_paths],
        drafts=len(state.get("drafts") or []),
        poisoned=list(state.get("poisoned_chapters") or []),
        phase4_usage=usage_dict,
        chapters_kept=kept,
    )
