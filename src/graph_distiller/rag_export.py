"""Export RAG skeleton artifacts to output/rag_rebuild/<run_id>/ (ADR-033/034).

Files only — no public.* WRITE. Ledger stays meta-only (no chapter bodies).
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from graph_distiller.config import PROJECT_ROOT, get_settings
from graph_distiller.ingestion.rdb import EdgeMeta, SamplePayload, counts_as_dict
from graph_distiller.schema import ChunkMeta, GraphEdge, GraphNode, Phase3Stats

DEFAULT_RAG_REBUILD_ROOT = PROJECT_ROOT / "output" / "rag_rebuild"


def make_rag_run_id(when: Optional[datetime] = None) -> str:
    ts = when or datetime.now(timezone.utc)
    return f"rag_{ts.strftime('%Y%m%d_%H%M%S')}"


def create_export_dir(
    root: Path | None = None,
    *,
    run_id: str | None = None,
) -> Path:
    base = Path(root) if root is not None else get_settings().rag_rebuild_dir
    rid = run_id or make_rag_run_id()
    path = base / rid
    path.mkdir(parents=True, exist_ok=False)
    return path


def _dump(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def export_skeleton(
    out_dir: Path,
    *,
    sample: SamplePayload,
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    edges_meta: Sequence[EdgeMeta],
    toc: Mapping[int, List[str]],
    chunks: Sequence[ChunkMeta],
    phase3_stats: Phase3Stats | Mapping[str, Any] | None,
    llm_mode: str,
    extra_meta: Mapping[str, Any] | None = None,
) -> Dict[str, str]:
    """
    Write JSON artifacts. Returns map of logical name → relative path string.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = out_dir / "nodes.json"
    edges_path = out_dir / "edges.json"
    meta_edges_path = out_dir / "edges_meta.json"
    toc_path = out_dir / "toc.json"
    chunks_path = out_dir / "chunks.json"
    source_map_path = out_dir / "source_map.json"
    run_meta_path = out_dir / "meta.json"
    sql_path = out_dir / "nodes_edges.sql"

    _dump(nodes_path, [n.model_dump() for n in nodes])
    _dump(edges_path, [e.model_dump() for e in edges])
    _dump(
        meta_edges_path,
        [asdict(m) if is_dataclass(m) else dict(m) for m in edges_meta],
    )
    _dump(toc_path, {str(k): list(v) for k, v in toc.items()})
    # Full chunk text is intentional for rebuild files (not ledger).
    _dump(chunks_path, [c.model_dump() for c in chunks])

    source_map = {
        "law_jo": [
            {
                "node_id": b.node_id,
                "labor_law_id": b.jo_row_id,
                "jang_id": b.jang_id,
                "descendant_ids": list(b.descendant_ids),
                "source_doc_id": b.source_doc_id,
                "chunk_id": b.chunk_id,
            }
            for b in sample.jo_bundles
        ],
        "qna_case": [
            {
                "node_id": c.node_id,
                "labor_qna_id": c.case_id,
                "source_doc_id": c.source_doc_id,
                "chunk_id": c.chunk_id,
                "cite_hits": c.cite_hits,
            }
            for c in sample.qna_cases
        ],
    }
    _dump(source_map_path, source_map)

    meta: Dict[str, Any] = {
        "purpose": "지식체계압축제공",
        "llm_mode": llm_mode,
        "sample_n_jo": sample.sample_n_jo,
        "sample_m_qna": sample.sample_m_qna,
        "counts": counts_as_dict(sample.counts),
        "selection_notes": list(sample.selection_notes),
        "phase3_stats": dict(phase3_stats or {}),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "chunk_count": len(chunks),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra_meta:
        meta.update(dict(extra_meta))
    _dump(run_meta_path, meta)

    sql_path.write_text(
        _sql_stub(nodes, edges),
        encoding="utf-8",
    )

    return {
        "nodes": str(nodes_path),
        "edges": str(edges_path),
        "edges_meta": str(meta_edges_path),
        "toc": str(toc_path),
        "chunks": str(chunks_path),
        "source_map": str(source_map_path),
        "meta": str(run_meta_path),
        "sql": str(sql_path),
    }


def _sql_stub(nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> str:
    """Comment-only SQL preview for later graph_distiller.* load (Phase D — not executed)."""
    lines = [
        "-- Preview only. Do NOT run against public.*. Phase D uses graph_distiller schema.",
        "-- CREATE SCHEMA IF NOT EXISTS graph_distiller;",
        f"-- nodes={len(nodes)} edges={len(edges)}",
        "",
    ]
    for n in nodes[:50]:
        safe_id = n.id.replace("'", "''")
        lines.append(
            f"-- INSERT INTO graph_distiller.nodes(id, pagerank) VALUES "
            f"('{safe_id}', {float(n.pagerank_score):.8f});"
        )
    if len(nodes) > 50:
        lines.append(f"-- ... {len(nodes) - 50} more nodes omitted")
    for e in edges[:50]:
        s = e.source.replace("'", "''")
        t = e.target.replace("'", "''")
        prov = str(getattr(e, "provenance", "") or "llm").replace("'", "''")
        lines.append(
            f"-- INSERT INTO graph_distiller.edges(source, target, weight, provenance) VALUES "
            f"('{s}', '{t}', {float(e.weight)}, '{prov}');"
        )
    if len(edges) > 50:
        lines.append(f"-- ... {len(edges) - 50} more edges omitted")
    return "\n".join(lines) + "\n"
