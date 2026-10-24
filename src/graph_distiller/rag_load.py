"""Phase D1: load rag_rebuild JSON into PostgreSQL schema ``graph_distiller`` (ADR-034).

Hard rules:
- CREATE/INSERT only under schema ``graph_distiller``.
- Never WRITE ``public.*`` (labor_law / labor_qna / …).
- Source of truth remains export files; DB is a mirror for downstream RAG.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from graph_distiller.ingestion.rdb import connect
from graph_distiller.rag_write_guard import apply_write_search_path, execute_graph_distiller

logger = logging.getLogger(__name__)

SCHEMA = "graph_distiller"

DDL = f"""
CREATE SCHEMA IF NOT EXISTS {SCHEMA};

CREATE TABLE IF NOT EXISTS {SCHEMA}.runs (
    run_id          TEXT PRIMARY KEY,
    export_dir      TEXT NOT NULL,
    llm_mode        TEXT NOT NULL DEFAULT '',
    sample_n_jo     INTEGER,
    sample_m_qna    INTEGER,
    meta_json       JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    loaded_at_utc   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.nodes (
    run_id          TEXT NOT NULL REFERENCES {SCHEMA}.runs(run_id) ON DELETE CASCADE,
    node_id         TEXT NOT NULL,
    node_type       TEXT NOT NULL DEFAULT 'CONCEPT',
    description     TEXT NOT NULL DEFAULT '',
    pagerank_score  DOUBLE PRECISION NOT NULL DEFAULT 0,
    source_chunk_ids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    PRIMARY KEY (run_id, node_id)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.edges (
    run_id          TEXT NOT NULL REFERENCES {SCHEMA}.runs(run_id) ON DELETE CASCADE,
    source          TEXT NOT NULL,
    target          TEXT NOT NULL,
    relationship    TEXT NOT NULL DEFAULT '',
    weight          DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    provenance      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, source, target, relationship)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.chapters (
    run_id          TEXT NOT NULL REFERENCES {SCHEMA}.runs(run_id) ON DELETE CASCADE,
    chapter_id      INTEGER NOT NULL,
    member_node_ids TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    PRIMARY KEY (run_id, chapter_id)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.source_map (
    run_id          TEXT NOT NULL REFERENCES {SCHEMA}.runs(run_id) ON DELETE CASCADE,
    kind            TEXT NOT NULL,
    node_id         TEXT NOT NULL,
    payload_json    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    PRIMARY KEY (run_id, kind, node_id)
);

CREATE INDEX IF NOT EXISTS graph_distiller_nodes_pagerank_idx
    ON {SCHEMA}.nodes (run_id, pagerank_score DESC);
CREATE INDEX IF NOT EXISTS graph_distiller_edges_src_idx
    ON {SCHEMA}.edges (run_id, source);
CREATE INDEX IF NOT EXISTS graph_distiller_edges_tgt_idx
    ON {SCHEMA}.edges (run_id, target);
"""


@dataclass(frozen=True)
class LoadResult:
    run_id: str
    export_dir: str
    nodes: int
    edges: int
    chapters: int
    source_map_rows: int


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_graph_distiller_schema(conn) -> None:
    """Idempotent DDL. Touches only ``graph_distiller`` schema (E2 write guard)."""
    apply_write_search_path(conn)
    with conn.cursor() as cur:
        execute_graph_distiller(cur, DDL)
    conn.commit()
    logger.info("Ensured schema %s and tables", SCHEMA)


def _assert_export_dir(export_dir: Path) -> Path:
    root = Path(export_dir).resolve()
    required = ("nodes.json", "edges.json", "toc.json", "meta.json", "source_map.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Export dir incomplete ({root}): missing {missing}")
    return root


def load_export_dir(
    uri: str,
    export_dir: Path | str,
    *,
    run_id: str | None = None,
    replace: bool = True,
) -> LoadResult:
    """
    Load one ``output/rag_rebuild/<run_id>/`` folder into ``graph_distiller.*``.

    ``replace=True`` deletes prior rows for the same run_id then re-inserts.
    """
    root = _assert_export_dir(Path(export_dir))
    rid = run_id or root.name
    nodes = _read_json(root / "nodes.json")
    edges = _read_json(root / "edges.json")
    toc = _read_json(root / "toc.json")
    meta = _read_json(root / "meta.json")
    source_map = _read_json(root / "source_map.json")
    edges_meta_path = root / "edges_meta.json"
    edges_meta: List[Dict[str, Any]] = []
    if edges_meta_path.is_file():
        edges_meta = _read_json(edges_meta_path)

    # provenance lookup (undirected-ish: first wins)
    prov: Dict[tuple[str, str], str] = {}
    for row in edges_meta:
        s, t = str(row.get("source", "")), str(row.get("target", ""))
        if s and t:
            prov[(s, t)] = str(row.get("provenance") or "cites")
            prov[(t, s)] = str(row.get("provenance") or "cites")

    conn = connect(uri)
    try:
        ensure_graph_distiller_schema(conn)
        apply_write_search_path(conn)
        with conn.cursor() as cur:
            if replace:
                execute_graph_distiller(
                    cur,
                    f"DELETE FROM {SCHEMA}.runs WHERE run_id = %s",
                    (rid,),
                )
            execute_graph_distiller(
                cur,
                f"""
                INSERT INTO {SCHEMA}.runs
                    (run_id, export_dir, llm_mode, sample_n_jo, sample_m_qna, meta_json, loaded_at_utc)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    export_dir = EXCLUDED.export_dir,
                    llm_mode = EXCLUDED.llm_mode,
                    sample_n_jo = EXCLUDED.sample_n_jo,
                    sample_m_qna = EXCLUDED.sample_m_qna,
                    meta_json = EXCLUDED.meta_json,
                    loaded_at_utc = EXCLUDED.loaded_at_utc
                """,
                (
                    rid,
                    str(root),
                    str(meta.get("llm_mode") or ""),
                    meta.get("sample_n_jo"),
                    meta.get("sample_m_qna"),
                    json.dumps(meta, ensure_ascii=False),
                    datetime.now(timezone.utc),
                ),
            )

            for n in nodes:
                execute_graph_distiller(
                    cur,
                    f"""
                    INSERT INTO {SCHEMA}.nodes
                        (run_id, node_id, node_type, description, pagerank_score, source_chunk_ids)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, node_id) DO UPDATE SET
                        node_type = EXCLUDED.node_type,
                        description = EXCLUDED.description,
                        pagerank_score = EXCLUDED.pagerank_score,
                        source_chunk_ids = EXCLUDED.source_chunk_ids
                    """,
                    (
                        rid,
                        str(n["id"]),
                        str(n.get("type") or "CONCEPT"),
                        str(n.get("description") or ""),
                        float(n.get("pagerank_score") or 0.0),
                        list(n.get("source_chunk_ids") or []),
                    ),
                )

            for e in edges:
                src, tgt = str(e["source"]), str(e["target"])
                rel = str(e.get("relationship") or "")
                execute_graph_distiller(
                    cur,
                    f"""
                    INSERT INTO {SCHEMA}.edges
                        (run_id, source, target, relationship, weight, provenance)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, source, target, relationship) DO UPDATE SET
                        weight = EXCLUDED.weight,
                        provenance = EXCLUDED.provenance
                    """,
                    (
                        rid,
                        src,
                        tgt,
                        rel,
                        float(e.get("weight") or 1.0),
                        prov.get((src, tgt), "cites"),
                    ),
                )

            chapter_n = 0
            for key, members in (toc or {}).items():
                execute_graph_distiller(
                    cur,
                    f"""
                    INSERT INTO {SCHEMA}.chapters (run_id, chapter_id, member_node_ids)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (run_id, chapter_id) DO UPDATE SET
                        member_node_ids = EXCLUDED.member_node_ids
                    """,
                    (rid, int(key), list(members or [])),
                )
                chapter_n += 1

            sm_n = 0
            for kind, rows in (source_map or {}).items():
                if not isinstance(rows, list):
                    continue
                for row in rows:
                    nid = str(row.get("node_id") or "")
                    if not nid:
                        continue
                    execute_graph_distiller(
                        cur,
                        f"""
                        INSERT INTO {SCHEMA}.source_map (run_id, kind, node_id, payload_json)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (run_id, kind, node_id) DO UPDATE SET
                            payload_json = EXCLUDED.payload_json
                        """,
                        (
                            rid,
                            str(kind),
                            nid,
                            json.dumps(row, ensure_ascii=False),
                        ),
                    )
                    sm_n += 1

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return LoadResult(
        run_id=rid,
        export_dir=str(root),
        nodes=len(nodes),
        edges=len(edges),
        chapters=chapter_n,
        source_map_rows=sm_n,
    )


def count_graph_distiller_rows(uri: str, run_id: str) -> Dict[str, int]:
    """Read-only counts for smoke checks (still SELECT on graph_distiller only)."""
    conn = connect(uri)
    try:
        out: Dict[str, int] = {}
        with conn.cursor() as cur:
            for table in ("nodes", "edges", "chapters", "source_map"):
                cur.execute(
                    f"SELECT COUNT(*) FROM {SCHEMA}.{table} WHERE run_id = %s",
                    (run_id,),
                )
                out[table] = int(cur.fetchone()[0])
        return out
    finally:
        conn.close()
