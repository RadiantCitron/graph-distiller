"""Phase E3: leave-no-trace Cleanup — tear down tent ``graph_distiller`` only (ADR-040).

Hard rules:
- DROP/DELETE targets are fixed to schema ``graph_distiller`` (never CLI-chosen schema names).
- All SQL goes through ``rag_write_guard``.
- Never WRITE ``public.*`` / labor_*.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from graph_distiller.ingestion.rdb import connect
from graph_distiller.rag_load import SCHEMA
from graph_distiller.rag_write_guard import apply_write_search_path, execute_graph_distiller

logger = logging.getLogger(__name__)

# Fixed SQL — schema name is not parameterized from user input.
DROP_GRAPH_DISTILLER_SCHEMA_SQL = f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"
DELETE_GRAPH_DISTILLER_RUN_SQL = f"DELETE FROM {SCHEMA}.runs WHERE run_id = %s"


@dataclass(frozen=True)
class CleanupResult:
    action: str  # "drop_schema" | "delete_run"
    schema: str
    run_id: Optional[str]
    export_dirs_removed: List[str]
    ok: bool


def drop_graph_distiller_schema(uri: str) -> CleanupResult:
    """``DROP SCHEMA IF EXISTS graph_distiller CASCADE``. Neighbor schemas untouched."""
    conn = connect(uri)
    try:
        apply_write_search_path(conn)
        with conn.cursor() as cur:
            execute_graph_distiller(cur, DROP_GRAPH_DISTILLER_SCHEMA_SQL)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    logger.info("Dropped schema %s CASCADE", SCHEMA)
    return CleanupResult(
        action="drop_schema",
        schema=SCHEMA,
        run_id=None,
        export_dirs_removed=[],
        ok=True,
    )


def delete_graph_distiller_run(uri: str, run_id: str) -> CleanupResult:
    """Delete one run (CASCADE to nodes/edges/…). Schema remains."""
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id required")
    conn = connect(uri)
    try:
        apply_write_search_path(conn)
        with conn.cursor() as cur:
            execute_graph_distiller(cur, DELETE_GRAPH_DISTILLER_RUN_SQL, (rid,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    logger.info("Deleted graph_distiller run_id=%s", rid)
    return CleanupResult(
        action="delete_run",
        schema=SCHEMA,
        run_id=rid,
        export_dirs_removed=[],
        ok=True,
    )


def list_graph_distiller_export_dirs(uri: str) -> List[str]:
    """Read-only: export_dir values from graph_distiller.runs (empty if no schema)."""
    conn = connect(uri)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS("
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s"
                ")",
                (SCHEMA,),
            )
            if not cur.fetchone()[0]:
                return []
            cur.execute(
                f"SELECT export_dir FROM {SCHEMA}.runs WHERE export_dir IS NOT NULL"
            )
            return [str(r[0]) for r in cur.fetchall() if r[0]]
    finally:
        conn.close()


def safe_remove_export_dir(
    path: Path | str,
    *,
    rebuild_root: Path | str,
) -> Optional[str]:
    """
    Delete a local export folder only if it resolves under ``rebuild_root``.
    Returns removed path string, or None if skipped.
    """
    root = Path(rebuild_root).resolve()
    target = Path(path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        logger.warning(
            "Skip export delete (not under rebuild_root): %s", target
        )
        return None
    if not target.is_dir():
        return None
    shutil.rmtree(target)
    logger.info("Removed export dir %s", target)
    return str(target)


def cleanup_with_optional_exports(
    uri: str,
    *,
    drop_schema: bool = False,
    run_id: str | None = None,
    also_export_dirs: bool = False,
    rebuild_root: Path | str | None = None,
    export_dir: Path | str | None = None,
) -> CleanupResult:
    """
    High-level Cleanup.

    Exactly one of ``drop_schema`` or ``run_id`` must be set.
    ``also_export_dirs`` deletes local folders under ``rebuild_root`` only.
    """
    if drop_schema and run_id:
        raise ValueError("choose either drop_schema or run_id, not both")
    if not drop_schema and not run_id:
        raise ValueError("drop_schema or run_id required")

    removed: List[str] = []
    if also_export_dirs:
        if rebuild_root is None:
            raise ValueError("rebuild_root required when also_export_dirs=True")
        paths: List[str] = []
        if export_dir is not None:
            paths.append(str(export_dir))
        elif run_id:
            paths.append(str(Path(rebuild_root) / run_id))
        elif drop_schema:
            # Snapshot paths before DROP
            paths.extend(list_graph_distiller_export_dirs(uri))
        for p in paths:
            gone = safe_remove_export_dir(p, rebuild_root=rebuild_root)
            if gone:
                removed.append(gone)

    if drop_schema:
        result = drop_graph_distiller_schema(uri)
    else:
        result = delete_graph_distiller_run(uri, str(run_id))

    return CleanupResult(
        action=result.action,
        schema=result.schema,
        run_id=result.run_id,
        export_dirs_removed=removed,
        ok=True,
    )
