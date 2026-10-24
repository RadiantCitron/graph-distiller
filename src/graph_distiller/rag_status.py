"""Phase E1: leave-no-trace STATUS report (ADR-040) — read-only.

Reports public COUNT (neighbor), graph_distiller inventory (tent), and local export
completeness. Never WRITE / DDL. Never log DB URIs or passwords.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from graph_distiller.ingestion.rdb import connect, counts_as_dict, fetch_counts
from graph_distiller.rag_load import SCHEMA

logger = logging.getLogger(__name__)

# Same set as rag_load._assert_export_dir — keep in sync until E6 consolidates.
EXPORT_REQUIRED_FILES: Sequence[str] = (
    "nodes.json",
    "edges.json",
    "toc.json",
    "meta.json",
    "source_map.json",
)

GRAPH_DISTILLER_TABLES_EXPECTED: Sequence[str] = (
    "runs",
    "nodes",
    "edges",
    "chapters",
    "source_map",
)


@dataclass(frozen=True)
class ExportDirStatus:
    run_id: str
    path: str
    status: str  # "complete" | "incomplete"
    missing: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphDistillerRunStatus:
    run_id: str
    export_dir: str
    llm_mode: str
    sample_n_jo: Optional[int]
    sample_m_qna: Optional[int]
    loaded_at_utc: Optional[str]
    nodes: int
    edges: int
    chapters: int
    source_map: int


@dataclass(frozen=True)
class GraphDistillerInventory:
    schema_exists: bool
    tables: List[str]
    missing_tables: List[str]
    runs: List[GraphDistillerRunStatus]


@dataclass(frozen=True)
class StatusReport:
    """Serializable STATUS payload (no secrets)."""

    generated_at_utc: str
    public: Optional[Dict[str, Any]]
    graph_distiller: Optional[GraphDistillerInventory]
    exports: List[ExportDirStatus]
    notes: List[str]
    db_queried: bool


def scan_export_dir(path: Path) -> ExportDirStatus:
    """Classify one rag_rebuild/<run_id>/ folder (filesystem only)."""
    root = Path(path)
    missing = [name for name in EXPORT_REQUIRED_FILES if not (root / name).is_file()]
    return ExportDirStatus(
        run_id=root.name,
        path=str(root.resolve()) if root.exists() else str(root),
        status="complete" if not missing else "incomplete",
        missing=list(missing),
    )


def scan_rebuild_root(rebuild_dir: Path | str) -> List[ExportDirStatus]:
    """Scan ``output/rag_rebuild/*`` children that look like export dirs."""
    root = Path(rebuild_dir)
    if not root.is_dir():
        return []
    out: List[ExportDirStatus] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        out.append(scan_export_dir(child))
    return out


def _iso(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)


def fetch_graph_distiller_inventory(conn) -> GraphDistillerInventory:
    """SELECT-only inventory of schema ``graph_distiller``. No writes."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS("
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s"
            ")",
            (SCHEMA,),
        )
        schema_exists = bool(cur.fetchone()[0])
        if not schema_exists:
            return GraphDistillerInventory(
                schema_exists=False,
                tables=[],
                missing_tables=list(GRAPH_DISTILLER_TABLES_EXPECTED),
                runs=[],
            )

        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s ORDER BY table_name",
            (SCHEMA,),
        )
        tables = [str(r[0]) for r in cur.fetchall()]
        missing = [t for t in GRAPH_DISTILLER_TABLES_EXPECTED if t not in tables]

        if "runs" not in tables:
            return GraphDistillerInventory(
                schema_exists=True,
                tables=tables,
                missing_tables=missing,
                runs=[],
            )

        cur.execute(
            f"""
            SELECT run_id, export_dir, llm_mode, sample_n_jo, sample_m_qna, loaded_at_utc
            FROM {SCHEMA}.runs
            ORDER BY loaded_at_utc DESC NULLS LAST, run_id
            """
        )
        rows = cur.fetchall()
        runs: List[GraphDistillerRunStatus] = []
        for row in rows:
            rid = str(row[0])
            counts = {"nodes": 0, "edges": 0, "chapters": 0, "source_map": 0}
            for table in ("nodes", "edges", "chapters", "source_map"):
                if table not in tables:
                    continue
                cur.execute(
                    f"SELECT COUNT(*) FROM {SCHEMA}.{table} WHERE run_id = %s",
                    (rid,),
                )
                counts[table] = int(cur.fetchone()[0])
            runs.append(
                GraphDistillerRunStatus(
                    run_id=rid,
                    export_dir=str(row[1] or ""),
                    llm_mode=str(row[2] or ""),
                    sample_n_jo=int(row[3]) if row[3] is not None else None,
                    sample_m_qna=int(row[4]) if row[4] is not None else None,
                    loaded_at_utc=_iso(row[5]),
                    nodes=counts["nodes"],
                    edges=counts["edges"],
                    chapters=counts["chapters"],
                    source_map=counts["source_map"],
                )
            )

    return GraphDistillerInventory(
        schema_exists=True,
        tables=tables,
        missing_tables=missing,
        runs=runs,
    )


def collect_status(
    *,
    uri: str | None,
    rebuild_dir: Path | str,
    skip_db: bool = False,
) -> StatusReport:
    """
    Build a full STATUS report.

    ``skip_db=True`` or empty ``uri`` → exports only (notes explain why).
    """
    notes: List[str] = [
        "Read-only STATUS (Phase E1). No WRITE/DDL.",
        f"Tent schema name: {SCHEMA}",
        "Neighbor tables (public labor_*) are COUNTed only when DB is queried.",
    ]
    exports = scan_rebuild_root(rebuild_dir)
    incomplete = [e.run_id for e in exports if e.status == "incomplete"]
    if incomplete:
        notes.append(f"Incomplete local exports: {incomplete}")

    public: Optional[Dict[str, Any]] = None
    graph_distiller: Optional[GraphDistillerInventory] = None
    db_queried = False
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if skip_db or not (uri or "").strip():
        notes.append("DB not queried (no URI or --offline).")
        return StatusReport(
            generated_at_utc=now,
            public=None,
            graph_distiller=None,
            exports=exports,
            notes=notes,
            db_queried=False,
        )

    conn = connect(str(uri).strip())
    try:
        public = counts_as_dict(fetch_counts(conn))
        graph_distiller = fetch_graph_distiller_inventory(conn)
        db_queried = True
        if not graph_distiller.schema_exists:
            notes.append("graph_distiller schema does not exist (no tent pitched).")
        elif graph_distiller.missing_tables:
            notes.append(f"graph_distiller missing tables: {graph_distiller.missing_tables}")
        elif not graph_distiller.runs:
            notes.append("graph_distiller schema exists but has no runs.")
    finally:
        conn.close()

    return StatusReport(
        generated_at_utc=now,
        public=public,
        graph_distiller=graph_distiller,
        exports=exports,
        notes=notes,
        db_queried=db_queried,
    )


def status_to_dict(report: StatusReport) -> Dict[str, Any]:
    """JSON-serializable dict (no URI)."""
    payload: Dict[str, Any] = {
        "generated_at_utc": report.generated_at_utc,
        "db_queried": report.db_queried,
        "public": report.public,
        "exports": [asdict(e) for e in report.exports],
        "notes": list(report.notes),
    }
    if report.graph_distiller is None:
        payload["graph_distiller"] = None
    else:
        payload["graph_distiller"] = {
            "schema_exists": report.graph_distiller.schema_exists,
            "tables": list(report.graph_distiller.tables),
            "missing_tables": list(report.graph_distiller.missing_tables),
            "runs": [asdict(r) for r in report.graph_distiller.runs],
        }
    return payload


def format_status_text(report: StatusReport) -> str:
    """Human-readable STATUS."""
    lines: List[str] = [
        f"[STATUS] generated={report.generated_at_utc} db_queried={report.db_queried}",
        "",
        "== public (neighbor, COUNT only) ==",
    ]
    if report.public is None:
        lines.append("  (skipped)")
    else:
        for k, v in report.public.items():
            lines.append(f"  {k}: {v}")

    lines.extend(["", "== graph_distiller (tent) =="])
    if report.graph_distiller is None:
        lines.append("  (skipped)")
    else:
        o = report.graph_distiller
        lines.append(f"  schema_exists: {o.schema_exists}")
        lines.append(f"  tables: {o.tables or '(none)'}")
        if o.missing_tables:
            lines.append(f"  missing_tables: {o.missing_tables}")
        if not o.runs:
            lines.append("  runs: (none)")
        else:
            for r in o.runs:
                lines.append(
                    f"  run {r.run_id}: nodes={r.nodes} edges={r.edges} "
                    f"chapters={r.chapters} source_map={r.source_map} "
                    f"loaded_at={r.loaded_at_utc}"
                )

    lines.extend(["", "== local exports =="])
    if not report.exports:
        lines.append("  (none)")
    else:
        for e in report.exports:
            miss = f" missing={e.missing}" if e.missing else ""
            lines.append(f"  {e.run_id}: {e.status}{miss}")

    lines.extend(["", "== notes =="])
    for n in report.notes:
        lines.append(f"  - {n}")
    return "\n".join(lines)
