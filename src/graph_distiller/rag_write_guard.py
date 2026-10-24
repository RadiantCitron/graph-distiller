"""Phase E2: graph_distiller-only WRITE SQL guard (ADR-040).

All mutating SQL for the tent must pass ``assert_graph_distiller_write_sql`` before
execute. READ paths (``rdb.py`` SELECT on public labor_*) must NOT use this
module.

Does not replace DB ROLE grants (E6 docs); this is the in-process belt.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Sequence

# Tent schema — single allowlisted write namespace.
ALLOWED_SCHEMA = "graph_distiller"

# Neighbor / forbidden identifiers in write statements.
_FORBIDDEN_TABLES = ("labor_law", "labor_qna")

_WRITE_HEAD = re.compile(
    r"^\s*(CREATE|DROP|INSERT|UPDATE|DELETE|TRUNCATE|ALTER|GRANT|REVOKE|COPY)\b",
    re.IGNORECASE | re.DOTALL,
)

_SET_SEARCH_PATH = re.compile(
    r"^\s*SET\s+(?:LOCAL\s+)?search_path\s+TO\s+(.+)$",
    re.IGNORECASE | re.DOTALL,
)

# Schema-qualified public writes
_PUBLIC_QUAL = re.compile(r"\bpublic\s*\.", re.IGNORECASE)

# DROP/CREATE SCHEMA … <name>
_SCHEMA_DDL = re.compile(
    r"^\s*(CREATE|DROP)\s+SCHEMA\s+(?:IF\s+(?:NOT\s+)?EXISTS\s+)?([a-zA-Z_][\w$]*)",
    re.IGNORECASE,
)


class GraphDistillerWriteGuardError(ValueError):
    """Raised when SQL would write outside schema ``graph_distiller``."""


def _strip_sql_comments(sql: str) -> str:
    """Remove /* */ and -- line comments (best-effort; strings kept simple)."""
    no_block = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    lines = []
    for line in no_block.splitlines():
        if "--" in line:
            line = line[: line.index("--")]
        lines.append(line)
    return "\n".join(lines)


def split_sql_statements(sql: str) -> list[str]:
    """Split on ``;`` outside single quotes (simple; enough for our DDL)."""
    text = _strip_sql_comments(sql)
    parts: list[str] = []
    buf: list[str] = []
    in_single = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'" and not in_single:
            in_single = True
            buf.append(ch)
        elif ch == "'" and in_single:
            # handle '' escape
            if i + 1 < len(text) and text[i + 1] == "'":
                buf.append("''")
                i += 2
                continue
            in_single = False
            buf.append(ch)
        elif ch == ";" and not in_single:
            stmt = "".join(buf).strip()
            if stmt:
                parts.append(stmt)
            buf = []
        else:
            buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _search_path_ok(rest: str) -> bool:
    """Allow only graph_distiller and pg_catalog in search_path."""
    # strip trailing ;
    rest = rest.strip().rstrip(";").strip()
    names = [p.strip().strip('"') for p in rest.split(",")]
    allowed = {ALLOWED_SCHEMA, "pg_catalog"}
    if not names or names == [""]:
        return False
    return all(n.lower() in allowed for n in names)


def assert_graph_distiller_write_sql(sql: str) -> None:
    """
    Raise ``GraphDistillerWriteGuardError`` if ``sql`` is not an graph_distiller-only write
    (or an allowed ``SET search_path``).

    SELECT / READ statements are rejected here on purpose — use a non-guarded
    cursor for reads (``rdb.py``, ``rag_status.py``).
    """
    if not (sql or "").strip():
        raise GraphDistillerWriteGuardError("empty SQL")

    for stmt in split_sql_statements(sql):
        _assert_one_statement(stmt)


def _assert_one_statement(stmt: str) -> None:
    s = stmt.strip()
    if not s:
        return

    m_set = _SET_SEARCH_PATH.match(s)
    if m_set:
        if not _search_path_ok(m_set.group(1)):
            raise GraphDistillerWriteGuardError(
                f"search_path must be {ALLOWED_SCHEMA}[, pg_catalog] only: {s!r}"
            )
        return

    if not _WRITE_HEAD.match(s):
        raise GraphDistillerWriteGuardError(
            f"only WRITE/DDL (or SET search_path) allowed via write guard: {s[:120]!r}"
        )

    low = s.lower()
    for table in _FORBIDDEN_TABLES:
        if re.search(rf"\b{table}\b", low):
            raise GraphDistillerWriteGuardError(
                f"forbidden table {table!r} in write SQL: {s[:160]!r}"
            )

    if _PUBLIC_QUAL.search(s):
        raise GraphDistillerWriteGuardError(
            f"public.* qualification forbidden in write SQL: {s[:160]!r}"
        )

    m_schema = _SCHEMA_DDL.match(s)
    if m_schema:
        name = m_schema.group(2).lower()
        if name != ALLOWED_SCHEMA:
            raise GraphDistillerWriteGuardError(
                f"schema DDL only allowed for {ALLOWED_SCHEMA!r}, got {name!r}"
            )
        return

    # All other writes must mention graph_distiller. (qualified) or INDEX ON graph_distiller.
    if not re.search(rf"\b{ALLOWED_SCHEMA}\s*\.", s, re.IGNORECASE):
        raise GraphDistillerWriteGuardError(
            f"write SQL must qualify {ALLOWED_SCHEMA}.* : {s[:160]!r}"
        )


def apply_write_search_path(conn) -> None:
    """Restrict session search_path for WRITE connections (E2)."""
    sql = f"SET search_path TO {ALLOWED_SCHEMA}, pg_catalog"
    assert_graph_distiller_write_sql(sql)
    with conn.cursor() as cur:
        cur.execute(sql)


def execute_graph_distiller(
    cur,
    sql: str,
    params: Optional[Sequence[Any] | dict[str, Any]] = None,
) -> Any:
    """assert + cursor.execute for tent writes."""
    assert_graph_distiller_write_sql(sql)
    if params is None:
        return cur.execute(sql)
    return cur.execute(sql, params)
