"""Read-only pull from ragdb (ADR-033/037). SELECT only — never WRITE public.*.

Sample gate: caller MUST pass positive integers n_jo / m_qna.
Full-table LLM is a failure mode; this module never defaults to “all rows”.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from graph_distiller.ingestion.law_cites import (
    JoRef,
    cite_pairs,
    count_selected_jo_hits,
    primary_jo_ref_from_article_text,
    qna_node_id,
)
from graph_distiller.schema import ChunkMeta, GraphEdge, GraphNode

logger = logging.getLogger(__name__)

REL_CITES = "CITES"
REL_GROUNDS = "GROUNDS"


@dataclass(frozen=True)
class LawRow:
    id: int
    parent_id: Optional[int]
    level: str
    content: str


@dataclass(frozen=True)
class RagCounts:
    labor_law_rows: int
    labor_law_jo: int
    labor_law_jang: int
    labor_qna_rows: int
    labor_qna_cases: int
    labor_qna_jang: int


@dataclass
class JoBundle:
    """One 「조」 plus merged descendant text (항/호/본문)."""

    jo_row_id: int
    jo_ref: JoRef
    jang_id: Optional[int]
    jang_title: str
    content_merged: str
    descendant_ids: List[int] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return self.jo_ref.node_id

    @property
    def source_doc_id(self) -> str:
        return f"labor_law_jo_{self.jo_row_id}"

    @property
    def chunk_id(self) -> str:
        return f"{self.source_doc_id}_merged"


@dataclass
class QnaCase:
    case_id: int
    parent_id: Optional[int]
    content: str
    cite_hits: int = 0

    @property
    def node_id(self) -> str:
        return qna_node_id(self.case_id)

    @property
    def source_doc_id(self) -> str:
        return f"labor_qna_case_{self.case_id}"

    @property
    def chunk_id(self) -> str:
        return f"{self.source_doc_id}_body"


@dataclass
class EdgeMeta:
    source: str
    target: str
    provenance: str  # "cites" | "llm" | "tree"
    relationship: str
    chunk_id: str = ""
    note: str = ""


@dataclass
class SamplePayload:
    counts: RagCounts
    sample_n_jo: int
    sample_m_qna: int
    jo_bundles: List[JoBundle]
    qna_cases: List[QnaCase]
    selection_notes: List[str] = field(default_factory=list)


def require_sample_sizes(n_jo: int, m_qna: int) -> Tuple[int, int]:
    """ADR-037: no silent full-DB. Both must be positive integers."""
    if int(n_jo) <= 0 or int(m_qna) < 0:
        raise ValueError(
            "Sample sizes required: --n-jo >= 1 and --m-qna >= 0 "
            "(full-DB default is forbidden)."
        )
    return int(n_jo), int(m_qna)


def connect(uri: str):
    """Open a psycopg connection. Caller closes. Never log the URI."""
    if not (uri or "").strip():
        raise ValueError("RAG_DB_URI is empty — set it in .env (never commit).")
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "psycopg is required for RAG pull. pip install 'psycopg[binary]'"
        ) from exc
    return psycopg.connect(uri.strip(), connect_timeout=10)


def fetch_counts(conn) -> RagCounts:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM labor_law")
        law_rows = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM labor_law WHERE level = %s", ("조",))
        law_jo = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM labor_law WHERE level = %s", ("장",))
        law_jang = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM labor_qna")
        qna_rows = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM labor_qna WHERE level = %s", ("케이스",))
        qna_cases = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM labor_qna WHERE level = %s", ("장",))
        qna_jang = int(cur.fetchone()[0])
    return RagCounts(
        labor_law_rows=law_rows,
        labor_law_jo=law_jo,
        labor_law_jang=law_jang,
        labor_qna_rows=qna_rows,
        labor_qna_cases=qna_cases,
        labor_qna_jang=qna_jang,
    )


def _fetch_all_law(conn) -> List[LawRow]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, parent_id, level, content FROM labor_law ORDER BY id"
        )
        return [
            LawRow(
                id=int(r[0]),
                parent_id=int(r[1]) if r[1] is not None else None,
                level=str(r[2] or ""),
                content=str(r[3] or ""),
            )
            for r in cur.fetchall()
        ]


def _fetch_all_qna_cases(conn) -> List[QnaCase]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, parent_id, content FROM labor_qna "
            "WHERE level = %s ORDER BY id",
            ("케이스",),
        )
        return [
            QnaCase(
                case_id=int(r[0]),
                parent_id=int(r[1]) if r[1] is not None else None,
                content=str(r[2] or ""),
            )
            for r in cur.fetchall()
        ]


def children_index(rows: Sequence[LawRow]) -> Dict[Optional[int], List[LawRow]]:
    by_parent: Dict[Optional[int], List[LawRow]] = defaultdict(list)
    for row in rows:
        by_parent[row.parent_id].append(row)
    for key in by_parent:
        by_parent[key].sort(key=lambda r: r.id)
    return by_parent


def collect_descendants(
    root_id: int,
    by_parent: Mapping[Optional[int], Sequence[LawRow]],
) -> List[LawRow]:
    """DFS descendants (not including root), stable id order of discovery."""
    out: List[LawRow] = []
    stack = list(by_parent.get(root_id, ()))
    while stack:
        node = stack.pop(0)
        out.append(node)
        stack[0:0] = list(by_parent.get(node.id, ()))
    return out


def merge_jo_bundle(
    jo: LawRow,
    by_parent: Mapping[Optional[int], Sequence[LawRow]],
    *,
    by_id: Mapping[int, LawRow],
    jang_by_id: Mapping[int, LawRow],
) -> JoBundle | None:
    """Merge 조 + 항/호/본문 into one text block. Skip if 조 number unparseable."""
    ref = primary_jo_ref_from_article_text(jo.content)
    if ref is None:
        logger.warning("Skip 조 id=%s: no 제N조 in content", jo.id)
        return None
    descendants = collect_descendants(jo.id, by_parent)
    parts = [jo.content.strip()]
    for d in descendants:
        body = (d.content or "").strip()
        if body:
            parts.append(f"[{d.level}] {body}")
    jang_id = _find_jang_ancestor(jo, by_id)
    jang_title = ""
    if jang_id is not None and jang_id in jang_by_id:
        jang_title = (jang_by_id[jang_id].content or "").strip()
    return JoBundle(
        jo_row_id=jo.id,
        jo_ref=ref,
        jang_id=jang_id,
        jang_title=jang_title,
        content_merged="\n\n".join(parts),
        descendant_ids=[d.id for d in descendants],
    )


def _find_jang_ancestor(row: LawRow, by_id: Mapping[int, LawRow]) -> Optional[int]:
    cur: Optional[LawRow] = row
    guard = 0
    while cur is not None and guard < 64:
        if cur.level == "장":
            return cur.id
        if cur.parent_id is None:
            return None
        cur = by_id.get(cur.parent_id)
        guard += 1
    return None


def select_jo_rows_chapter_first(
    rows: Sequence[LawRow],
    n_jo: int,
) -> Tuple[List[LawRow], List[str]]:
    """
    Prefer whole chapters: take 조 under the largest 장 first, then next, …
    Never random scatter (ADR-037).
    """
    notes: List[str] = []
    by_id = {r.id: r for r in rows}
    by_parent = children_index(rows)
    jang_rows = [r for r in rows if r.level == "장"]
    jo_rows = [r for r in rows if r.level == "조"]

    def jo_under_jang(jang_id: int) -> List[LawRow]:
        # 조 may be direct child of 장, or nested — walk all 조 and check ancestor
        out: List[LawRow] = []
        for jo in jo_rows:
            if _find_jang_ancestor(jo, by_id) == jang_id:
                out.append(jo)
        out.sort(key=lambda r: r.id)
        return out

    ranked_jang = sorted(
        jang_rows,
        key=lambda j: (-len(jo_under_jang(j.id)), j.id),
    )
    selected: List[LawRow] = []
    selected_ids: set[int] = set()
    # Phase C: when sampling enough 조, keep ≥2 장 if the DB has them (metric A).
    reserve_second_jang = 1 if (n_jo >= 4 and len(ranked_jang) >= 2) else 0
    primary_budget = max(1, n_jo - reserve_second_jang)

    for jang in ranked_jang[:1]:
        bundle = jo_under_jang(jang.id)
        notes.append(
            f"jang id={jang.id} candidates={len(bundle)} "
            f"title={((jang.content or '')[:40])!r}"
        )
        for jo in bundle:
            if len(selected) >= primary_budget:
                break
            if jo.id in selected_ids:
                continue
            selected.append(jo)
            selected_ids.add(jo.id)

    for jang in ranked_jang[1:]:
        if len(selected) >= n_jo:
            break
        bundle = jo_under_jang(jang.id)
        notes.append(
            f"jang id={jang.id} candidates={len(bundle)} "
            f"title={((jang.content or '')[:40])!r}"
        )
        for jo in bundle:
            if len(selected) >= n_jo:
                break
            if jo.id in selected_ids:
                continue
            selected.append(jo)
            selected_ids.add(jo.id)

    if len(selected) < n_jo:
        for jo in sorted(jo_rows, key=lambda r: r.id):
            if len(selected) >= n_jo:
                break
            if jo.id in selected_ids:
                continue
            selected.append(jo)
            selected_ids.add(jo.id)
        notes.append("filled remaining 조 by ascending id (cross-chapter)")

    notes.append(f"selected_jo_count={len(selected)} requested={n_jo}")
    return selected, notes


def select_qna_cases_cite_first(
    cases: Sequence[QnaCase],
    selected_refs: Sequence[JoRef],
    m_qna: int,
) -> Tuple[List[QnaCase], List[str]]:
    """Rank cases by how many selected 조 they cite; fill by id if short."""
    notes: List[str] = []
    if m_qna == 0:
        notes.append("m_qna=0 — no QnA cases")
        return [], notes

    scored: List[QnaCase] = []
    for case in cases:
        hits = count_selected_jo_hits(case.content, selected_refs)
        scored.append(
            QnaCase(
                case_id=case.case_id,
                parent_id=case.parent_id,
                content=case.content,
                cite_hits=hits,
            )
        )
    scored.sort(key=lambda c: (-c.cite_hits, c.case_id))
    with_hits = [c for c in scored if c.cite_hits > 0]
    without = [c for c in scored if c.cite_hits == 0]
    chosen = with_hits[:m_qna]
    if len(chosen) < m_qna:
        need = m_qna - len(chosen)
        chosen.extend(without[:need])
        notes.append(
            f"cite_hits cases={len(with_hits)}; filled {need} without hits by id"
        )
    else:
        notes.append(f"all {len(chosen)} cases have cite_hits>0 among selected 조")
    return chosen, notes


def build_sample_from_rows(
    law_rows: Sequence[LawRow],
    qna_cases: Sequence[QnaCase],
    counts: RagCounts,
    n_jo: int,
    m_qna: int,
) -> SamplePayload:
    """Pure sample builder (unit-testable without Postgres)."""
    n_jo, m_qna = require_sample_sizes(n_jo, m_qna)
    if n_jo > counts.labor_law_jo:
        raise ValueError(
            f"--n-jo {n_jo} > available 조 count {counts.labor_law_jo}"
        )
    if m_qna > counts.labor_qna_cases:
        raise ValueError(
            f"--m-qna {m_qna} > available 케이스 count {counts.labor_qna_cases}"
        )

    by_parent = children_index(law_rows)
    by_id = {r.id: r for r in law_rows}
    jang_by_id = {r.id: r for r in law_rows if r.level == "장"}
    selected_jo, notes = select_jo_rows_chapter_first(law_rows, n_jo)

    bundles: List[JoBundle] = []
    for jo in selected_jo:
        bundle = merge_jo_bundle(
            jo, by_parent, by_id=by_id, jang_by_id=jang_by_id
        )
        if bundle is not None:
            bundles.append(bundle)
    if not bundles:
        raise RuntimeError("No 조 bundles could be parsed (제N조 missing?)")

    refs = [b.jo_ref for b in bundles]
    cases, qnotes = select_qna_cases_cite_first(qna_cases, refs, m_qna)
    notes.extend(qnotes)
    return SamplePayload(
        counts=counts,
        sample_n_jo=n_jo,
        sample_m_qna=m_qna,
        jo_bundles=bundles,
        qna_cases=cases,
        selection_notes=notes,
    )


def load_sample(conn, n_jo: int, m_qna: int) -> SamplePayload:
    """COUNT-aware sample pull (SELECT only)."""
    counts = fetch_counts(conn)
    law_rows = _fetch_all_law(conn)
    qna_cases = _fetch_all_qna_cases(conn)
    return build_sample_from_rows(law_rows, qna_cases, counts, n_jo, m_qna)


def load_full(conn) -> SamplePayload:
    """
    Explicit full-corpus pull (SELECT only). Never a silent default.

    Caller must have passed --full-db (ADR-037 / ADR-053). Uses every 조
    bundle and every QnA 케이스. Phase 2 LLM remains opt-in + budget-sliced.
    """
    counts = fetch_counts(conn)
    if counts.labor_law_jo < 1:
        raise ValueError("load_full: labor_law has no 조 rows")
    law_rows = _fetch_all_law(conn)
    qna_cases = _fetch_all_qna_cases(conn)
    return build_sample_from_rows(
        law_rows,
        qna_cases,
        counts,
        counts.labor_law_jo,
        counts.labor_qna_cases,
    )


def sample_to_chunks(sample: SamplePayload) -> List[ChunkMeta]:
    """Fabricate ChunkMeta for Phase 2 / RepoMap (no SemanticChunker)."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    chunks: List[ChunkMeta] = []
    for b in sample.jo_bundles:
        chunks.append(
            ChunkMeta(
                chunk_id=b.chunk_id,
                source_doc_id=b.source_doc_id,
                content=b.content_merged,
                token_count=len(enc.encode(b.content_merged)),
            )
        )
    for c in sample.qna_cases:
        chunks.append(
            ChunkMeta(
                chunk_id=c.chunk_id,
                source_doc_id=c.source_doc_id,
                content=c.content,
                token_count=len(enc.encode(c.content)),
            )
        )
    return chunks


def build_cites_graph(
    sample: SamplePayload,
) -> Tuple[List[GraphNode], List[GraphEdge], List[EdgeMeta]]:
    """
    Regex-only graph: law nodes for sampled 조 + QnA nodes + CITES/GROUNDS edges.
    Also adds law→law CITES when a 조 body cites another sampled 조.
    """
    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    meta: List[EdgeMeta] = []

    def ensure_law(ref: JoRef, chunk_id: str, description: str) -> None:
        nid = ref.node_id
        if nid in nodes:
            prev = nodes[nid]
            ids = list(dict.fromkeys(prev.source_chunk_ids + [chunk_id]))
            nodes[nid] = prev.model_copy(update={"source_chunk_ids": ids})
            return
        nodes[nid] = GraphNode(
            id=nid,
            type="CONCEPT",
            description=description[:500] or ref.label_ko(),
            source_chunk_ids=[chunk_id],
        )

    for b in sample.jo_bundles:
        ensure_law(
            b.jo_ref,
            b.chunk_id,
            f"{b.jo_ref.label_ko()} | {b.jang_title}".strip(" |"),
        )
        for src, tgt, ref in cite_pairs(
            b.content_merged, source_node_id=b.node_id, exclude_self=True
        ):
            # Stub law nodes for cited-but-unsampled 조 keep B (GT) measurable.
            ensure_law(ref, b.chunk_id, ref.label_ko())
            edges.append(
                GraphEdge(
                    source=src,
                    target=tgt,
                    relationship=REL_CITES,
                    weight=1.0,
                    provenance="cites",
                )
            )
            meta.append(
                EdgeMeta(
                    source=src,
                    target=tgt,
                    provenance="cites",
                    relationship=REL_CITES,
                    chunk_id=b.chunk_id,
                    note=ref.label_ko(),
                )
            )

    for c in sample.qna_cases:
        nodes[c.node_id] = GraphNode(
            id=c.node_id,
            type="CONCEPT",
            description=f"labor_qna case {c.case_id} (cite_hits={c.cite_hits})",
            source_chunk_ids=[c.chunk_id],
        )
        for src, tgt, ref in cite_pairs(
            c.content, source_node_id=c.node_id, exclude_self=True
        ):
            ensure_law(ref, c.chunk_id, ref.label_ko())
            edges.append(
                GraphEdge(
                    source=src,
                    target=tgt,
                    relationship=REL_GROUNDS,
                    weight=1.0 + float(c.cite_hits) * 0.1,
                    provenance="cites",
                )
            )
            meta.append(
                EdgeMeta(
                    source=src,
                    target=tgt,
                    provenance="cites",
                    relationship=REL_GROUNDS,
                    chunk_id=c.chunk_id,
                    note=ref.label_ko(),
                )
            )

    return list(nodes.values()), edges, meta


def counts_as_dict(counts: RagCounts) -> Dict[str, Any]:
    return {
        "labor_law_rows": counts.labor_law_rows,
        "labor_law_jo": counts.labor_law_jo,
        "labor_law_jang": counts.labor_law_jang,
        "labor_qna_rows": counts.labor_qna_rows,
        "labor_qna_cases": counts.labor_qna_cases,
        "labor_qna_jang": counts.labor_qna_jang,
    }
