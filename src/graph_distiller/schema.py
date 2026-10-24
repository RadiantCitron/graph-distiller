"""Pydantic models and GraphDistillerState — SPEC Level 3 §1 + Level 3.5 §4."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# --- Sub-schemas ---


class ChunkMeta(BaseModel):
    chunk_id: str
    source_doc_id: str
    content: str
    token_count: int


class GraphNode(BaseModel):
    id: str = Field(description="엔티티의 고유 이름 (대문자 정규화)")
    type: Literal["CONCEPT", "PERSON", "SYSTEM"] = Field(
        description="CONCEPT, PERSON, SYSTEM 중 택 1"
    )
    description: str = Field(description="이 엔티티에 대한 1~2줄 요약")
    source_chunk_ids: List[str] = Field(
        description="이 엔티티가 발견된 청크 ID 목록"
    )
    pagerank_score: float = Field(
        default=0.0,
        description="Phase 3 PageRank 점수 (nx.pagerank)",
    )


class GraphEdge(BaseModel):
    source: str
    target: str
    relationship: str = Field(description="두 엔티티 간의 관계 설명")
    weight: float = Field(default=1.0)
    # W2-A/B: edge trust channel (ADR-042). Default llm for Phase 2 extract.
    provenance: str = Field(
        default="llm",
        description="cites | llm | alias-merge | unknown",
    )


class ChapterDraft(BaseModel):
    chapter_id: int
    title: str
    content: str
    cited_doc_ids: List[str]


class ExtractionResult(BaseModel):
    """Phase 2 structured LLM output: GraphNode + GraphEdge lists."""

    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


class ChapterTitleResult(BaseModel):
    """Planner LLM output: human-readable chapter title (<= 5 words)."""

    title: str = Field(description="사람이 읽기 좋은 챕터 제목 (5단어 이내)")


# --- Main State (LangGraph TypedDict) ---


class Phase3Stats(TypedDict):
    """Phase 3 counters for CLI (ADR-024). Not used by Phase 4 nodes."""

    extracted_nodes: int
    nodes_after_orphan: int
    edges_after_orphan: int
    orphans_dropped: int
    communities_detected: int
    communities_kept: int
    thin_dropped: int
    poc_dropped: int
    poc_top_chapters: int
    algorithm: str  # "leiden" | "louvain" | ""


def empty_phase3_stats() -> Phase3Stats:
    return {
        "extracted_nodes": 0,
        "nodes_after_orphan": 0,
        "edges_after_orphan": 0,
        "orphans_dropped": 0,
        "communities_detected": 0,
        "communities_kept": 0,
        "thin_dropped": 0,
        "poc_dropped": 0,
        "poc_top_chapters": 0,
        "algorithm": "",
    }


class GraphDistillerState(TypedDict):
    raw_documents: List[Dict[str, Any]]
    chunks: List[ChunkMeta]
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    toc: Dict[int, List[str]]
    repo_map: str
    drafts: List[ChapterDraft]
    errors: List[str]
    retry_counts: Dict[int, int]
    # Phase 4 loop control
    current_chapter_id: Optional[int]
    chapter_titles: Dict[int, str]
    poisoned_chapters: List[int]
    last_verification: str  # "pass" | "fail" | ""
    chapter_fail_reasons: Dict[int, str]  # ADR-057 조치2: 최근 Verifier FAIL 사유
    phase3_stats: Phase3Stats


def empty_state() -> GraphDistillerState:
    return {
        "raw_documents": [],
        "chunks": [],
        "nodes": [],
        "edges": [],
        "toc": {},
        "repo_map": "",
        "drafts": [],
        "errors": [],
        "retry_counts": {},
        "current_chapter_id": None,
        "chapter_titles": {},
        "poisoned_chapters": [],
        "last_verification": "",
        "chapter_fail_reasons": {},
        "phase3_stats": empty_phase3_stats(),
    }
