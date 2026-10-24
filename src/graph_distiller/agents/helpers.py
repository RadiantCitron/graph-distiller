"""Shared helpers for Phase 4 agents."""

from __future__ import annotations

from typing import List, Set

from graph_distiller.schema import ChapterDraft, GraphDistillerState


def drafted_chapter_ids(state: GraphDistillerState) -> Set[int]:
    return {d.chapter_id for d in state.get("drafts") or []}


def poisoned_chapter_ids(state: GraphDistillerState) -> Set[int]:
    return set(state.get("poisoned_chapters") or [])


def pending_chapter_ids(state: GraphDistillerState) -> List[int]:
    """TOC chapters not yet successfully drafted and not poisoned."""
    done = drafted_chapter_ids(state)
    poisoned = poisoned_chapter_ids(state)
    toc = state.get("toc") or {}
    return sorted(cid for cid in toc.keys() if cid not in done and cid not in poisoned)


def upsert_draft(state: GraphDistillerState, draft: ChapterDraft) -> List[ChapterDraft]:
    drafts = [d for d in (state.get("drafts") or []) if d.chapter_id != draft.chapter_id]
    drafts.append(draft)
    drafts.sort(key=lambda d: d.chapter_id)
    return drafts


def drop_draft(state: GraphDistillerState, chapter_id: int) -> List[ChapterDraft]:
    return [d for d in (state.get("drafts") or []) if d.chapter_id != chapter_id]


def append_error(state: GraphDistillerState, message: str) -> List[str]:
    errors = list(state.get("errors") or [])
    errors.append(message)
    return errors
