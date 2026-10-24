"""Planner Agent — pick next chapter + generate human-readable title."""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from graph_distiller.adapters.openrouter import default_llm_chat
from graph_distiller.agents.helpers import pending_chapter_ids
from graph_distiller.config import get_settings
from graph_distiller.prompts import load_prompt
from graph_distiller.rules import get_rules
from graph_distiller.schema import ChapterTitleResult, GraphNode, GraphDistillerState

logger = logging.getLogger(__name__)

TITLE_SYSTEM = """\
You name book chapters for a knowledge codex.
Return a concise human-readable chapter title in at most 5 words.
Prefer Korean if the entity names are Korean-domain; otherwise English is fine.
Do not include chapter numbers or quotes.
Do not output schema names, JSON, or field names such as description.
"""


# Fallback only — the live list is data/rules/phase4_planner.toml::leak_markers
# (ADR-060). Kept here so a missing/corrupt rules dir still catches schema leaks.
_LEAK_MARKERS = (
    "chaptertitleresult",
    "basemodel",
    "field(description",
)


def _fallback_title(entity_names: list[str], chapter_id: int) -> str:
    if entity_names:
        return " / ".join(entity_names[:3]).replace("_", " ").title()
    return f"Chapter {chapter_id + 1}"


def looks_like_schema_leak(title: str, *, leak_markers: tuple[str, ...] = _LEAK_MARKERS) -> bool:
    """True when the string is a dumped Pydantic/JSON object, not a chapter title."""
    text = (title or "").strip()
    if not text:
        return True
    compact = re.sub(r"[\s_'\"]+", "", text).lower()
    if any(marker in compact for marker in leak_markers):
        return True
    if text[0] in "{[":
        return True
    if "description':" in text or '"description"' in text:
        return True
    if "title':" in text and "description" in compact:
        return True
    return False


def coerce_chapter_title(
    result: object,
    fallback: str,
    *,
    max_words: int = 5,
    leak_markers: tuple[str, ...] = _LEAK_MARKERS,
) -> str:
    """
    Pull only the title field. Never str() the whole LLM/schema object
    (ADR-025 stage 5 — that produced chapter_03_chaptertitleresultdescription…).
    """
    raw = ""
    if isinstance(result, ChapterTitleResult):
        raw = result.title or ""
    elif isinstance(result, dict):
        value = result.get("title")
        raw = value if isinstance(value, str) else ""
    elif isinstance(result, str):
        raw = result
    else:
        attr = getattr(result, "title", None)
        raw = attr if isinstance(attr, str) else ""

    raw = raw.strip()
    raw = re.sub(r"^[\'\"]|[\'\"]$", "", raw)
    words = raw.split()
    if len(words) > max_words:
        raw = " ".join(words[:max_words])
    if not raw or looks_like_schema_leak(raw, leak_markers=leak_markers):
        return fallback
    return raw


def _top_entity_names(state: GraphDistillerState, chapter_id: int, limit: int = 5) -> list[str]:
    member_ids = set((state.get("toc") or {}).get(chapter_id) or [])
    nodes: list[GraphNode] = [
        n for n in (state.get("nodes") or []) if n.id in member_ids
    ]
    nodes.sort(key=lambda n: n.pagerank_score, reverse=True)
    return [n.id for n in nodes[:limit]]


def _generate_title(entity_names: list[str]) -> str:
    if not entity_names:
        return "Untitled Chapter"
    cfg = get_settings()
    rules = get_rules(cfg.rules_dir).planner
    llm = default_llm_chat(cfg, model=cfg.extraction_model)
    structured = llm.with_structured_output(ChapterTitleResult)
    prompt = (
        f"이 개념들을 포괄하는 사람이 읽기 좋은 챕터 제목을 {rules.max_title_words}단어 이내로 생성하라.\n"
        f"Entities: {', '.join(entity_names)}"
    )
    system_prompt = load_prompt(
        cfg.prompts_dir, "phase4_planner.md", fallback=TITLE_SYSTEM
    )
    result = structured.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
    )
    fallback = entity_names[0].replace("_", " ").title()
    return coerce_chapter_title(
        result, fallback, max_words=rules.max_title_words, leak_markers=tuple(rules.leak_markers)
    )


def planner_node(state: GraphDistillerState) -> dict:
    """
    Select next unfinished chapter and assign a title (SPEC Level 2 + 3.5).
    If nothing remains, current_chapter_id becomes None (END signal).
    """
    pending = pending_chapter_ids(state)
    if not pending:
        logger.debug("Planner: no pending chapters — synthesis complete.")
        return {"current_chapter_id": None, "repo_map": ""}

    leak_markers = tuple(get_rules(get_settings().rules_dir).planner.leak_markers)
    chapter_id = pending[0]
    entities = _top_entity_names(state, chapter_id, limit=5)
    titles = dict(state.get("chapter_titles") or {})
    cached = titles.get(chapter_id)
    if isinstance(cached, str) and cached.strip() and not looks_like_schema_leak(cached, leak_markers=leak_markers):
        title = cached.strip()
    else:
        fallback = _fallback_title(entities, chapter_id)
        try:
            title = _generate_title(entities)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Title LLM failed (%s); using fallback.", exc)
            title = fallback
        if looks_like_schema_leak(title, leak_markers=leak_markers):
            title = fallback

    titles[chapter_id] = title
    logger.debug(
        "Planner: chapter_id=%s title=%r entities=%s", chapter_id, title, entities
    )
    return {
        "current_chapter_id": chapter_id,
        "chapter_titles": titles,
        "repo_map": "",
        "last_verification": "",
    }
