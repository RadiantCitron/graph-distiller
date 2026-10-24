"""Synthesizer Agent — write markdown chapter from RepoMap (SPEC Phase 4)."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from graph_distiller.adapters.openrouter import default_llm_chat
from graph_distiller.agents.helpers import upsert_draft
from graph_distiller.config import get_settings
from graph_distiller.contracts import DOC_ID_RE
from graph_distiller.prompts import load_prompt
from graph_distiller.schema import ChapterDraft, GraphDistillerState

logger = logging.getLogger(__name__)

SYNTH_SYSTEM = """\
You are a technical book author compiling a Living Codex from a RepoMap.
Write a clear markdown chapter for human readers.

The RepoMap has two parts:
- Part 1 (Core Concepts): node NAMES and PageRank scores only. These are
  graph entity ids (concepts), NOT document ids.
- Part 2 (Grounding Context): source passages already tagged as
  [DOC_ID: <id>]. Those <id> values are the only legal citations.

MUST:
1. Use markdown headings (# / ##) and prose paragraphs.
2. Cite sources inline exactly as [DOC_ID: <id>] using IDs printed in
   Part 2 (Grounding Context), copied verbatim.
3. Do not invent document IDs that are not in Part 2.
4. Do not wrap the entire chapter in a single fenced code block.
5. Start with a single H1 matching the given chapter title.
6. Every prose paragraph must include at least one inline [DOC_ID: <id>]
   (do not park all citations in a single leftover line).
7. HARD CONSTRAINT — this is not a suggestion, it has caused real failures:
   Part 1 (Core Concepts) node names, chapter/book titles, and any string
   that is not printed verbatim inside a "[DOC_ID: ...]" tag in Part 2 are
   NEVER valid citations. Study the examples below before writing:

   CORRECT:
   Part 2 shows: [DOC_ID: labor_law_jo_12] "정당한 사유 없이 해고..."
   Chapter text: "...정당한 사유가 필요하다 [DOC_ID: labor_law_jo_12]."

   WRONG (Part 1 concept node used as a citation):
   Part 1 shows: - PROGRAMMER (PageRank: 0.18)
   Chapter text: "...프로그래머는... [DOC_ID: PROGRAMMER]"
   -> PROGRAMMER is a concept name, not a DOC_ID. Never cite it.

   WRONG (invented string, title, or stray token used as a citation):
   Chapter text: "...『어떻게 살 것인가』에서... [DOC_ID: 어떻게살인가]"
   -> "어떻게살인가" never appears as a DOC_ID in Part 2. Do not invent IDs
      from titles, headings, or any text you generate yourself, and never
      let typos or stray model tokens leak into a [DOC_ID: ...] tag.
"""


def synthesizer_node(state: GraphDistillerState) -> dict:
    chapter_id = state.get("current_chapter_id")
    if chapter_id is None:
        return {}

    title = (state.get("chapter_titles") or {}).get(chapter_id) or f"Chapter {chapter_id + 1}"
    repo_map = state.get("repo_map") or ""

    cfg = get_settings()
    llm = default_llm_chat(cfg, model=cfg.synthesis_model)
    fail_reason = (state.get("chapter_fail_reasons") or {}).get(chapter_id)
    retry_feedback = ""
    if fail_reason:
        # ADR-057 조치2: 이전 시도가 왜 거부됐는지 Verifier 사유를 그대로 알려줘서
        # 재시도가 같은 실수(예: 같은 잘못된 DOC_ID)를 반복하지 않게 한다.
        retry_feedback = (
            "\n[이전 시도 재작성 지시] 직전 시도는 다음 이유로 Verifier에게 거부됐다:\n"
            f"  {fail_reason}\n"
            "이번에는 반드시 이 문제를 피해서 다시 써라 — 위 사유에 언급된 잘못된 "
            "DOC_ID나 문제를 그대로 반복하지 말고, Part 2에 실제로 인쇄된 [DOC_ID: ...] "
            "값만 정확히 복사해서 인용하라.\n"
        )
    user_prompt = (
        f"Chapter title: {title}\n\n"
        "제공된 RepoMap을 바탕으로 마크다운 형식의 챕터를 집필하라. "
        "본문 문단마다 [DOC_ID: ...] 형태로 출처를 인용(Citation)하라.\n"
        "절대 Part 1(Core Concepts)의 노드 이름, 책/챕터 제목, 또는 네가 스스로 "
        "만들어낸 문자열을 DOC_ID로 사용하지 마라. 오직 Part 2(Grounding Context)에 "
        "명시된 [DOC_ID: ...] 형태의 실제 문서 ID만, 인쇄된 문자열 그대로 정확히 "
        "복사해서 인용하라. 오타나 특수 토큰이 인용에 섞이지 않도록 주의하라.\n"
        f"{retry_feedback}\n"
        f"--- RepoMap ---\n{repo_map}"
    )
    system_prompt = load_prompt(
        cfg.prompts_dir, "phase4_synthesizer.md", fallback=SYNTH_SYSTEM
    )
    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    )
    content = getattr(response, "content", str(response)).strip()
    if not content.startswith("#"):
        content = f"# {title}\n\n{content}"

    cited = list(dict.fromkeys(DOC_ID_RE.findall(content)))
    draft = ChapterDraft(
        chapter_id=chapter_id,
        title=title,
        content=content,
        cited_doc_ids=cited,
    )
    drafts = upsert_draft(state, draft)
    logger.debug(
        "Synthesizer: chapter_id=%s title=%r citations=%s chars=%s",
        chapter_id,
        title,
        cited,
        len(content),
    )
    return {"drafts": drafts}
