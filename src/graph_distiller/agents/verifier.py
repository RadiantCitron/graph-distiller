"""Verifier Agent — citation / hallucination gate + poison-pill (SPEC 3.5)."""

from __future__ import annotations

import logging
import re
from typing import List, Set, Tuple

from graph_distiller.agents.helpers import append_error, drop_draft
from graph_distiller.agents.repomap import part2_doc_ids
from graph_distiller.config import get_settings
from graph_distiller.contracts import (
    DOC_ID_ONLY_BLOCK_RE,
    DOC_ID_RE,
    TAG_FATAL,
    TAG_VERIFY_FAIL,
)
from graph_distiller.rules import get_rules
from graph_distiller.schema import ChapterDraft, GraphDistillerState

logger = logging.getLogger(__name__)

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
# Fallback only — the live pattern is data/rules/phase4_verifier.toml::sentence_split_pattern
# (ADR-060). Kept here so a missing/corrupt rules dir still verifies chapters correctly.
SENTENCE_RE = re.compile(r"(?<=[.!?。])\s+|(?<=다\.)\s+")

# ADR-057 조치3 — Synthesizer(deepseek-v4.1-flash) 자체 생성 결함으로 관측된 두 종류의
# 텍스트 손상. "음절 누락"(예: 풀리지→리지) 자체는 결과물이 여전히 정상적인 한글이라
# 사전/형태소 분석 없이는 못 잡는다(범위 밖) — 여기서는 실제로 관측됐던, 사전 없이도
# 명확히 판별 가능한 두 패턴만 잡는다.
# (1) 외래 문자(라틴/키릴)가 한글 음절 사이에 공백 없이 끼어드는 경우 (예: "어떻게살а것인가"
#     — 러시아어 а가 한글 "ㅏ"처럼 보이는 homoglyph로 섞여든 사례가 실제로 있었음).
_FOREIGN_IN_HANGUL_RE = re.compile(
    r"[가-힣][a-zA-Zа-яА-ЯёЁ]+[가-힣]"
)
# (2) 모델/토크나이저의 특수 토큰이 그대로 본문에 새어나온 경우 (예: "<bos>").
_STRAY_SPECIAL_TOKEN_RE = re.compile(
    r"<\s*/?\s*(bos|eos|pad|unk|s|sep|cls|mask)\s*>", re.IGNORECASE
)
# (3) 완성된 한글 음절(가~힣)이 아니라 낱자(자모, ㄱ~ㅣ)가 단어 중간에 홀로 낀 경우.
#     "ㅋㅋㅋ"/"ㅎㅎ" 같은 반복 감탄 표현(같은 자모가 연달아 나오는 경우)은 정상적인
#     구어체 인용이므로 제외한다.
_LONE_JAMO_RE = re.compile(r"[가-힣][ㄱ-ㅣ](?![ㄱ-ㅣ])[가-힣]")


def detect_text_corruption(content: str) -> List[str]:
    """
    Heuristic, dependency-free corruption detectors (ADR-057 조치3).

    Returns a list of human-readable findings; empty means "nothing caught"
    (NOT the same as "definitely clean" — silent single-syllable omission is
    a known blind spot, see module docstring above the regexes).
    """
    text = _strip_fenced_code(content)
    findings: List[str] = []
    foreign = _FOREIGN_IN_HANGUL_RE.findall(text)
    if foreign:
        findings.append(
            f"foreign character embedded inside Hangul word (found {len(foreign)}x)"
        )
    stray = _STRAY_SPECIAL_TOKEN_RE.findall(text)
    if stray:
        findings.append(f"stray model special token(s): {sorted(set(stray))}")
    lone_jamo = _LONE_JAMO_RE.findall(text)
    if lone_jamo:
        findings.append(f"lone Hangul jamo mid-word (found {len(lone_jamo)}x)")
    return findings


def _valid_doc_ids(state: GraphDistillerState) -> Set[str]:
    """ADR-055: allowlist = this chapter's RepoMap Part 2 DOC_IDs only."""
    return part2_doc_ids(state.get("repo_map") or "")


def _find_draft(state: GraphDistillerState, chapter_id: int) -> ChapterDraft | None:
    for draft in state.get("drafts") or []:
        if draft.chapter_id == chapter_id:
            return draft
    return None


def _markdown_broken(content: str, *, check_fences: bool = True) -> bool:
    if not content.strip():
        return True
    if check_fences and content.count("```") % 2 != 0:
        return True
    return False


def _strip_fenced_code(content: str) -> str:
    return FENCE_RE.sub("\n", content)


def body_paragraphs(content: str) -> List[str]:
    """Prose blocks that must be grounded (ADR-025 stage 4)."""
    text = _strip_fenced_code(content)
    parts = re.split(r"\n\s*\n", text)
    body: List[str] = []
    for part in parts:
        block = part.strip()
        if not block:
            continue
        if re.fullmatch(r"#{1,6}\s+\S[^\n]*", block):
            continue
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", block):
            continue
        if DOC_ID_ONLY_BLOCK_RE.fullmatch(block):
            continue
        body.append(block)
    return body


def _sentence_count(text: str, *, sentence_re: "re.Pattern[str]" = SENTENCE_RE) -> int:
    stripped = DOC_ID_RE.sub(" ", text).strip()
    if not stripped:
        return 0
    bits = [s.strip() for s in sentence_re.split(stripped) if s.strip()]
    return max(1, len(bits))


def verify_draft(
    draft: ChapterDraft,
    valid_ids: Set[str],
    *,
    paragraph_ratio: float | None = None,
    sentence_ratio: float | None = None,
) -> Tuple[bool, str]:
    """
    PASS:
      - at least one [DOC_ID: xxx]
      - every cited id is in valid_ids (caller: Part 2 DOC_IDs, ADR-055)
      - citation density over body paragraphs / sentences (ADR-025 stage 4)
    FAIL:
      - missing/unknown citations, thin density, or broken markdown
    """
    cfg = get_settings()
    rules = get_rules(cfg.rules_dir).verifier
    p_ratio = (
        cfg.verify_citation_paragraph_ratio
        if paragraph_ratio is None
        else paragraph_ratio
    )
    s_ratio = (
        cfg.verify_citation_sentence_ratio
        if sentence_ratio is None
        else sentence_ratio
    )
    p_ratio = min(1.0, max(0.0, float(p_ratio)))
    s_ratio = min(1.0, max(0.0, float(s_ratio)))
    sentence_re = re.compile(rules.sentence_split_pattern)

    citations = DOC_ID_RE.findall(draft.content)
    if len(citations) < rules.min_citations:
        return False, f"Fewer than {rules.min_citations} [DOC_ID: ...] citation(s) found"
    if _markdown_broken(draft.content, check_fences=rules.check_markdown_fences_balanced):
        return False, "Broken markdown format"
    corruption = detect_text_corruption(draft.content)
    if corruption:
        return False, f"Text corruption detected: {corruption}"
    unknown = sorted({cid for cid in citations if cid not in valid_ids})
    if unknown:
        return False, f"Unknown DOC_IDs cited: {unknown}"

    paras = body_paragraphs(draft.content)
    if paras and p_ratio > 0:
        cited_paras = sum(1 for p in paras if DOC_ID_RE.search(p))
        got = cited_paras / len(paras)
        if got + 1e-9 < p_ratio:
            return False, (
                f"Citation paragraph density {cited_paras}/{len(paras)} "
                f"< {p_ratio}"
            )

    if paras and s_ratio > 0:
        sentences = sum(_sentence_count(p, sentence_re=sentence_re) for p in paras)
        n_cites = len(citations)
        if sentences > 0:
            got_s = n_cites / sentences
            if got_s + 1e-9 < s_ratio:
                return False, (
                    f"Citation sentence density {n_cites}/{sentences} "
                    f"< {s_ratio}"
                )

    return True, "ok"


def _append_fatal_log(message: str) -> None:
    from graph_distiller.observability import emit_loop_error

    emit_loop_error(message)


def verifier_node(state: GraphDistillerState) -> dict:
    chapter_id = state.get("current_chapter_id")
    if chapter_id is None:
        return {"last_verification": "pass"}

    draft = _find_draft(state, chapter_id)
    errors = list(state.get("errors") or [])
    retry_counts = dict(state.get("retry_counts") or {})
    poisoned = list(state.get("poisoned_chapters") or [])
    max_retries = get_settings().max_chapter_retries

    if draft is None:
        ok, reason = False, "No draft found for current chapter"
    else:
        ok, reason = verify_draft(draft, _valid_doc_ids(state))

    if ok:
        logger.debug("Verifier PASS: chapter_id=%s", chapter_id)
        return {"last_verification": "pass", "errors": errors}

    # FAIL path
    count = retry_counts.get(chapter_id, 0) + 1
    retry_counts[chapter_id] = count
    errors = append_error(
        {**state, "errors": errors},
        f"{TAG_VERIFY_FAIL} Chapter {chapter_id} attempt={count}: {reason}",
    )
    # Remove bad draft so chapter stays pending (unless poisoned)
    drafts = drop_draft(state, chapter_id)
    # ADR-057 조치2: 다음 재시도의 Synthesizer가 같은 실수를 반복하지 않도록
    # 실패 사유를 챕터별로 남긴다 (retry feedback loop).
    fail_reasons = dict(state.get("chapter_fail_reasons") or {})
    fail_reasons[chapter_id] = reason

    if count >= max_retries:
        poisoned = list(dict.fromkeys(poisoned + [chapter_id]))
        fatal = (
            f"{TAG_FATAL} Chapter {chapter_id} failed after {max_retries} retries. "
            f"Reason: {reason}"
        )
        errors.append(fatal)
        _append_fatal_log(fatal)
        # Poison: skip permanently; route like "continue" via planner
        return {
            "last_verification": "pass",
            "errors": errors,
            "retry_counts": retry_counts,
            "poisoned_chapters": poisoned,
            "drafts": drafts,
            "chapter_fail_reasons": fail_reasons,
            "current_chapter_id": None,  # force planner to pick next
        }

    # Retryable failure: debug only (console silenced under tqdm; FATAL uses tqdm.write)
    logger.debug(
        "Verifier FAIL: chapter_id=%s attempt=%s reason=%s",
        chapter_id,
        count,
        reason,
    )
    return {
        "last_verification": "fail",
        "errors": errors,
        "retry_counts": retry_counts,
        "drafts": drafts,
        "chapter_fail_reasons": fail_reasons,
    }
