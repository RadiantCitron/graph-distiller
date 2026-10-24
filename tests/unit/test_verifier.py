"""Unit: Verifier citation density — no network / $0."""

from __future__ import annotations

import pytest

from graph_distiller.agents.repomap import part2_doc_ids
from graph_distiller.agents.verifier import (
    _valid_doc_ids,
    body_paragraphs,
    detect_text_corruption,
    verifier_node,
    verify_draft,
)
from graph_distiller.schema import ChapterDraft, ChunkMeta, empty_state

pytestmark = pytest.mark.unit


def _draft(content: str) -> ChapterDraft:
    return ChapterDraft(
        chapter_id=0,
        title="T",
        content=content,
        cited_doc_ids=[],
    )


def test_part2_doc_ids_ignores_part1_and_extracts_part2() -> None:
    repo = (
        "[Chapter: 0]\n"
        "## Part 1: Core Concepts\n"
        "- HOW_TO_LIVE (PageRank: 0.5)\n"
        "## Part 2: Grounding Context\n"
        "[DOC_ID: memo_a] body a\n"
        "[DOC_ID: memo_b] body b\n"
    )
    assert part2_doc_ids(repo) == {"memo_a", "memo_b"}
    assert "HOW_TO_LIVE" not in part2_doc_ids(repo)


def test_valid_doc_ids_uses_repo_map_part2_not_all_chunks() -> None:
    state = empty_state()
    state["chunks"] = [
        ChunkMeta(
            chunk_id="c1",
            source_doc_id="memo_a",
            content="a",
            token_count=1,
        ),
        ChunkMeta(
            chunk_id="c2",
            source_doc_id="memo_other_in_run",
            content="other",
            token_count=1,
        ),
    ]
    state["repo_map"] = (
        "## Part 1: Core Concepts\n"
        "- NODE_X (PageRank: 0.1)\n"
        "## Part 2: Grounding Context\n"
        "[DOC_ID: memo_a] only this chapter grounding\n"
    )
    assert _valid_doc_ids(state) == {"memo_a"}


def test_verify_draft_rejects_run_wide_id_absent_from_part2() -> None:
    ok, reason = verify_draft(
        _draft(
            "# Title\n\n"
            "Cites another doc in the run. [DOC_ID: memo_other_in_run]\n"
        ),
        {"memo_a"},
        paragraph_ratio=1.0,
        sentence_ratio=0.0,
    )
    assert ok is False
    assert "Unknown DOC_IDs" in reason
    assert "memo_other_in_run" in reason


def test_body_paragraphs_skips_heading_and_citation_only() -> None:
    text = (
        "# Title\n\n"
        "First claim. [DOC_ID: a]\n\n"
        "[DOC_ID: a]\n\n"
        "Second claim without cite.\n"
    )
    paras = body_paragraphs(text)
    assert len(paras) == 2
    assert "First claim" in paras[0]
    assert "Second claim" in paras[1]


def test_verify_draft_rejects_unsourced_paragraph() -> None:
    ok, reason = verify_draft(
        _draft(
            "# Title\n\n"
            "Grounded sentence. [DOC_ID: memo1]\n\n"
            "Generic filler with no source.\n"
        ),
        {"memo1"},
        paragraph_ratio=1.0,
        sentence_ratio=0.0,
    )
    assert ok is False
    assert "paragraph density" in reason


def test_verify_draft_accepts_inline_cite_per_paragraph() -> None:
    ok, reason = verify_draft(
        _draft(
            "# Title\n\n"
            "Auth talks to billing. [DOC_ID: memo1]\n"
        ),
        {"memo1"},
        paragraph_ratio=1.0,
        sentence_ratio=0.25,
    )
    assert ok is True
    assert reason == "ok"


def test_verify_draft_rejects_thin_sentence_density() -> None:
    ok, reason = verify_draft(
        _draft(
            "# Title\n\n"
            "One. Two. Three. Four. Five. [DOC_ID: memo1]\n"
        ),
        {"memo1"},
        paragraph_ratio=1.0,
        sentence_ratio=0.25,
    )
    assert ok is False
    assert "sentence density" in reason


def test_verifier_node_fail_records_reason_for_retry_feedback() -> None:
    """ADR-057 조치2: FAIL 시 chapter_fail_reasons[chapter_id]에 사유를 남긴다."""
    state = empty_state()
    state["current_chapter_id"] = 0
    state["drafts"] = []  # No draft -> "No draft found for current chapter"

    out = verifier_node(state)

    assert out["last_verification"] == "fail"
    assert out["chapter_fail_reasons"][0] == "No draft found for current chapter"
    assert out["retry_counts"][0] == 1


def test_detect_text_corruption_clean_text_has_no_findings() -> None:
    assert detect_text_corruption("정상적인 한글 문장이다. 아무 문제 없다.") == []


def test_detect_text_corruption_catches_foreign_char_inside_hangul() -> None:
    # "어떻게살а것인가" — Cyrillic а (U+0430) sandwiched between Hangul syllables,
    # an actually-observed homoglyph corruption in this project's history.
    findings = detect_text_corruption("제목: 어떻게살а것인가")
    assert any("foreign character" in f for f in findings)


def test_detect_text_corruption_catches_stray_special_token() -> None:
    findings = detect_text_corruption("문장이 끝났다<bos>다음 문장.")
    assert any("special token" in f for f in findings)


def test_detect_text_corruption_catches_lone_jamo_mid_word() -> None:
    findings = detect_text_corruption("이것ㅅ은 손상된 예시다.")
    assert any("lone Hangul jamo" in f for f in findings)


def test_detect_text_corruption_allows_laughter_interjections() -> None:
    # "ㅋㅋㅋ"/"ㅎㅎ" are legitimate colloquial quotes, not corruption.
    assert detect_text_corruption("웃겨서 그렇다ㅋㅋㅋ진짜 그렇다니까ㅎㅎ보라고.") == []


def test_verify_draft_rejects_corrupted_text_even_with_valid_citation() -> None:
    ok, reason = verify_draft(
        _draft(
            "# Title\n\n"
            "어떻게살а것인가에 대한 이야기다. [DOC_ID: memo1]\n"
        ),
        {"memo1"},
    )
    assert ok is False
    assert "Text corruption detected" in reason


def test_verifier_node_poison_still_records_last_reason() -> None:
    """포이즌되는 마지막 시도도 사유는 남겨(향후 분석용), 다만 재시도는 없다."""
    from dataclasses import replace

    from graph_distiller.config import get_settings, set_runtime_settings

    prev = get_settings()
    set_runtime_settings(replace(prev, max_chapter_retries=1))
    try:
        state = empty_state()
        state["current_chapter_id"] = 0
        state["drafts"] = []
        out = verifier_node(state)
        assert 0 in out["poisoned_chapters"]
        assert out["chapter_fail_reasons"][0] == "No draft found for current chapter"
    finally:
        set_runtime_settings(prev)
