"""Unit: W3-B / ADR-037 — full-DB Phase 2 LLM defaults stay off ($0)."""

from __future__ import annotations

import pytest

from graph_distiller.cli_rag_skeleton import build_parser
from graph_distiller.rag_gates import (
    PHASE2_LLM_DEFAULT,
    assert_sample_gate,
    fatal_if_db_mode_ambiguous,
    fatal_if_full_db_default,
    missing_sample_sizes,
    phase2_llm_default_enabled,
)

pytestmark = pytest.mark.unit


def test_phase2_llm_default_is_hard_off() -> None:
    assert PHASE2_LLM_DEFAULT is False
    assert phase2_llm_default_enabled() is False


def test_parser_defaults_forbid_silent_full_phase2() -> None:
    args = build_parser().parse_args([])
    assert args.llm is False
    assert args.n_jo is None
    assert args.m_qna is None
    assert missing_sample_sizes(args.n_jo, args.m_qna)
    assert fatal_if_full_db_default(
        n_jo=args.n_jo, m_qna=args.m_qna, llm=args.llm
    )


def test_llm_flag_alone_still_requires_sample() -> None:
    args = build_parser().parse_args(["--llm"])
    assert args.llm is True
    reason = fatal_if_full_db_default(
        n_jo=args.n_jo, m_qna=args.m_qna, llm=True
    )
    assert reason is not None
    assert "LLM" in reason or "sample" in reason.lower()


def test_explicit_sample_without_llm_is_allowed() -> None:
    assert (
        fatal_if_full_db_default(n_jo=5, m_qna=10, llm=False) is None
    )
    assert_sample_gate(n_jo=5, m_qna=10, llm=False)


def test_explicit_sample_with_llm_opt_in_is_allowed() -> None:
    assert fatal_if_full_db_default(n_jo=5, m_qna=10, llm=True) is None
    assert_sample_gate(n_jo=5, m_qna=0, llm=True)


def test_assert_sample_gate_raises_on_omit() -> None:
    with pytest.raises(ValueError, match="ADR-037"):
        assert_sample_gate(n_jo=None, m_qna=10, llm=False)


def test_explicit_full_db_flag_is_not_silent_default() -> None:
    assert fatal_if_db_mode_ambiguous(sample=False, full_db=True) is None
    assert fatal_if_db_mode_ambiguous(sample=True, full_db=True) is not None
    assert fatal_if_db_mode_ambiguous(sample=False, full_db=False) is not None
    assert (
        fatal_if_db_mode_ambiguous(
            sample=True, full_db=False, n_jo=2, m_qna=2, llm=False
        )
        is None
    )
