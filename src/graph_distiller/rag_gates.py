"""ADR-037 / W3-B — full-DB Phase 2 LLM must not be a silent default.

Pure predicates. No DB, no LLM, no Settings I/O.
CLI and skeleton call these before any paid Phase 2 path.
"""

from __future__ import annotations

from typing import Optional


# Locked contract: Phase 2 LLM on RAG is opt-in only (--llm).
PHASE2_LLM_DEFAULT = False

_MISSING_SAMPLE_MSG = (
    "Sample sizes required: pass --n-jo N and --m-qna M "
    "(full-DB default forbidden — ADR-037 / W3-B)."
)


def phase2_llm_default_enabled() -> bool:
    """Return False forever unless ADR-037 is explicitly revised."""
    return PHASE2_LLM_DEFAULT


def missing_sample_sizes(n_jo: Optional[int], m_qna: Optional[int]) -> bool:
    """True when caller omitted sample sizes (would imply full-DB default)."""
    return n_jo is None or m_qna is None


def fatal_if_full_db_default(
    *,
    n_jo: Optional[int],
    m_qna: Optional[int],
    llm: bool = False,
) -> Optional[str]:
    """
    Return a fatal reason string, or None if the planned run is allowed.

    - Missing --n-jo/--m-qna → fatal (even for cites-only).
    - --llm with missing samples → fatal (same gate; never invent full-DB).
    """
    if missing_sample_sizes(n_jo, m_qna):
        if llm:
            return (
                "Phase 2 LLM requires an explicit sample: pass --n-jo N and "
                "--m-qna M together with --llm (full-DB LLM forbidden — ADR-037)."
            )
        return _MISSING_SAMPLE_MSG
    return None


def assert_sample_gate(
    *,
    n_jo: Optional[int],
    m_qna: Optional[int],
    llm: bool = False,
) -> None:
    """Raise ValueError when full-DB default would apply."""
    reason = fatal_if_full_db_default(n_jo=n_jo, m_qna=m_qna, llm=llm)
    if reason:
        raise ValueError(reason)


def fatal_if_db_mode_ambiguous(
    *,
    sample: bool,
    full_db: bool,
    n_jo: Optional[int] = None,
    m_qna: Optional[int] = None,
    llm: bool = False,
) -> Optional[str]:
    """
    Compiler --source db: require exactly one of --sample / --full-db.

    Explicit --full-db is allowed (ADR-037 silent default remains fatal).
    --sample still requires --n-jo / --m-qna.
    """
    if sample and full_db:
        return "Pass exactly one of --sample or --full-db (not both)."
    if not sample and not full_db:
        return (
            "Pass --sample (with --n-jo N --m-qna M) or --full-db "
            "(silent full-DB forbidden — ADR-037)."
        )
    if full_db:
        return None
    return fatal_if_full_db_default(n_jo=n_jo, m_qna=m_qna, llm=llm)
