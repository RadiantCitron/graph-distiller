"""Unit: ragdb sample merge / chapter-first select — $0 (no Postgres)."""

from __future__ import annotations

import pytest

from graph_distiller.ingestion.rdb import (
    LawRow,
    QnaCase,
    RagCounts,
    build_cites_graph,
    build_sample_from_rows,
    children_index,
    collect_descendants,
    merge_jo_bundle,
    require_sample_sizes,
    select_jo_rows_chapter_first,
    select_qna_cases_cite_first,
)
from graph_distiller.ingestion.law_cites import JoRef

pytestmark = pytest.mark.unit


def _tree() -> list[LawRow]:
    # 장1 → 조10(제10조) → 항, 조11(제11조)
    # 장2 → 조20(제20조)
    return [
        LawRow(1, None, "장", "제1장 총칙"),
        LawRow(10, 1, "조", "제10조(정의) 본조."),
        LawRow(101, 10, "항", "① 근로자란…"),
        LawRow(11, 1, "조", "제11조(적용) 제10조를 따른다."),
        LawRow(2, None, "장", "제2장 근로계약"),
        LawRow(20, 2, "조", "제20조(계약) 계약."),
    ]


def test_require_sample_sizes_rejects_full_default() -> None:
    with pytest.raises(ValueError):
        require_sample_sizes(0, 5)
    assert require_sample_sizes(3, 0) == (3, 0)


def test_merge_jo_includes_hang() -> None:
    rows = _tree()
    by_parent = children_index(rows)
    by_id = {r.id: r for r in rows}
    jang = {r.id: r for r in rows if r.level == "장"}
    jo = next(r for r in rows if r.id == 10)
    bundle = merge_jo_bundle(jo, by_parent, by_id=by_id, jang_by_id=jang)
    assert bundle is not None
    assert bundle.jo_ref == JoRef(10)
    assert "① 근로자" in bundle.content_merged
    assert bundle.descendant_ids == [101]
    assert bundle.jang_id == 1


def test_collect_descendants() -> None:
    by_parent = children_index(_tree())
    desc = collect_descendants(10, by_parent)
    assert [d.id for d in desc] == [101]


def test_select_jo_chapter_first_prefers_largest_jang() -> None:
    selected, notes = select_jo_rows_chapter_first(_tree(), n_jo=2)
    ids = [r.id for r in selected]
    # n_jo=2 < 4 → no second-jang reserve; take both from largest 장
    assert ids == [10, 11]
    assert any("selected_jo_count=2" in n for n in notes)


def test_select_jo_reserves_second_jang_when_n_ge_4() -> None:
    rows = _tree() + [
        LawRow(12, 1, "조", "제12조 추가."),
        LawRow(13, 1, "조", "제13조 추가."),
    ]
    selected, _ = select_jo_rows_chapter_first(rows, n_jo=4)
    jang_ids = set()
    by_id = {r.id: r for r in rows}
    from graph_distiller.ingestion.rdb import _find_jang_ancestor

    for jo in selected:
        jang_ids.add(_find_jang_ancestor(jo, by_id))
    assert len(selected) == 4
    assert len(jang_ids) >= 2


def test_select_qna_cite_first() -> None:
    cases = [
        QnaCase(1, None, "관련 없음"),
        QnaCase(2, None, "근거는 제10조이다"),
        QnaCase(3, None, "제10조와 제11조"),
    ]
    chosen, _ = select_qna_cases_cite_first(
        cases, [JoRef(10), JoRef(11)], m_qna=2
    )
    assert [c.case_id for c in chosen] == [3, 2]
    assert chosen[0].cite_hits == 2


def test_build_sample_and_cites_graph() -> None:
    rows = _tree()
    counts = RagCounts(6, 3, 2, 3, 3, 0)
    cases = [
        QnaCase(1, None, "제10조 위반"),
        QnaCase(2, None, "제99조만"),
    ]
    sample = build_sample_from_rows(rows, cases, counts, n_jo=2, m_qna=2)
    assert len(sample.jo_bundles) == 2
    nodes, edges, meta = build_cites_graph(sample)
    ids = {n.id for n in nodes}
    assert "LAW_JO_10" in ids
    assert "LAW_JO_11" in ids
    assert any(e.relationship == "GROUNDS" for e in edges)
    assert any(m.provenance == "cites" for m in meta)
    # 제11조 body cites 제10조
    assert any(
        {e.source, e.target} == {"LAW_JO_11", "LAW_JO_10"} for e in edges
    )


def test_full_corpus_payload_uses_all_jo() -> None:
    """load_full is build_sample_from_rows with n_jo = available 조 (explicit)."""
    rows = _tree()
    counts = RagCounts(6, 3, 2, 2, 2, 0)
    cases = [
        QnaCase(1, None, "제10조 위반"),
        QnaCase(2, None, "제20조"),
    ]
    sample = build_sample_from_rows(
        rows, cases, counts, n_jo=counts.labor_law_jo, m_qna=counts.labor_qna_cases
    )
    assert sample.sample_n_jo == 3
    assert len(sample.jo_bundles) == 3
    assert sample.sample_m_qna == 2
    assert len(sample.qna_cases) == 2
