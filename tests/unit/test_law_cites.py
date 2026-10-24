"""Unit: Korean statute citation regex — $0 / no network."""

from __future__ import annotations

import pytest

from graph_distiller.ingestion.law_cites import (
    JoRef,
    cite_pairs,
    count_selected_jo_hits,
    law_node_id,
    parse_jo_refs,
    primary_jo_ref_from_article_text,
    qna_node_id,
)

pytestmark = pytest.mark.unit


def test_law_node_id_plain_and_sub() -> None:
    assert law_node_id(23) == "LAW_JO_23"
    assert law_node_id(23, 2) == "LAW_JO_23_2"
    with pytest.raises(ValueError):
        law_node_id(0)


def test_qna_node_id() -> None:
    assert qna_node_id(12) == "QNA_12"


def test_parse_jo_refs_variants() -> None:
    text = "근로기준법 제23조 및 제 24 조, 다시 제23조, 제56조의2를 본다."
    refs = parse_jo_refs(text)
    assert refs == [
        JoRef(23),
        JoRef(24),
        JoRef(56, 2),
    ]


def test_primary_jo_ref() -> None:
    assert primary_jo_ref_from_article_text("제1조(목적) 이 법은…").jo == 1
    assert primary_jo_ref_from_article_text("관련 없음") is None


def test_cite_pairs_exclude_self() -> None:
    pairs = cite_pairs("제10조는 제11조를 본다", source_node_id="LAW_JO_10")
    assert ("LAW_JO_10", "LAW_JO_11", JoRef(11)) in pairs
    assert all(t != "LAW_JO_10" for _, t, _ in pairs)


def test_count_selected_jo_hits() -> None:
    selected = [JoRef(23), JoRef(24)]
    assert count_selected_jo_hits("제23조와 제99조", selected) == 1
    assert count_selected_jo_hits("없음", selected) == 0
