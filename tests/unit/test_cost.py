"""Unit: cost.ModelRate / calculate_cost — no network."""

from __future__ import annotations

import pytest
import tiktoken

from graph_distiller.cost import (
    ModelRate,
    calculate_cost,
    count_tokens,
    format_cost_status,
    merge_cost_status,
    rate_from_per_1m,
)

pytestmark = pytest.mark.unit


def test_count_tokens_empty() -> None:
    assert count_tokens("") == 0
    assert count_tokens(None) == 0  # type: ignore[arg-type]


def test_count_tokens_matches_tiktoken_cl100k() -> None:
    text = "hello graph_distiller token diet"
    expected = len(tiktoken.get_encoding("cl100k_base").encode(text))
    assert count_tokens(text) == expected


def test_rate_from_per_1m_free_and_known() -> None:
    assert rate_from_per_1m(0.0, 0.0).status == "free"
    known = rate_from_per_1m(0.15, 0.60)
    assert known.status == "known"
    assert known.input_usd_per_1m == pytest.approx(0.15)


def test_calculate_cost_known_matches_table() -> None:
    rate = rate_from_per_1m(0.15, 0.60)
    # 1M in + 1M out → 0.15 + 0.60
    assert calculate_cost("x", 1_000_000, 1_000_000, rate=rate) == pytest.approx(0.75)


def test_calculate_cost_unknown_and_free_are_zero() -> None:
    assert calculate_cost("x", 1000, 1000, rate=ModelRate.unknown()) == 0.0
    assert calculate_cost("x", 1000, 1000, rate=ModelRate.free()) == 0.0
    assert calculate_cost("x", 1000, 1000) == 0.0


def test_format_and_merge_status() -> None:
    assert format_cost_status(0.0, "free") == "FREE"
    assert format_cost_status(0.0, "unknown") == "UNKNOWN"
    assert format_cost_status(0.12, "known").startswith("$")
    assert merge_cost_status("known", "unknown") == "unknown"
    assert merge_cost_status("free", "free") == "free"
    assert merge_cost_status("known", "free") == "known"
