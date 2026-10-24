"""Unit: OpenRouter pricing parse + RateBook — no live network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph_distiller.cost import ModelRate, rate_from_per_1m
from graph_distiller.observability import PhaseUsage
from graph_distiller.pricing import (
    RateBook,
    rates_from_models_payload,
    set_active_rate_book,
)

pytestmark = pytest.mark.unit


SAMPLE_PAYLOAD = {
    "data": [
        {
            "id": "openai/gpt-4o-mini",
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
        },
        {
            "id": "vendor/free-model",
            "pricing": {"prompt": "0", "completion": "0"},
        },
        {
            "id": "vendor/broken",
            "pricing": {"prompt": "not-a-number", "completion": "0"},
        },
    ]
}


def test_rates_from_models_payload_known_free_unknown() -> None:
    rates = rates_from_models_payload(
        SAMPLE_PAYLOAD,
        wanted=["openai/gpt-4o-mini", "vendor/free-model", "vendor/broken"],
    )
    assert rates["openai/gpt-4o-mini"].status == "known"
    assert rates["openai/gpt-4o-mini"].input_usd_per_1m == pytest.approx(0.15)
    assert rates["openai/gpt-4o-mini"].output_usd_per_1m == pytest.approx(0.60)
    assert rates["vendor/free-model"].status == "free"
    assert rates["vendor/broken"].status == "unknown"


def test_rate_book_missing_slug_is_unknown() -> None:
    book = RateBook(by_id={"a/b": rate_from_per_1m(1.0, 2.0)}, source="test")
    assert book.rate_for("a/b").status == "known"
    assert book.rate_for("missing/model").status == "unknown"
    lines = book.lines_for(["a/b", "missing/model"])
    assert any("a/b:" in line and "per 1M" in line for line in lines)
    assert any("missing/model: UNKNOWN" in line for line in lines)


def test_phase_usage_postfix_free_unknown_known() -> None:
    cb = SimpleNamespace(prompt_tokens=1000, completion_tokens=500, total_tokens=1500, total_cost=0.0)
    set_active_rate_book(
        RateBook(
            by_id={
                "free/m": ModelRate.free(),
                "pay/m": rate_from_per_1m(1.0, 2.0),
            },
            source="test",
        )
    )
    try:
        free_u = PhaseUsage.from_callback(cb, model="free/m")
        assert free_u.cost_status == "free"
        assert "FREE" in free_u.cost_postfix

        unk = PhaseUsage.from_callback(cb, model="no/such")
        assert unk.cost_status == "unknown"
        assert "UNKNOWN" in unk.cost_postfix

        known = PhaseUsage.from_callback(cb, model="pay/m")
        assert known.cost_status == "known"
        assert known.total_cost_usd == pytest.approx(0.001 + 0.001)
        assert "$" in known.cost_postfix
    finally:
        set_active_rate_book(RateBook())
