"""Unit: judge collector — the only paid collector, exercised here with a fake LLM ($0)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from harness.collectors import judge as judge_collector
from harness.judge_schema import EvaluationResult
from graph_distiller.config import Settings, settings as real_settings
from graph_distiller.rules import Rules

pytestmark = pytest.mark.unit


class _FakeStructuredLLM:
    def __init__(self, result: EvaluationResult) -> None:
        self._result = result
        self.last_messages = None

    def invoke(self, messages):  # noqa: ANN001
        self.last_messages = messages
        return self._result


class _FakeLLM:
    def __init__(self, result: EvaluationResult) -> None:
        self._structured = _FakeStructuredLLM(result)

    def with_structured_output(self, schema):  # noqa: ANN001
        return self._structured


def _diet(**overrides) -> dict:
    base = {
        "phase1_tokens": 1000,
        "repo_map_tokens": 400,
        "compression_ratio_pct": 60.0,
        "core_retention_pct": 100.0,
        "expected_core_nodes": ["A", "B"],
        "missing_core_nodes": [],
        "repo_map_text": "[Chapter: 0]\n[DOC_ID: memo1] body",
    }
    base.update(overrides)
    return base


def test_run_judge_short_circuits_on_zero_phase1_tokens() -> None:
    result = judge_collector.run_judge(diet=_diet(phase1_tokens=0), settings=real_settings)
    assert result.compression_score == 1 and result.information_retention == 1


def test_run_judge_calls_llm_and_clips_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = EvaluationResult(
        compression_score=8, information_retention=9, feedback="line1\nline2\nline3\nline4"
    )
    fake = _FakeLLM(fake_result)
    monkeypatch.setattr(Settings, "build_chat_model", lambda self, *a, **k: fake)

    result = judge_collector.run_judge(diet=_diet(), settings=real_settings)
    assert result.compression_score == 8
    assert result.feedback.count("\n") == 2  # clipped to 3 lines (default rule)
    assert fake._structured.last_messages is not None


def test_run_judge_with_fallback_degrades_on_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self, *a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(Settings, "build_chat_model", boom)
    result, error = judge_collector.run_judge_with_fallback(
        diet=_diet(compression_ratio_pct=35.0, core_retention_pct=100.0), settings=real_settings
    )
    assert error == "network down"
    assert result.compression_score == 4  # 35/10 rounded
    assert result.information_retention == 10


def test_collect_reports_metrics_and_call_succeeded_check(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = EvaluationResult(compression_score=6, information_retention=7, feedback="ok")
    monkeypatch.setattr(Settings, "build_chat_model", lambda self, *a, **k: _FakeLLM(fake_result))

    out = judge_collector.collect(_diet(), settings=real_settings)
    names = {m.name: m.value for m in out.metrics}
    assert names["compression_score"] == 6
    assert names["information_retention"] == 7
    assert names["diet_score"] == 13.0
    assert any(c.check_id == "judge.call_succeeded" and c.passed for c in out.checks)


def test_collect_records_failed_check_when_judge_call_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self, *a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(Settings, "build_chat_model", boom)
    out = judge_collector.collect(_diet(), settings=real_settings)
    chk = next(c for c in out.checks if c.check_id == "judge.call_succeeded")
    assert chk.passed is False and chk.severity == "warn"
    assert "boom" in chk.message
