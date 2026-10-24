"""Unit: audit collectors for control room, logs and ledgers — synthetic inputs, $0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.collectors import config as config_collector
from harness.collectors import ledger as ledger_collector
from harness.collectors import logs as logs_collector
from graph_distiller.agents.synthesizer import SYNTH_SYSTEM
from graph_distiller.contracts import TAG_DLQ, TAG_FATAL, TAG_VERIFY_FAIL

pytestmark = pytest.mark.unit


def _metric(collected, name, subject=None, run_id=None):
    hits = [
        m
        for m in collected.metrics
        if m.name == name
        and (subject is None or m.subject == subject)
        and (run_id is None or m.run_id == run_id)
    ]
    assert len(hits) == 1, f"{name}: {hits}"
    return hits[0].value


def _check(collected, check_id, subject=None, run_id=None):
    hits = [
        c
        for c in collected.checks
        if c.check_id == check_id
        and (subject is None or c.subject == subject)
        and (run_id is None or c.run_id == run_id)
    ]
    assert len(hits) == 1, f"{check_id}: {hits}"
    return hits[0]


# --- logs -------------------------------------------------------------------


def test_count_error_tags_uses_contract_tags() -> None:
    errors = [
        f"{TAG_DLQ} chunk=a",
        f"{TAG_VERIFY_FAIL} Chapter 1 attempt=1",
        f"{TAG_FATAL} Chapter 1 failed after 3 retries.",
    ]
    assert logs_collector.count_error_tags(errors) == {
        "dlq": 1,
        "verify_fail": 1,
        "fatal": 1,
        "poison": 1,
    }


def test_logs_collect_counts_levels_tags_llm_requests_and_since(tmp_path: Path) -> None:
    log = tmp_path / "errors.log"
    log.write_text(
        "2026-09-16 10:00:00,001 [WARNING] graph_distiller.graph.extraction: [DLQ] chunk=a\n"
        "2026-09-16 10:00:01,001 [INFO] httpx2: HTTP Request: POST https://openrouter.ai/api/v1/chat/completions \"HTTP/1.1 200 OK\"\n"
        "2026-09-17 12:00:00,001 [ERROR] graph_distiller.agents.verifier: [FATAL] Chapter 2 failed after 3 retries.\n"
        "  continuation line of the error\n",
        encoding="utf-8",
    )
    whole = logs_collector.collect(log)
    assert _metric(whole, "lines") == 4
    assert _metric(whole, "tag_dlq") == 1
    assert _metric(whole, "tag_fatal") == 1
    assert _metric(whole, "llm_http_requests") == 1
    assert _metric(whole, "level_warning") == 1
    assert _check(whole, "logs.no_fatal").passed is False

    recent = logs_collector.collect(log, since="2026-09-17 00:00:00")
    assert _metric(recent, "lines") == 2
    assert _metric(recent, "tag_dlq") == 0
    assert _metric(recent, "llm_http_requests") == 0


def test_logs_missing_file_is_a_warn_check_not_a_crash(tmp_path: Path) -> None:
    out = logs_collector.collect(tmp_path / "nope.log")
    chk = _check(out, "logs.input_present")
    assert chk.passed is False and chk.severity == "warn"
    assert out.metrics == []


# --- ledger -----------------------------------------------------------------


def _write_ledger(runs: Path, run_id: str, *, poisoned: int, cost: float) -> None:
    d = runs / run_id
    d.mkdir(parents=True)
    ledger = {
        "run_id": run_id,
        "finished_at": "2026-09-16T10:00:00",
        "duration_s": 12.5,
        "exit_code": 0,
        "phases": {
            "phase4": {
                "duration_s": 10.0,
                "counts": {"chapters_planned": 2, "drafts": 2 - poisoned, "poisoned": poisoned},
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "total_cost_usd": cost},
                "estimate": {"total_tokens": 20, "cost_usd": cost * 2},
            }
        },
        "chapters": [
            {"chapter_id": 0, "chars": 100, "cited_doc_ids": ["a", "b"], "retries": 1, "poisoned": False, "drafted": True},
            {"chapter_id": 1, "chars": 0, "cited_doc_ids": [], "retries": 3, "poisoned": bool(poisoned), "drafted": not poisoned},
        ],
        "errors": ["x"],
        "improvement": {
            "orphans_dropped": 1,
            "extracted_nodes": 10,
            "nodes_after_orphan": 9,
            "communities_detected": 3,
            "communities_kept": 2,
            "chapters_in_toc": 2,
            "drafts": 2 - poisoned,
            "poisoned_count": poisoned,
            "cost_usd": {"phase2": 0.0, "phase4": cost, "phase2_plus_4": cost},
            "tokens": {"phase2": 0, "phase4": 15, "phase2_plus_4": 15},
        },
    }
    (d / "ledger.json").write_text(json.dumps(ledger), encoding="utf-8")


def test_ledger_collect_every_run_chapters_and_trend(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write_ledger(runs, "codex_20260916_000001", poisoned=0, cost=0.10)
    _write_ledger(runs, "codex_20260916_000002", poisoned=1, cost=0.25)
    out = ledger_collector.collect(runs)

    assert _metric(out, "runs") == 2
    assert _metric(out, "poisoned_count", run_id="codex_20260916_000002") == 1
    assert _metric(out, "phase4.usage.total_cost_usd", run_id="codex_20260916_000001") == 0.10
    assert _metric(out, "chapter.retries", subject="chapter_01", run_id="codex_20260916_000002") == 3
    assert _metric(out, "chapters.retries_total", run_id="codex_20260916_000002") == 4
    assert _check(out, "ledger.no_poison", run_id="codex_20260916_000002").passed is False
    assert _check(out, "ledger.all_toc_drafted", run_id="codex_20260916_000002").passed is False
    assert _metric(out, "delta.poisoned_count") == 1
    assert _metric(out, "delta.cost_usd_total") == pytest.approx(0.15)


def test_ledger_corrupt_file_raises_collector_unavailable(tmp_path: Path) -> None:
    bad = tmp_path / "runs" / "codex_bad"
    bad.mkdir(parents=True)
    (bad / "ledger.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ledger_collector.CollectorUnavailable):
        ledger_collector.collect(tmp_path / "runs")


# --- config -----------------------------------------------------------------


def _prompts(tmp_path: Path, synth_text: str) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "phase2_extraction.md").write_text("extract", encoding="utf-8")
    (d / "phase4_planner.md").write_text("", encoding="utf-8")
    (d / "phase4_synthesizer.md").write_text(synth_text, encoding="utf-8")
    return d


def test_config_detects_edits_empty_files_and_missing_files(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path, SYNTH_SYSTEM)
    lexicon = tmp_path / "aliases.json"
    lexicon.write_text(json.dumps({"CANON": ["a", "b"], "OTHER": "c"}), encoding="utf-8")
    out = config_collector.collect(prompts, lexicon)

    assert _metric(out, "matches_fallback", subject="prompts/phase4_synthesizer.md") is True
    assert _metric(out, "matches_fallback", subject="prompts/phase2_extraction.md") is False
    assert _check(out, "config.prompt_present", subject="prompts/phase4_planner.md").passed is False
    assert _check(out, "config.prompt_present", subject="prompts/eval_judge.md").passed is False
    assert _check(out, "config.synthesizer_prompt_keeps_citation_contract").passed is True
    assert _check(out, "config.lexicon_valid").passed is True
    assert _metric(out, "aliases") == 3


def test_config_flags_synthesizer_prompt_that_drops_citation_contract(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path, "Write a nice chapter. Cite nothing.")
    out = config_collector.collect(prompts, tmp_path / "missing.json")
    chk = _check(out, "config.synthesizer_prompt_keeps_citation_contract")
    assert chk.passed is False and chk.severity == "error"


def test_config_flags_invalid_lexicon_json(tmp_path: Path) -> None:
    prompts = _prompts(tmp_path, SYNTH_SYSTEM)
    lexicon = tmp_path / "aliases.json"
    lexicon.write_text("[1, 2", encoding="utf-8")
    out = config_collector.collect(prompts, lexicon)
    assert _check(out, "config.lexicon_valid").passed is False
