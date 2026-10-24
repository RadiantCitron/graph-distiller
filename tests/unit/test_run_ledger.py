"""Unit: RunLedger skeleton — no network / $0."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from graph_distiller.config import settings
from graph_distiller.run_ledger import (
    RunLedger,
    create_run_directory,
    make_run_id,
    settings_public_snapshot,
)

pytestmark = pytest.mark.unit


def test_make_run_id_format() -> None:
    when = datetime(2026, 9, 15, 17, 30, 0, tzinfo=timezone.utc)
    assert make_run_id(when) == "codex_20260915_173000"


def test_create_run_directory_and_ledger_json(tmp_path) -> None:
    when = datetime(2026, 9, 15, 17, 31, 0, tzinfo=timezone.utc)
    ledger = RunLedger.begin(
        cli={"yes": True, "budget_usd": 0.25, "token_limit": 1_000_000},
        settings_snap=settings_public_snapshot(settings),
        runs_root=tmp_path,
        when=when,
    )
    assert ledger.run_dir == tmp_path / "codex_20260915_173100"
    assert ledger.run_dir.is_dir()
    ledger.sources = ["memo.txt"]
    path = ledger.finish(0, when=datetime(2026, 9, 15, 17, 32, 0, tzinfo=timezone.utc))
    assert path.name == "ledger.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["run_id"] == "codex_20260915_173100"
    assert data["exit_code"] == 0
    assert data["duration_s"] == 60.0
    assert data["cli"]["yes"] is True
    assert data["sources"] == ["memo.txt"]
    assert "openrouter_api_key" not in data["settings"]
    assert "openrouter_api_key_set" in data["settings"]
    blob = path.read_text(encoding="utf-8")
    assert "sk-" not in blob
    assert "OPENROUTER_API_KEY" not in blob


def test_create_run_directory_refuses_collision(tmp_path) -> None:
    when = datetime(2026, 9, 15, 17, 33, 0, tzinfo=timezone.utc)
    create_run_directory(tmp_path, when=when)
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, when=when)


def test_settings_snapshot_has_no_secret_value() -> None:
    snap = settings_public_snapshot(settings)
    assert "openrouter_api_key" not in snap
    assert isinstance(snap["openrouter_api_key_set"], bool)
    assert snap["extraction_model"]
    assert snap["synthesis_model"]


def test_record_phase_and_budget_slice(tmp_path) -> None:
    from graph_distiller.observability import PhaseUsage

    when = datetime(2026, 9, 15, 18, 0, 0, tzinfo=timezone.utc)
    ledger = RunLedger.begin(runs_root=tmp_path, when=when)
    ledger.record_phase(
        "phase1",
        duration_s=1.5,
        counts={"files": 1, "chunks": 10},
        persist=False,
    )
    ledger.record_phase(
        "phase2",
        duration_s=2.0,
        counts={"nodes": 5, "edges": 4},
        usage=PhaseUsage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            total_cost_usd=0.01,
        ),
        estimate={"chunk_count": 10, "cost_usd": 0.02, "model_name": "m"},
        persist=False,
    )
    ledger.record_budget_slice(
        "phase2_chunks",
        {"kept": 8, "dropped": 2, "stop_reason": "budget_usd", "secret": "no"},
        persist=False,
    )
    path = ledger.finish(0)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["phases"]["phase1"]["duration_s"] == 1.5
    assert data["phases"]["phase1"]["counts"]["chunks"] == 10
    assert data["phases"]["phase2"]["usage"]["total_tokens"] == 120
    assert data["phases"]["phase2"]["estimate"]["cost_usd"] == 0.02
    assert data["budget_slices"]["phase2_chunks"] == {
        "kept": 8,
        "dropped": 2,
        "stop_reason": "budget_usd",
    }
    assert "secret" not in data["budget_slices"]["phase2_chunks"]
    ledger.record_graph_slice(
        {
            "action": "pass",
            "stop_reason": "none",
            "nodes_before": 5,
            "nodes_after": 5,
            "secret": "no",
        },
        persist=False,
    )
    data2 = json.loads(ledger.write().read_text(encoding="utf-8"))
    assert data2["graph_slice"]["action"] == "pass"
    assert "secret" not in data2["graph_slice"]


def test_attach_run_log_tees_print(tmp_path) -> None:
    from graph_distiller.run_ledger import attach_run_log

    run_dir = tmp_path / "codex_test"
    run_dir.mkdir()
    with attach_run_log(run_dir) as log_path:
        print("hello-run-log")
        assert log_path.name == "run.log"
    text = log_path.read_text(encoding="utf-8")
    assert "hello-run-log" in text


def test_chapters_meta_and_errors_summary() -> None:
    from graph_distiller.run_ledger import chapters_meta_from_state, errors_summary
    from graph_distiller.schema import ChapterDraft, empty_state

    state = empty_state()
    state["toc"] = {0: ["A"], 1: ["B"]}
    state["chapter_titles"] = {0: "T0", 1: "T1"}
    state["drafts"] = [
        ChapterDraft(
            chapter_id=0,
            title="T0",
            content="hello world",
            cited_doc_ids=["doc"],
        )
    ]
    state["poisoned_chapters"] = [1]
    state["retry_counts"] = {1: 3}
    state["errors"] = ["short", "x" * 600]
    rows = chapters_meta_from_state(state)
    by_id = {r["chapter_id"]: r for r in rows}
    assert by_id[0]["drafted"] is True
    assert by_id[0]["chars"] == 11
    assert by_id[0]["cited_doc_ids"] == ["doc"]
    assert by_id[1]["poisoned"] is True
    assert by_id[1]["retries"] == 3
    assert by_id[1]["drafted"] is False
    errs = errors_summary(state["errors"], max_chars=20)
    assert errs[0] == "short"
    assert errs[1].endswith("...")
    assert len(errs[1]) == 20


def test_improvement_snapshot_orphan_poison_cost(tmp_path) -> None:
    from graph_distiller.observability import PhaseUsage
    from graph_distiller.run_ledger import improvement_snapshot
    from graph_distiller.schema import ChapterDraft, empty_state

    when = datetime(2026, 9, 15, 19, 0, 0, tzinfo=timezone.utc)
    ledger = RunLedger.begin(runs_root=tmp_path, when=when)
    ledger.record_phase(
        "phase2",
        duration_s=1.0,
        usage=PhaseUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            total_cost_usd=0.01,
        ),
        persist=False,
    )
    ledger.record_phase(
        "phase3",
        duration_s=0.5,
        counts={
            "extracted_nodes": 100,
            "nodes_after_orphan": 40,
            "orphans_dropped": 60,
            "communities_detected": 10,
            "communities_kept": 3,
        },
        persist=False,
    )
    ledger.record_phase(
        "phase4",
        duration_s=2.0,
        counts={"chapters_planned": 3, "drafts": 2, "poisoned": 1},
        usage=PhaseUsage(
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            total_cost_usd=0.02,
        ),
        persist=False,
    )
    state = empty_state()
    state["toc"] = {0: ["A"], 1: ["B"], 2: ["C"]}
    state["poisoned_chapters"] = [2]
    state["drafts"] = [
        ChapterDraft(chapter_id=0, title="T0", content="aa", cited_doc_ids=[]),
        ChapterDraft(chapter_id=1, title="T1", content="bb", cited_doc_ids=[]),
    ]
    state["phase3_stats"] = {
        "extracted_nodes": 100,
        "nodes_after_orphan": 40,
        "edges_after_orphan": 50,
        "orphans_dropped": 60,
        "communities_detected": 10,
        "communities_kept": 3,
        "poc_dropped": 7,
        "poc_top_chapters": 3,
        "algorithm": "louvain",
    }
    snap = improvement_snapshot(ledger.phases, state)
    assert snap["orphans_dropped"] == 60
    assert snap["poisoned_count"] == 1
    assert snap["poisoned_chapter_ids"] == [2]
    assert snap["drafts"] == 2
    assert snap["chapters_in_toc"] == 3
    assert snap["cost_usd"]["phase2_plus_4"] == pytest.approx(0.03)
    assert snap["tokens"]["phase2_plus_4"] == 45
    assert snap["phase3_stats"]["algorithm"] == "louvain"

    ledger.record_outcome_meta(state, persist=True)
    data = json.loads(ledger.ledger_path.read_text(encoding="utf-8"))
    assert data["improvement"]["orphans_dropped"] == 60
    assert data["improvement"]["poisoned_count"] == 1
    assert "content" not in json.dumps(data["improvement"])
