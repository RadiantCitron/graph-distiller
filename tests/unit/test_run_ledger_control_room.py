"""Unit: RunLedger.control_room snapshot (ADR-060) — applied rules + prompt/rule file hashes, $0."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from graph_distiller.config import settings
from graph_distiller.rules import load_rules
from graph_distiller.run_ledger import RunLedger, control_room_snapshot

pytestmark = pytest.mark.unit


def test_control_room_snapshot_hashes_present_files_and_nulls_missing(tmp_path: Path) -> None:
    prompts = tmp_path / "prompts"
    rules_dir = tmp_path / "rules"
    prompts.mkdir()
    rules_dir.mkdir()
    (prompts / "phase2_extraction.md").write_text("hello", encoding="utf-8")
    (rules_dir / "phase4_repomap.toml").write_text("top_fraction = 0.3\n", encoding="utf-8")

    cfg = replace(settings, prompts_dir=prompts, rules_dir=rules_dir)
    snap = control_room_snapshot(cfg, rules=load_rules(rules_dir))

    assert snap["prompt_file_sha256"]["phase2_extraction.md"] is not None
    assert snap["prompt_file_sha256"]["phase4_planner.md"] is None  # missing -> null, not crash
    assert snap["rule_file_sha256"]["phase4_repomap.toml"] is not None
    assert snap["applied_rules"]["repomap"]["top_fraction"] == 0.3


def test_control_room_snapshot_is_deterministic_for_same_content(tmp_path: Path) -> None:
    prompts, rules_dir = tmp_path / "prompts", tmp_path / "rules"
    prompts.mkdir()
    rules_dir.mkdir()
    (prompts / "phase2_extraction.md").write_text("same content", encoding="utf-8")
    cfg = replace(settings, prompts_dir=prompts, rules_dir=rules_dir)

    a = control_room_snapshot(cfg, rules=load_rules(rules_dir))
    b = control_room_snapshot(cfg, rules=load_rules(rules_dir))
    assert a["prompt_file_sha256"] == b["prompt_file_sha256"]


def test_ledger_json_includes_control_room(tmp_path: Path) -> None:
    when = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    ledger = RunLedger.begin(
        control_room_snap={"prompts_dir": "x", "applied_rules": {"a": 1}},
        runs_root=tmp_path,
        when=when,
    )
    path = ledger.finish(0, when=when)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["control_room"] == {"prompts_dir": "x", "applied_rules": {"a": 1}}
