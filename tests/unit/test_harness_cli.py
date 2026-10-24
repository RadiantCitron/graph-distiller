"""Unit: audit harness CLI wiring — synthetic repo root, $0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import cli
from harness.render import json_out

pytestmark = pytest.mark.unit


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "data" / "prompts").mkdir(parents=True)
    (root / "data" / "lexicon").mkdir(parents=True)
    (root / "data" / "checkpoints").mkdir(parents=True)
    (root / "output" / "codex").mkdir(parents=True)
    (root / "output" / "runs").mkdir(parents=True)
    (root / "output").joinpath("errors.log").write_text(
        "2026-09-17 10:00:00,000 [INFO] x: hello\n", encoding="utf-8"
    )
    for name, text in {
        "phase2_extraction.md": "extract",
        "phase4_planner.md": "title",
        "phase4_synthesizer.md": "synth [DOC_ID: x] Part 2",
        "eval_judge.md": "judge",
    }.items():
        (root / "data" / "prompts" / name).write_text(text, encoding="utf-8")
    return root


def test_build_report_runs_every_collector_without_crashing(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    report = cli.build_report(root)
    assert report.collector_errors == []
    assert report.exit_code() in (0, 1)
    scopes = {m.scope for m in report.metrics}
    assert {"config", "logs"} <= scopes


def test_audit_writes_json_csv_markdown(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    report = cli.build_report(root)
    out_dir = cli.write_report(report, tmp_path / "reports")
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "metrics.csv").is_file()
    assert (out_dir / "checks.csv").is_file()
    assert (out_dir / "report.md").is_file()
    reread = json_out.read_json(out_dir / "metrics.json")
    assert reread.audit_id == report.audit_id


def test_validate_passes_on_clean_control_room(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _make_root(tmp_path)
    code = cli.main(["--root", str(root), "validate"])
    assert code == 0
    assert "control room OK" in capsys.readouterr().out


def test_validate_fails_when_synthesizer_prompt_drops_citation_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    root = _make_root(tmp_path)
    (root / "data" / "prompts" / "phase4_synthesizer.md").write_text("no citations here", encoding="utf-8")
    code = cli.main(["--root", str(root), "validate"])
    assert code == 2
    assert "FATAL" in capsys.readouterr().err


def test_compare_reports_no_diff_against_its_own_baseline(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _make_root(tmp_path)
    report = cli.build_report(root)
    out_dir = cli.write_report(report, tmp_path / "reports")
    code = cli.main(["--root", str(root), "compare", "--baseline", str(out_dir), "--current", str(out_dir)])
    assert code == 0
    assert "no metric differences" in capsys.readouterr().out


def test_compare_detects_a_changed_metric(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _make_root(tmp_path)
    baseline = cli.build_report(root)
    base_dir = cli.write_report(baseline, tmp_path / "reports")

    (root / "data" / "prompts" / "phase2_extraction.md").write_text("changed content", encoding="utf-8")
    changed = cli.build_report(root)
    changed_dir = cli.write_report(changed, tmp_path / "reports2")

    code = cli.main(["--root", str(root), "compare", "--baseline", str(base_dir), "--current", str(changed_dir)])
    assert code == 1
    out = capsys.readouterr().out
    assert "differ from baseline" in out
    assert "sha256" in out
