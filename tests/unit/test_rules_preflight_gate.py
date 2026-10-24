"""Unit: invalid data/rules/*.toml must stop the pipeline before any LLM call, $0.

Covers both entrypoints that can start a run: cli_compiler.main() (both
--source local and --source db dispatch through it) and
local_pipeline.run_pipeline() directly (the path tests/e2e/run_e2e_eval.py
uses without going through cli_compiler.main()).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import graph_distiller.cli_compiler as cli_compiler
import graph_distiller.local_pipeline as local_pipeline
import graph_distiller.rules as rules_mod
from graph_distiller.config import Settings

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_runtime_rules_and_settings():
    """Both graph_distiller.rules._runtime_rules and graph_distiller.config._runtime_settings are process-wide
    globals; leaving either mutated after a test (e.g. set_runtime_settings(cfg_pointing_at_a_
    temp_dir)) silently poisons every later test in the session — a real bug hit while wiring
    ADR-060 (a temp `rules_dir` with a deliberately-invalid token_budget leaked into unrelated
    verifier/topology tests). Snapshot/restore both around every test in this file."""
    import graph_distiller.config as config_mod

    prev_rules = rules_mod._runtime_rules
    prev_runtime_settings = config_mod._runtime_settings
    prev_module_settings = config_mod.settings
    rules_mod._runtime_rules = None
    yield
    rules_mod._runtime_rules = prev_rules
    config_mod._runtime_settings = prev_runtime_settings
    config_mod.settings = prev_module_settings


def test_cli_compiler_main_exits_2_on_invalid_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad_rules = tmp_path / "rules"
    bad_rules.mkdir()
    (bad_rules / "phase4_repomap.toml").write_text("token_budget = 999999\n", encoding="utf-8")

    def fake_settings_from_cli(args):
        base = Settings.from_env(models_required=False)
        cfg = replace(
            base,
            extraction_model="x",
            synthesis_model="y",
            rules_dir=bad_rules,
            errors_log_path=tmp_path / "errors.log",  # never touch the real output/errors.log
        )
        cli_compiler.set_runtime_settings(cfg)
        return cfg

    monkeypatch.setattr(cli_compiler, "settings_from_cli", fake_settings_from_cli)
    code = cli_compiler.main(
        [
            "--extraction-model", "x",
            "--synthesis-model", "y",
            "--source", "local",
        ]
    )
    assert code == 2


def test_run_pipeline_returns_1_on_invalid_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    bad_rules = tmp_path / "rules"
    bad_rules.mkdir()
    (bad_rules / "phase2_canonicalize.toml").write_text("unknown_key = 1\n", encoding="utf-8")

    base = Settings.from_env(models_required=False)
    cfg = replace(base, extraction_model="x", synthesis_model="y", rules_dir=bad_rules)
    monkeypatch.setattr(local_pipeline, "get_settings", lambda: cfg)

    code = local_pipeline.run_pipeline(auto_confirm=True)
    assert code == 1
    assert "FATAL" in capsys.readouterr().err


def test_run_pipeline_proceeds_past_gate_with_valid_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing rules dir (defaults) must not itself block the run at this gate."""
    base = Settings.from_env(models_required=False)
    cfg = replace(base, extraction_model="x", synthesis_model="y", rules_dir=tmp_path / "no_such_dir")
    monkeypatch.setattr(local_pipeline, "get_settings", lambda: cfg)
    monkeypatch.setattr(
        local_pipeline, "RunLedger", type("Boom", (), {"begin": staticmethod(lambda **_: (_ for _ in ()).throw(RuntimeError("stop after gate")))})
    )

    with pytest.raises(RuntimeError, match="stop after gate"):
        local_pipeline.run_pipeline(auto_confirm=True)
