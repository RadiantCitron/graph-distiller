"""Unit: RunLedger workspace provenance and control room extension (Step 060, ADR-075, $0).

Verifies that:
1. `control_room_snapshot` records workspace metadata, policy file provenance (D5),
   knob values & sources, and lexicon sha256 when in workspace mode.
2. In non-workspace mode, backward compatibility is strictly maintained:
   `workspace` is None, all standard policy files and knobs are still tracked,
   and legacy keys remain unchanged.
3. No secrets (e.g. openrouter_api_key) or absolute user directories leak into
   new ledger fields (relative paths only).
4. `settings_public_snapshot` safely exports new configuration fields without secrets.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from graph_distiller.config import PROJECT_ROOT, Settings, settings
from graph_distiller.run_ledger import (
    RunLedger,
    control_room_snapshot,
    settings_public_snapshot,
)
from graph_distiller.workspace import (
    KNOB_ENV_TO_FIELD,
    LEXICON_FILENAME,
    PROMPT_FILES,
    RULE_FILES,
    resolve_workspace,
)

pytestmark = pytest.mark.unit


def _create_sample_workspace(base_dir: Path) -> Path:
    ws_dir = base_dir / "test_workspace"
    ws_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = ws_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "doc1.txt").write_text("Test corpus content", encoding="utf-8")

    profiles_dir = base_dir / "profiles" / "custom"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / "profile.toml").write_text(
        """schema_version = 1
name = "custom-profile"

[pipeline]
min_community_size = 5
""",
        encoding="utf-8",
    )
    (profiles_dir / "rules").mkdir(parents=True, exist_ok=True)
    (profiles_dir / "rules" / "phase4_repomap.toml").write_text(
        "top_fraction = 0.4\n",
        encoding="utf-8",
    )
    (profiles_dir / "lexicon").mkdir(parents=True, exist_ok=True)
    (profiles_dir / "lexicon" / "entity_aliases.json").write_text(
        '{"alias1": "canonical1"}',
        encoding="utf-8",
    )

    (ws_dir / "workspace.toml").write_text(
        """schema_version = 1
name = "my-workspace"

[corpus]
path = "corpus"

[profile]
path = "../profiles/custom"

[pipeline]
max_chunk_tokens = 500
""",
        encoding="utf-8",
    )
    return ws_dir


def test_control_room_snapshot_workspace_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for env_k in KNOB_ENV_TO_FIELD:
        monkeypatch.delenv(env_k, raising=False)

    ws_dir = _create_sample_workspace(tmp_path)
    resolved = resolve_workspace(ws_dir)

    cfg = replace(
        settings,
        workspace_dir=resolved.workspace_dir,
        raw_memos_dir=resolved.raw_memos_dir,
        checkpoint_dir=resolved.checkpoint_dir,
        codex_output_dir=resolved.codex_output_dir,
        errors_log_path=resolved.errors_log_path,
        rag_rebuild_dir=resolved.rag_rebuild_dir,
        runs_dir=resolved.runs_dir,
        prompts_dir=resolved.prompts_dir,
        rules_dir=resolved.rules_dir,
        canon_alias_lexicon_path=resolved.canon_alias_lexicon_path,
        **resolved.knobs,
    )

    snap = control_room_snapshot(cfg)

    # 1. Workspace metadata
    assert snap["workspace"] is not None
    assert snap["workspace"]["name"] == "my-workspace"
    assert snap["workspace"]["corpus_path"] == "corpus"
    assert snap["workspace"]["profile_path"] == "../profiles/custom"
    assert snap["workspace"]["profile_name"] == "custom-profile"

    # 2. Lexicon sha256
    assert snap["lexicon_file_sha256"] is not None

    # 3. Policy files provenance (D5)
    policy_files = snap["policy_files"]
    assert "phase4_repomap.toml" in policy_files
    assert policy_files["phase4_repomap.toml"]["source"] == "profile"
    assert policy_files["phase4_repomap.toml"]["sha256"] is not None

    # Verify no absolute tmp_path leaks in policy file paths
    for name, pmeta in policy_files.items():
        assert not pmeta["path"].startswith(str(tmp_path)), f"Absolute path leak in {name}: {pmeta['path']}"
        assert not pmeta["path"].startswith("/Users/"), f"Username path leak in {name}: {pmeta['path']}"

    # 4. Knobs provenance
    knobs = snap["knobs"]
    assert "max_chunk_tokens" in knobs
    assert knobs["max_chunk_tokens"]["value"] == 500
    assert knobs["max_chunk_tokens"]["source"] == "workspace"

    assert "min_community_size" in knobs
    assert knobs["min_community_size"]["value"] == 5
    assert knobs["min_community_size"]["source"] == "profile"

    # All knobs from KNOB_ENV_TO_FIELD should be tracked
    for field in KNOB_ENV_TO_FIELD.values():
        assert field in knobs, f"Missing knob field: {field}"

    # 5. Legacy keys remain present and non-empty
    assert "prompts_dir" in snap
    assert "rules_dir" in snap
    assert "prompt_file_sha256" in snap
    assert "rule_file_sha256" in snap
    assert "applied_rules" in snap


def test_control_room_snapshot_non_workspace_mode() -> None:
    snap = control_room_snapshot(settings)

    # Workspace field is None in non-workspace mode
    assert snap["workspace"] is None

    # Standard policy files are still enumerated
    for rf in RULE_FILES:
        assert rf in snap["policy_files"]
        assert snap["policy_files"][rf]["source"] in ("engine_default", "env")
    for pf in PROMPT_FILES:
        assert pf in snap["policy_files"]
        assert snap["policy_files"][pf]["source"] in ("engine_default", "env")

    # Knobs are all recorded
    for field in KNOB_ENV_TO_FIELD.values():
        assert field in snap["knobs"]
        assert snap["knobs"][field]["source"] in ("engine_default", "env")

    # Legacy keys intact
    assert "prompt_file_sha256" in snap
    assert "rule_file_sha256" in snap
    assert "applied_rules" in snap


def test_no_secrets_in_settings_or_control_room(tmp_path: Path) -> None:
    ws_dir = _create_sample_workspace(tmp_path)
    resolved = resolve_workspace(ws_dir)

    cfg = replace(
        settings,
        openrouter_api_key="sk-test-secret-key-do-not-leak",
        workspace_dir=resolved.workspace_dir,
    )

    settings_snap = settings_public_snapshot(cfg)
    control_snap = control_room_snapshot(cfg)

    # settings_public_snapshot checks
    assert "openrouter_api_key" not in settings_snap
    assert settings_snap.get("openrouter_api_key_set") is True
    assert "workspace_dir" in settings_snap
    assert "runs_dir" in settings_snap
    assert "reports_dir" in settings_snap
    assert "canon_alias_lexicon_path" in settings_snap
    assert "min_chapter_grounding_chunks" in settings_snap
    assert "canon_embed_merge" in settings_snap

    # JSON serialized checks
    settings_json = json.dumps(settings_snap)
    control_json = json.dumps(control_snap)

    assert "sk-test" not in settings_json
    assert "sk-test" not in control_json
    assert "do-not-leak" not in settings_json
    assert "do-not-leak" not in control_json


def test_run_ledger_end_to_end_workspace_serialization(tmp_path: Path) -> None:
    ws_dir = _create_sample_workspace(tmp_path)
    resolved = resolve_workspace(ws_dir)

    cfg = replace(
        settings,
        workspace_dir=resolved.workspace_dir,
        canon_alias_lexicon_path=resolved.canon_alias_lexicon_path,
        **resolved.knobs,
    )

    when = datetime(2026, 10, 5, 12, 0, 0, tzinfo=timezone.utc)
    ledger = RunLedger.begin(
        cli={"yes": True, "workspace": str(ws_dir)},
        settings_snap=settings_public_snapshot(cfg),
        control_room_snap=control_room_snapshot(cfg),
        runs_root=tmp_path / "runs",
        when=when,
    )

    path = ledger.finish(0, when=when)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["exit_code"] == 0
    assert "control_room" in data
    assert data["control_room"]["workspace"]["name"] == "my-workspace"
    assert data["control_room"]["lexicon_file_sha256"] is not None
    assert "knobs" in data["control_room"]
    assert "policy_files" in data["control_room"]
    assert "workspace_dir" in data["settings"]
