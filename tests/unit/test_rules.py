"""Unit: rule control room loader (ADR-060) — $0."""

from __future__ import annotations

from pathlib import Path

import pytest

from graph_distiller.agents.repomap import REPOMAP_TOKEN_BUDGET
from graph_distiller.rules import (
    Rules,
    RulesValidationError,
    get_rules,
    load_rules,
    set_runtime_rules,
)

pytestmark = pytest.mark.unit


def test_hard_ceiling_constant_matches_repomap_module() -> None:
    from graph_distiller.rules import REPOMAP_HARD_TOKEN_CEILING

    assert REPOMAP_HARD_TOKEN_CEILING == REPOMAP_TOKEN_BUDGET


def test_missing_directory_falls_back_to_defaults(tmp_path: Path) -> None:
    rules = load_rules(tmp_path / "nope")
    assert rules == Rules()


def test_missing_individual_file_falls_back_for_that_file_only(tmp_path: Path) -> None:
    (tmp_path / "phase2_canonicalize.toml").write_text("jaccard_threshold = 0.5\n", encoding="utf-8")
    rules = load_rules(tmp_path)
    assert rules.canonicalize.jaccard_threshold == 0.5
    assert rules.repomap == Rules().repomap  # phase4_repomap.toml absent -> default


def test_valid_files_load_all_four(tmp_path: Path) -> None:
    (tmp_path / "phase2_canonicalize.toml").write_text(
        "min_fingerprint_len = 3\nmin_token_len = 1\njaccard_threshold = 0.7\n", encoding="utf-8"
    )
    (tmp_path / "phase4_repomap.toml").write_text(
        "top_fraction = 0.25\ngrounding_budget_fraction = 0.8\ntoken_budget = 60000\n", encoding="utf-8"
    )
    (tmp_path / "phase4_planner.toml").write_text(
        'leak_markers = ["basemodel", "custom_leak"]\nmax_title_words = 6\n', encoding="utf-8"
    )
    (tmp_path / "phase4_verifier.toml").write_text(
        'min_citations = 2\nsentence_split_pattern = "\\\\.\\\\s+"\ncheck_markdown_fences_balanced = false\n',
        encoding="utf-8",
    )
    rules = load_rules(tmp_path)
    assert rules.canonicalize.min_fingerprint_len == 3
    assert rules.repomap.token_budget == 60000
    assert rules.planner.max_title_words == 6
    assert rules.verifier.min_citations == 2
    assert rules.verifier.check_markdown_fences_balanced is False


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "phase2_canonicalize.toml").write_text("jaccrd_threshold = 0.7\n", encoding="utf-8")
    with pytest.raises(RulesValidationError):
        load_rules(tmp_path)


def test_out_of_range_value_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "phase2_canonicalize.toml").write_text("jaccard_threshold = 1.5\n", encoding="utf-8")
    with pytest.raises(RulesValidationError):
        load_rules(tmp_path)


def test_token_budget_above_hard_ceiling_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "phase4_repomap.toml").write_text("token_budget = 80001\n", encoding="utf-8")
    with pytest.raises(RulesValidationError):
        load_rules(tmp_path)


def test_invalid_regex_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "phase4_verifier.toml").write_text('sentence_split_pattern = "(unclosed"\n', encoding="utf-8")
    with pytest.raises(RulesValidationError):
        load_rules(tmp_path)


def test_malformed_toml_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "phase2_canonicalize.toml").write_text("not = [valid toml\n", encoding="utf-8")
    with pytest.raises(RulesValidationError):
        load_rules(tmp_path)


def test_get_rules_without_dir_falls_back_to_settings_rules_dir() -> None:
    import graph_distiller.rules as rules_mod

    rules_mod._runtime_rules = None
    rules = get_rules()  # mirrors config.get_settings(): no explicit dir needed
    assert isinstance(rules, Rules)


def test_set_runtime_rules_overrides_get_rules(tmp_path: Path) -> None:
    custom = load_rules(tmp_path)
    set_runtime_rules(custom)
    assert get_rules() is custom
