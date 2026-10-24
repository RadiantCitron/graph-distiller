"""Unit: lexicon bootstrap helpers — no network / $0."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from graph_distiller.schema import GraphNode

_SPEC = importlib.util.spec_from_file_location(
    "bootstrap_lexicon",
    Path(__file__).resolve().parents[1] / "tools" / "bootstrap_lexicon.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_mod)

AliasGroup = _mod.AliasGroup
filter_groups = _mod.filter_groups
groups_from_payload = _mod.groups_from_payload
known_id_index = _mod.known_id_index
lexicon_from_groups = _mod.lexicon_from_groups
merge_alias_groups = _mod.merge_alias_groups
resolve_to_known = _mod.resolve_to_known

pytestmark = pytest.mark.unit


def _node(nid: str, desc: str = "d") -> GraphNode:
    return GraphNode(id=nid, type="CONCEPT", description=desc, source_chunk_ids=["c1"])


def test_resolve_and_filter_drops_hallucinations() -> None:
    nodes = [_node("근로기준법"), _node("노동법"), _node("근기법")]
    by_norm, by_fp = known_id_index(nodes)
    assert resolve_to_known(" 근로 기준법 ", by_norm, by_fp) == "근로기준법"
    groups = [
        AliasGroup(canon="근로기준법", aliases=["노동법", "근기법", "존재하지않는법"]),
        AliasGroup(canon="FAKE_CANON", aliases=["유령"]),
    ]
    kept = filter_groups(groups, by_norm, by_fp)
    assert kept == [("근로기준법", ["노동법", "근기법"])]


def test_groups_from_payload_accepts_dict_shape() -> None:
    groups = groups_from_payload({"근로기준법": ["노동법", "근기법"]})
    assert groups[0].canon == "근로기준법"
    assert groups[0].aliases == ["노동법", "근기법"]


def test_merge_overlapping_groups() -> None:
    merged = merge_alias_groups(
        [
            ("A", ["B"]),
            ("B", ["C"]),
        ]
    )
    assert set(merged.keys()) | set(next(iter(merged.values()))) == {"A", "B", "C"}
    assert len(merged) == 1
    canon, aliases = next(iter(merged.items()))
    assert canon not in aliases
    assert len(aliases) == 2


def test_lexicon_from_groups_end_to_end() -> None:
    nodes = [
        _node("근로기준법"),
        _node("노동법"),
        _node("근기법"),
        _node("UNRELATED"),
    ]
    lex = lexicon_from_groups(
        [
            AliasGroup(canon="근로기준법", aliases=["노동법"]),
            AliasGroup(canon="노동법", aliases=["근기법"]),
            AliasGroup(canon="MADE_UP", aliases=["ALSO_FAKE"]),
        ],
        nodes,
    )
    assert "UNRELATED" not in lex
    members = set(lex) | {a for als in lex.values() for a in als}
    assert members == {"근로기준법", "노동법", "근기법"}
