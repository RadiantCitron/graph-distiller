"""Unit: embed cosine distribution dump — mock / $0."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from graph_distiller.schema import GraphNode

_SPEC = importlib.util.spec_from_file_location(
    "embed_sim_distribution",
    Path(__file__).resolve().parents[1] / "tools" / "embed_sim_distribution.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_mod)

build_distribution_report = _mod.build_distribution_report
summarize_scores = _mod.summarize_scores

pytestmark = pytest.mark.unit


class _ClusterEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            if "korea" in text.lower() or "한국" in text:
                out.append([1.0, 0.0, 0.0])
            else:
                out.append([0.0, 1.0, 0.0])
        return out


def test_summarize_scores_percentiles() -> None:
    s = summarize_scores([0.1, 0.5, 0.9, 0.95, 0.99])
    assert s["n"] == 5
    assert s["min"] == pytest.approx(0.1)
    assert s["max"] == pytest.approx(0.99)
    assert s["pairs_ge_threshold"]["0.9"] == 3


def test_distribution_skips_type_mismatch_and_counts_synonym_pair() -> None:
    nodes = [
        GraphNode(id="KOREA", type="CONCEPT", description="korea", source_chunk_ids=["a"]),
        GraphNode(id="한국", type="CONCEPT", description="korea", source_chunk_ids=["b"]),
        GraphNode(id="PERSON_X", type="PERSON", description="korea", source_chunk_ids=["c"]),
    ]
    report = build_distribution_report(
        nodes,
        embeddings=_ClusterEmbedder(),
        max_embed_nodes=100,
        current_threshold=0.90,
    )
    # Only KOREA–한국 is same-type candidate (PERSON skipped)
    assert report["summary"]["n"] == 1
    assert report["summary"]["max"] == pytest.approx(1.0)
    assert report["summary"]["pairs_ge_threshold"]["0.9"] == 1
    assert report["product_default_canon_embed_merge"] is False
