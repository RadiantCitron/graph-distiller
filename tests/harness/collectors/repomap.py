"""Compression ratio and PageRank-core retention, recomputed from checkpoints.

Runs the real `graph.build.run_phase3_networkx` (not a reimplementation) on
the Phase 1/2 checkpoints, then measures compression the same way
`tests/e2e/run_e2e_eval.py` does (same formula, deliberately duplicated —
see the note on `_collect_repomap_metrics` below).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from graph_distiller.agents.repomap import build_repo_map, core_keep_count
from graph_distiller.cost import count_tokens
from graph_distiller.graph.build import run_phase3_networkx
from graph_distiller.graph.checkpoint import load_extraction_checkpoint
from graph_distiller.ingestion.checkpoint import load_chunks_checkpoint
from graph_distiller.schema import GraphNode, GraphDistillerState, empty_state

from harness.collectors import Collected

SCOPE = "repomap"


def _core_node_ids_for_chapter(member_ids: Sequence[str], nodes: Sequence[GraphNode]) -> List[str]:
    """Same keep rule as build_repo_map / e2e.run_e2e_eval._core_node_ids_for_chapter (ADR-025 stage 3)."""
    members = set(member_ids)
    chapter_nodes = [n for n in nodes if n.id in members]
    chapter_nodes.sort(key=lambda n: n.pagerank_score, reverse=True)
    if not chapter_nodes:
        return []
    keep_n = core_keep_count(len(chapter_nodes))
    return [n.id for n in chapter_nodes[:keep_n]]


def _collect_repomap_metrics(state: GraphDistillerState) -> Dict[str, Any]:
    """
    Deliberately duplicated from tests/e2e/run_e2e_eval.py rather than imported:
    that module imports graph_distiller.local_pipeline, whose module body calls
    configure_pipeline_logging() as an import-time side effect and attaches a
    FileHandler to output/errors.log — importing it here would make every
    audit run corrupt the very log file the `logs` collector reads (a real
    bug hit while building this harness). Keep this in sync with
    `_collect_repomap_metrics` in run_e2e_eval.py if either changes.
    """
    toc = state.get("toc") or {}
    nodes = list(state.get("nodes") or [])
    chunks = list(state.get("chunks") or [])
    maps: List[str] = []
    expected_cores: List[str] = []
    for chapter_id, member_ids in sorted(toc.items()):
        members = list(member_ids)
        expected_cores.extend(_core_node_ids_for_chapter(members, nodes))
        maps.append(
            build_repo_map(chapter_id=chapter_id, member_ids=members, nodes=nodes, chunks=chunks)
        )
    combined = "\n".join(maps)
    repo_tokens = count_tokens(combined) if combined else 0
    seen: set = set()
    unique_cores: List[str] = []
    for nid in expected_cores:
        if nid not in seen:
            seen.add(nid)
            unique_cores.append(nid)
    present = [nid for nid in unique_cores if nid in combined]
    missing = [nid for nid in unique_cores if nid not in combined]
    retention_pct = (100.0 * len(present) / len(unique_cores)) if unique_cores else 100.0
    return {
        "repo_map_tokens": repo_tokens,
        "chapter_repomap_count": len(maps),
        "expected_core_nodes": unique_cores,
        "present_core_nodes": present,
        "missing_core_nodes": missing,
        "core_retention_pct": retention_pct,
    }


def _phase1_source_tokens(state: GraphDistillerState) -> int:
    chunks = list(state.get("chunks") or [])
    return sum(count_tokens(c.content or "") for c in chunks)


def compute_compression_ratio_pct(phase1_tokens: int, repo_map_tokens: int) -> float:
    """(1 - repo_map/phase1) * 100, clamped to [0, 100]. Must match run_e2e_eval.py's copy."""
    if phase1_tokens <= 0:
        return 0.0
    return max(0.0, min(100.0, (1.0 - (repo_map_tokens / phase1_tokens)) * 100.0))


def compute_diet(checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """Recompute the same 'diet' dict the (paid) judge scores, from checkpoints alone.

    Returns None when the checkpoints are missing or Phase 3 finds no chapters —
    callers (this module's collect(), and `tests/audit.py audit --judge`) treat
    that as "nothing to judge" rather than an error.
    """
    chunks_path = Path(checkpoint_dir) / "phase1_chunks.json"
    graph_path = Path(checkpoint_dir) / "phase2_extraction.json"
    if not chunks_path.is_file() or not graph_path.is_file():
        return None

    chunks = load_chunks_checkpoint(checkpoint_dir)
    nodes, edges = load_extraction_checkpoint(checkpoint_dir)

    state = empty_state()
    state["chunks"] = chunks
    state["nodes"] = nodes
    state["edges"] = edges
    state = run_phase3_networkx(state)

    toc: Dict[int, Any] = state.get("toc") or {}
    if not toc:
        return None

    phase1_tokens = _phase1_source_tokens(state)
    metrics = _collect_repomap_metrics(state)
    ratio = compute_compression_ratio_pct(phase1_tokens, metrics["repo_map_tokens"])
    return {"phase1_tokens": phase1_tokens, "compression_ratio_pct": ratio, "chapters": len(toc), **metrics}


def collect(checkpoint_dir: Path) -> Collected:
    out = Collected()
    chunks_path = Path(checkpoint_dir) / "phase1_chunks.json"
    graph_path = Path(checkpoint_dir) / "phase2_extraction.json"
    if not out.require_input(SCOPE, chunks_path) or not out.require_input(SCOPE, graph_path):
        return out

    diet = compute_diet(checkpoint_dir)
    if diet is None:
        out.metric(SCOPE, "chapters", 0, unit="count")
        return out

    out.metric(SCOPE, "chapters", diet["chapters"], unit="count")
    phase1_tokens = diet["phase1_tokens"]
    metrics = diet
    ratio = diet["compression_ratio_pct"]
    toc_len = diet["chapters"]

    out.metric(SCOPE, "phase1_tokens", phase1_tokens, unit="tokens")
    out.metric(SCOPE, "repo_map_tokens", metrics["repo_map_tokens"], unit="tokens")
    out.metric(SCOPE, "compression_ratio_pct", ratio, unit="pct")
    out.metric(SCOPE, "core_retention_pct", metrics["core_retention_pct"], unit="pct", threshold=100.0)
    out.metric(SCOPE, "expected_core_nodes", len(metrics["expected_core_nodes"]), unit="count")
    out.metric(SCOPE, "missing_core_nodes", len(metrics["missing_core_nodes"]), unit="count")

    per_chapter_budget = 80_000 * toc_len
    out.check(
        "repomap.token_budget_respected",
        SCOPE,
        metrics["repo_map_tokens"] <= per_chapter_budget,
        severity="error",
        message=f"{metrics['repo_map_tokens']} > hard ceiling {per_chapter_budget} "
        f"({toc_len} chapters x 80k)",
    )
    out.check(
        "repomap.no_missing_core_nodes",
        SCOPE,
        not metrics["missing_core_nodes"],
        severity="warn",
        message="" if not metrics["missing_core_nodes"] else f"missing: {metrics['missing_core_nodes'][:10]}",
    )
    out.check(
        "repomap.compression_positive",
        SCOPE,
        ratio > 0,
        severity="warn",
        message=f"compression_ratio_pct={ratio:.2f}",
    )
    return out
