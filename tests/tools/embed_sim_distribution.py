"""Dump pairwise embedding cosine distribution for Phase 2 checkpoint roots.

Informs ``CANON_EMBED_SIM_THRESHOLD`` upper/lower bounds (ADR-042).
Does **not** change product defaults. Does **not** merge nodes.

Candidate pairs match Reduce 2 filters: same type, not proper token-subset.
ADR-030: lives under tests/tools.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from graph_distiller.config import PROJECT_ROOT, settings as default_settings
from graph_distiller.graph.checkpoint import load_extraction_checkpoint
from graph_distiller.graph.extraction import (
    _cosine,
    _is_proper_token_subset_pair,
    _normalize_node_id,
)
from graph_distiller.schema import GraphNode

_VS = importlib.util.spec_from_file_location(
    "verify_skeleton",
    Path(__file__).resolve().parent / "verify_skeleton.py",
)
_vs = importlib.util.module_from_spec(_VS)
assert _VS and _VS.loader
_VS.loader.exec_module(_vs)
DEFAULT_CHECKPOINT = _vs.DEFAULT_CHECKPOINT

REPORTS_DIR = PROJECT_ROOT / "tests" / "reports"

# Thresholds to score "how many candidate pairs would merge"
PROBE_THRESHOLDS = (0.99, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70)


class MockEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            h = abs(hash(text.strip().lower())) % (10**9)
            out.append([((h >> (i * 3)) & 7) / 7.0 for i in range(8)])
        return out


def _percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    # nearest-rank
    k = max(0, min(len(sorted_vals) - 1, int(round((p / 100.0) * (len(sorted_vals) - 1)))))
    return float(sorted_vals[k])


def candidate_pair_scores(
    nodes: Sequence[GraphNode],
    *,
    embeddings: Any,
    max_embed_nodes: int,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Embed root ids; return cosine scores for Reduce-2-eligible pairs + meta.
    """
    by_id = {_normalize_node_id(n.id): n for n in nodes}
    roots = sorted(by_id.keys())
    meta: Dict[str, Any] = {
        "n_roots": len(roots),
        "skipped": None,
    }
    if len(roots) < 2:
        return [], meta
    if len(roots) > max_embed_nodes:
        meta["skipped"] = f"n_roots={len(roots)} > max_embed_nodes={max_embed_nodes}"
        return [], meta

    texts = [f"{nid} {by_id[nid].description}" for nid in roots]
    vectors = embeddings.embed_documents(texts)
    if len(vectors) != len(roots):
        meta["skipped"] = f"vector_count={len(vectors)} != roots={len(roots)}"
        return [], meta

    scores: List[float] = []
    n_type_skip = 0
    n_subset_skip = 0
    for i, left in enumerate(roots):
        for j in range(i + 1, len(roots)):
            right = roots[j]
            if by_id[left].type != by_id[right].type:
                n_type_skip += 1
                continue
            if _is_proper_token_subset_pair(left, right):
                n_subset_skip += 1
                continue
            scores.append(_cosine(vectors[i], vectors[j]))

    meta["n_type_skip"] = n_type_skip
    meta["n_subset_skip"] = n_subset_skip
    meta["n_candidate_pairs"] = len(scores)
    return scores, meta


def summarize_scores(scores: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(s) for s in scores)
    n = len(ordered)
    if n == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "pairs_ge_threshold": {str(t): 0 for t in PROBE_THRESHOLDS},
        }
    mean = sum(ordered) / n
    ge = {
        str(t): sum(1 for s in ordered if s >= t) for t in PROBE_THRESHOLDS
    }
    return {
        "n": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "p50": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
        "pairs_ge_threshold": ge,
    }


def build_distribution_report(
    nodes: Sequence[GraphNode],
    *,
    embeddings: Any,
    max_embed_nodes: int,
    current_threshold: float,
) -> Dict[str, Any]:
    scores, meta = candidate_pair_scores(
        nodes, embeddings=embeddings, max_embed_nodes=max_embed_nodes
    )
    summary = summarize_scores(scores)
    return {
        "meta": meta,
        "summary": summary,
        "current_threshold": current_threshold,
        "product_default_canon_embed_merge": False,
        "note": (
            "Distribution only — does not change CANON_EMBED_SIM_THRESHOLD. "
            "ADR-042: set bounds after measurement, keep merge default off."
        ),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump Reduce-2 candidate cosine distribution from checkpoint"
    )
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--mock", action="store_true", help="$0 mock embedder")
    p.add_argument("--json", type=Path, default=None)
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path: Path = args.checkpoint
    if not path.is_file():
        print(f"[FATAL] checkpoint not found: {path}", file=sys.stderr)
        return 1

    nodes, _edges = load_extraction_checkpoint(path.parent, filename=path.name)
    thr = float(default_settings.canon_embed_sim_threshold)
    max_n = int(default_settings.canon_embed_max_nodes)

    if args.mock:
        embeddings: Any = MockEmbedder()
        mode = "mock"
    else:
        from graph_distiller.adapters.openrouter import default_embedder

        embeddings = default_embedder(default_settings)
        mode = "live"

    report = build_distribution_report(
        nodes,
        embeddings=embeddings,
        max_embed_nodes=max_n,
        current_threshold=thr,
    )
    report["mode"] = mode
    report["checkpoint"] = str(path)

    s = report["summary"]
    print(f"mode={mode} roots={report['meta'].get('n_roots')} "
          f"candidates={s.get('n')} current_threshold={thr}")
    if report["meta"].get("skipped"):
        print(f"SKIPPED: {report['meta']['skipped']}")
        return 1
    print(
        f"min={s['min']:.4f} p50={s['p50']:.4f} p90={s['p90']:.4f} "
        f"p95={s['p95']:.4f} p99={s['p99']:.4f} max={s['max']:.4f} mean={s['mean']:.4f}"
    )
    print("pairs >= threshold:")
    for t in PROBE_THRESHOLDS:
        print(f"  {t:.2f}: {s['pairs_ge_threshold'][str(t)]}")

    out = args.json
    if out is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out = REPORTS_DIR / f"embed_sim_distribution_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("NOTE: CANON_EMBED_SIM_THRESHOLD / CANON_EMBED_MERGE unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
