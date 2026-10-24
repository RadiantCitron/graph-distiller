"""Phase D2b: offline RAG A/B probe — PageRank core vs random ($0, no LLM).

For each QnA chunk in a rag_rebuild export, measure whether top-K PageRank
law nodes recover regex-cited 제N조 ground truth better than a random baseline.
This is a wiring/compression probe for the user's later live RAG A/B — not a
chatbot and not a substitute for their online search eval.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from graph_distiller.ingestion.law_cites import parse_jo_refs
from graph_distiller.schema import ChunkMeta, GraphNode


@dataclass
class QueryProbe:
    source_doc_id: str
    gt_law_nodes: List[str]
    graph_hits: int
    random_mean_hits: float
    graph_recall: float | None
    random_mean_recall: float | None
    pass_vs_random: bool | None


@dataclass
class ABReport:
    export_dir: str
    k: int
    n_queries: int
    measurable_queries: int
    mean_graph_recall: float | None
    mean_random_recall: float | None
    queries_graph_beats_random: int
    pass_overall: bool | None
    queries: List[QueryProbe] = field(default_factory=list)
    note: str = (
        "Offline probe only. Live user RAG search/generation A/B remains manual."
    )


def _law_nodes(nodes: Sequence[GraphNode]) -> List[GraphNode]:
    return [n for n in nodes if n.id.startswith("LAW_JO_")]


def _load_export(export_dir: Path) -> Tuple[List[GraphNode], List[ChunkMeta]]:
    nodes = [
        GraphNode.model_validate(n)
        for n in json.loads((export_dir / "nodes.json").read_text(encoding="utf-8"))
    ]
    chunks = [
        ChunkMeta.model_validate(c)
        for c in json.loads((export_dir / "chunks.json").read_text(encoding="utf-8"))
    ]
    return nodes, chunks


def _qna_chunks(chunks: Sequence[ChunkMeta]) -> List[ChunkMeta]:
    return [c for c in chunks if c.source_doc_id.startswith("labor_qna_case_")]


def run_ab_probe(
    export_dir: Path | str,
    *,
    k: int = 5,
    random_trials: int = 50,
    seed: int = 42,
) -> ABReport:
    root = Path(export_dir).resolve()
    nodes, chunks = _load_export(root)
    law = _law_nodes(nodes)
    law_ids = [n.id for n in law]
    ranked = sorted(law, key=lambda n: n.pagerank_score, reverse=True)
    top_k_ids: Set[str] = {n.id for n in ranked[: max(1, k)]}

    probes: List[QueryProbe] = []
    rng = random.Random(seed)

    for ch in _qna_chunks(chunks):
        gt = []
        seen: Set[str] = set()
        for ref in parse_jo_refs(ch.content):
            if ref.node_id in {n.id for n in law} and ref.node_id not in seen:
                seen.add(ref.node_id)
                gt.append(ref.node_id)
        if not gt:
            continue

        g_hits = len(top_k_ids & set(gt))
        g_rec = g_hits / len(gt)

        r_hits_list: List[int] = []
        for _ in range(random_trials):
            if not law_ids:
                r_hits_list.append(0)
                continue
            pick = set(rng.sample(law_ids, min(k, len(law_ids))))
            r_hits_list.append(len(pick & set(gt)))
        r_mean = sum(r_hits_list) / len(r_hits_list) if r_hits_list else 0.0
        r_rec = r_mean / len(gt)

        probes.append(
            QueryProbe(
                source_doc_id=ch.source_doc_id,
                gt_law_nodes=gt,
                graph_hits=g_hits,
                random_mean_hits=r_mean,
                graph_recall=g_rec,
                random_mean_recall=r_rec,
                pass_vs_random=(g_hits >= r_mean),
            )
        )

    if not probes:
        return ABReport(
            export_dir=str(root),
            k=k,
            n_queries=len(_qna_chunks(chunks)),
            measurable_queries=0,
            mean_graph_recall=None,
            mean_random_recall=None,
            queries_graph_beats_random=0,
            pass_overall=None,
            queries=[],
            note="No QnA chunks with GT 제N조 intersecting sampled law nodes.",
        )

    mean_g = sum(p.graph_recall or 0.0 for p in probes) / len(probes)
    mean_r = sum(p.random_mean_recall or 0.0 for p in probes) / len(probes)
    beats = sum(1 for p in probes if p.pass_vs_random)
    return ABReport(
        export_dir=str(root),
        k=k,
        n_queries=len(_qna_chunks(chunks)),
        measurable_queries=len(probes),
        mean_graph_recall=mean_g,
        mean_random_recall=mean_r,
        queries_graph_beats_random=beats,
        pass_overall=(mean_g >= mean_r),
        queries=probes,
    )


def write_ab_report(
    report: ABReport,
    *,
    reports_dir: Path,
    export_dir: Path | None = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = reports_dir / f"rag_ab_probe_{stamp}.json"
    payload = asdict(report)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = reports_dir / "rag_ab_probe_latest.json"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    if export_dir is not None:
        (Path(export_dir) / "ab_probe.json").write_text(
            out.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return out
