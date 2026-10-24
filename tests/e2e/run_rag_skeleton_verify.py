"""
RAG skeleton verify harness — wiring A–D on a sample (ADR-034/037).

Default: in-memory fixture ($0, no DB). Optional live:

    python tests/e2e/run_rag_skeleton_verify.py
    python tests/e2e/run_rag_skeleton_verify.py --live --n-jo 5 --m-qna 10 -y

Does not run Phase 4. Does not default to full-DB LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_distiller.graph.build import run_phase3  # noqa: E402
from graph_distiller.ingestion.rdb import (  # noqa: E402
    LawRow,
    QnaCase,
    RagCounts,
    build_cites_graph,
    build_sample_from_rows,
    sample_to_chunks,
)
from graph_distiller.rag_skeleton import verify_skeleton  # noqa: E402
from graph_distiller.schema import empty_state  # noqa: E402


def _fixture_sample():
    rows = [
        LawRow(1, None, "장", "제1장 총칙"),
        LawRow(10, 1, "조", "제10조(정의) 본조."),
        LawRow(101, 10, "항", "① 근로자란…"),
        LawRow(11, 1, "조", "제11조(적용) 제10조를 따른다."),
        LawRow(2, None, "장", "제2장 근로계약"),
        LawRow(20, 2, "조", "제20조(계약) 계약은 제10조를 본다."),
    ]
    counts = RagCounts(6, 3, 2, 2, 2, 0)
    cases = [
        QnaCase(1, None, "질의: 제10조와 제11조 관계"),
        QnaCase(2, None, "제20조 관련 회시"),
    ]
    return build_sample_from_rows(rows, cases, counts, n_jo=3, m_qna=2)


def run_fixture_verify() -> Dict[str, Any]:
    from graph_distiller.graph.build import run_phase3

    sample = _fixture_sample()
    chunks = sample_to_chunks(sample)
    nodes, edges, _meta = build_cites_graph(sample)
    state = empty_state()
    state["chunks"] = chunks
    state["nodes"] = nodes
    state["edges"] = edges
    # ADR-055: default poc_top=0 keeps all communities (no 10k hack).
    state = run_phase3(state)
    verify = verify_skeleton(
        sample,
        state["nodes"],
        state["edges"],
        chunks,
        state.get("toc") or {},
    )
    return {
        "mode": "fixture",
        "llm_mode": "cites_only",
        "sample_n_jo": sample.sample_n_jo,
        "sample_m_qna": sample.sample_m_qna,
        "nodes": len(state["nodes"]),
        "edges": len(state["edges"]),
        "phase3_stats": dict(state.get("phase3_stats") or {}),
        "verify": verify,
    }


def run_live_verify(n_jo: int, m_qna: int, yes: bool) -> Dict[str, Any]:
    from graph_distiller.config import settings
    from graph_distiller.rag_skeleton import run_rag_skeleton

    uri = (settings.rag_db_uri or "").strip()
    if not uri:
        raise SystemExit("[FATAL] RAG_DB_URI required for --live")
    if not yes:
        print(f"Live sample n_jo={n_jo} m_qna={m_qna}. Pass -y to confirm.")
        raise SystemExit(1)
    result = run_rag_skeleton(
        n_jo=n_jo,
        m_qna=m_qna,
        rag_db_uri=uri,
        llm=False,
        export=True,
    )
    return {
        "mode": "live",
        "llm_mode": result.llm_mode,
        "sample_n_jo": result.sample.sample_n_jo,
        "sample_m_qna": result.sample.sample_m_qna,
        "nodes": len(result.nodes),
        "edges": len(result.edges),
        "phase3_stats": result.phase3_stats,
        "verify": result.verify,
        "export_dir": str(result.export_dir) if result.export_dir else None,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true")
    p.add_argument("--n-jo", type=int, default=5)
    p.add_argument("--m-qna", type=int, default=10)
    p.add_argument("-y", "--yes", action="store_true")
    args = p.parse_args(argv)

    if args.live:
        report = run_live_verify(args.n_jo, args.m_qna, args.yes)
    else:
        report = run_fixture_verify()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"rag_skeleton_verify_{stamp}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = REPORTS_DIR / "rag_skeleton_verify_latest.json"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[report] {out}")
    v = report["verify"]
    ok_b = v["B_citation_survival"].get("pass")
    ok_c = v["C_pagerank_vs_random"].get("pass")
    ok_d = v["D_token_diet"].get("pass")
    if ok_b is False or ok_d is False:
        return 1
    if ok_c is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
