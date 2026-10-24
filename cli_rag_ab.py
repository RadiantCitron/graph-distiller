"""CLI: Phase D2b — offline PageRank vs random A/B probe ($0).

    python cli_rag_ab.py --from output/rag_rebuild/rag_... 
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from graph_distiller.config import PROJECT_ROOT  # noqa: E402
from graph_distiller.rag_ab_probe import run_ab_probe, write_ab_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Offline RAG A/B probe (PageRank vs random)")
    p.add_argument("--from", dest="export_dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=5, help="Top-K law nodes (default 5)")
    p.add_argument("--trials", type=int, default=50)
    args = p.parse_args(argv)

    export_dir = Path(args.export_dir)
    if not export_dir.is_dir():
        print(f"[FATAL] missing export dir: {export_dir}", file=sys.stderr)
        return 2

    report = run_ab_probe(export_dir, k=args.k, random_trials=args.trials)
    out = write_ab_report(
        report,
        reports_dir=PROJECT_ROOT / "tests" / "reports",
        export_dir=export_dir,
    )
    print(json.dumps(
        {
            "k": report.k,
            "measurable_queries": report.measurable_queries,
            "mean_graph_recall": report.mean_graph_recall,
            "mean_random_recall": report.mean_random_recall,
            "queries_graph_beats_random": report.queries_graph_beats_random,
            "pass_overall": report.pass_overall,
            "report": str(out),
            "note": report.note,
        },
        ensure_ascii=False,
        indent=2,
    ))
    if report.pass_overall is False:
        return 1
    if report.pass_overall is None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
