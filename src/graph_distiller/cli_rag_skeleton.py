"""CLI: sample RAG skeleton (Phase 1–3 wiring, no Phase 4). ADR-033~037.

Usage (repo root):

    python -m graph_distiller.cli_rag_skeleton --n-jo 5 --m-qna 10
    python -m graph_distiller.cli_rag_skeleton --counts-only
    python -m graph_distiller.cli_rag_skeleton --n-jo 5 --m-qna 10 --llm --budget-usd 0.25

Full-DB / omitting --n-jo is rejected (ADR-037).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph_distiller.config import settings as default_settings
from graph_distiller.rag_export import DEFAULT_RAG_REBUILD_ROOT
from graph_distiller.rag_gates import fatal_if_full_db_default, phase2_llm_default_enabled
from graph_distiller.rag_skeleton import preview_counts, run_rag_skeleton
from graph_distiller.run_ledger import RunLedger, settings_public_snapshot


def _resolve_uri(cli_uri: str | None) -> str:
    uri = (cli_uri or default_settings.rag_db_uri or "").strip()
    if not uri:
        print(
            "[FATAL] RAG_DB_URI unset. Add it to .env (never commit) or pass --uri.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return uri


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RAG skeleton: sample cites→graph→PageRank→export (no Phase 4)."
    )
    p.add_argument(
        "--n-jo",
        type=int,
        default=None,
        help="How many 「조」 bundles to sample (required unless --counts-only).",
    )
    p.add_argument(
        "--m-qna",
        type=int,
        default=None,
        help="How many QnA 케이스 to sample (required unless --counts-only).",
    )
    p.add_argument(
        "--counts-only",
        action="store_true",
        help="Print COUNT summary ($0) and exit. No sample default = all.",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        default=phase2_llm_default_enabled(),
        help=(
            "Phase B: run Phase 2 LLM on the sample only (costs money). "
            "Default off — full-DB LLM forbidden (ADR-037 / W3-B)."
        ),
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        default=0.25,
        help="Phase 2 hard ceiling when --llm (default 0.25, max 1.00).",
    )
    p.add_argument(
        "--uri",
        default=None,
        help="Override RAG_DB_URI for this run (prefer .env; do not log value).",
    )
    p.add_argument(
        "--export-root",
        type=Path,
        default=None,
        help="Parent dir for rag_rebuild/<run_id>/ (default Settings.rag_rebuild_dir).",
    )
    p.add_argument(
        "--no-export",
        action="store_true",
        help="Skip writing output/rag_rebuild (still prints verify JSON).",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Parent for meta-only ledger run folder.",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive confirm after COUNT preview.",
    )
    p.add_argument(
        "--load-graph_distiller",
        action="store_true",
        help="Phase D1: after export, load JSON into PostgreSQL schema graph_distiller (not public.*).",
    )
    p.add_argument(
        "--ab-probe",
        action="store_true",
        help="Phase D2b: offline PageRank vs random A/B probe on the export ($0).",
    )
    p.add_argument(
        "--synthesize",
        action="store_true",
        help="Phase D2a: run sample Phase 4 into <export>/codex/ (costs money).",
    )
    p.add_argument(
        "--max-chapters",
        type=int,
        default=2,
        help="With --synthesize: cap communities (default 2).",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="E6: leave-no-trace STATUS only (read-only; skips sample run).",
    )
    p.add_argument(
        "--cleanup-schema",
        action="store_true",
        help="E6: DROP SCHEMA graph_distiller CASCADE only (skips sample run).",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="E6: open tent console (STATUS→OPTIONS→PLAN); skips sample run.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    print(
        "[DEPRECATED] python cli_rag_skeleton.py compile path -> "
        "python cli_rag_compiler.py --source db --sample "
        "(ADR-053). Ops flags --status/--cleanup-schema/--interactive still work here; "
        "prefer cli_rag_status.py / cli_rag_cleanup.py / cli_rag_tent.py.",
        file=sys.stderr,
    )
    args = build_parser().parse_args(argv)
    cfg = default_settings

    # E6 lifecycle shortcuts (before COUNT / sample gate)
    if args.status or args.cleanup_schema or args.interactive:
        from graph_distiller.rag_cli_bridge import (
            run_cleanup_schema,
            run_interactive_tent,
            run_status,
        )

        if args.status:
            return run_status(uri=args.uri, text=True)
        if args.cleanup_schema:
            return run_cleanup_schema(uri=args.uri, yes=args.yes)
        return run_interactive_tent(["-y"] if args.yes else None)

    uri = _resolve_uri(args.uri)

    counts = preview_counts(uri)
    print("[RAG] COUNT (LLM $0):")
    print(json.dumps(counts, ensure_ascii=False, indent=2))

    if args.counts_only:
        return 0

    ops_extra = bool(args.load_graph_distiller or args.ab_probe or args.synthesize)
    if not ops_extra and args.n_jo is not None and args.m_qna is not None:
        import os

        from graph_distiller.cli_compiler import main as compiler_main

        extraction = (os.getenv("EXTRACTION_MODEL") or "").strip()
        synthesis = (os.getenv("SYNTHESIS_MODEL") or "").strip()
        if not extraction or not synthesis:
            print(
                "[FATAL] shim needs EXTRACTION_MODEL and SYNTHESIS_MODEL in env "
                "to forward as CLI model args.",
                file=sys.stderr,
            )
            return 2
        forwarded = [
            "--source",
            "db",
            "--sample",
            "--n-jo",
            str(args.n_jo),
            "--m-qna",
            str(args.m_qna),
            "--extraction-model",
            extraction,
            "--synthesis-model",
            synthesis,
            "--budget-usd",
            str(args.budget_usd),
        ]
        if args.llm:
            forwarded.append("--llm")
        if args.yes:
            forwarded.append("-y")
        if args.uri:
            forwarded.extend(["--uri", args.uri])
        if args.max_chapters:
            forwarded.extend(["--max-chapters", str(args.max_chapters)])
        print(
            "[DEPRECATED] forwarding to cli_rag_compiler.py --source db --sample",
            file=sys.stderr,
        )
        return compiler_main(forwarded)

    fatal = fatal_if_full_db_default(
        n_jo=args.n_jo, m_qna=args.m_qna, llm=bool(args.llm)
    )
    if fatal:
        print(f"[FATAL] {fatal}", file=sys.stderr)
        return 2

    if args.n_jo > counts["labor_law_jo"] or args.m_qna > counts["labor_qna_cases"]:
        print(
            f"[FATAL] Sample exceeds DB: n_jo<={counts['labor_law_jo']} "
            f"m_qna<={counts['labor_qna_cases']}.",
            file=sys.stderr,
        )
        return 2

    llm_mode = "phase2_sample" if args.llm else "cites_only"
    print(
        f"[RAG] Planned sample: n_jo={args.n_jo} m_qna={args.m_qna} "
        f"llm_mode={llm_mode}"
    )
    if not args.yes:
        ans = input("Proceed? [Y/n] ").strip().lower()
        if ans not in ("", "y", "yes"):
            print("Aborted.")
            return 1

    export_root = Path(args.export_root) if args.export_root else cfg.rag_rebuild_dir
    if args.export_root is None and not export_root:
        export_root = DEFAULT_RAG_REBUILD_ROOT

    ledger = RunLedger.begin(
        cli={
            "kind": "rag_skeleton",
            "sample_n_jo": args.n_jo,
            "sample_m_qna": args.m_qna,
            "llm_mode": llm_mode,
            "llm": bool(args.llm),
            "budget_usd": float(args.budget_usd),
            "yes": bool(args.yes),
        },
        settings_snap=settings_public_snapshot(cfg),
        runs_root=args.runs_root,
    )
    # Phase C: sample meta on the ledger spine (also mirrored in cli).
    ledger.sources = [
        f"ragdb:sample n_jo={args.n_jo} m_qna={args.m_qna}",
    ]
    ledger.record_phase(
        "counts",
        duration_s=0.0,
        counts=counts,
        extras={"sample_n_jo": args.n_jo, "sample_m_qna": args.m_qna},
    )

    result = run_rag_skeleton(
        n_jo=args.n_jo,
        m_qna=args.m_qna,
        rag_db_uri=uri,
        settings=cfg,
        llm=bool(args.llm),
        budget_usd=float(args.budget_usd),
        export=not args.no_export,
        export_root=export_root,
    )

    ledger.record_phase(
        "rag_skeleton",
        duration_s=0.0,
        counts={
            "nodes": len(result.nodes),
            "edges": len(result.edges),
            "chunks": len(result.chunks),
            **dict(result.phase3_stats),
        },
        extras={
            "llm_mode": result.llm_mode,
            "sample_n_jo": result.sample.sample_n_jo,
            "sample_m_qna": result.sample.sample_m_qna,
            "verify": result.verify,
            "phase2_usage": result.phase2_usage,
        },
    )
    arts = dict(ledger.artifacts)
    if result.export_dir:
        arts["rag_rebuild_dir"] = str(result.export_dir)
        arts["rag_rebuild_files"] = dict(result.export_paths)
    ledger.artifacts = arts
    ledger.finish(0)

    print("[RAG] verify:")
    print(json.dumps(result.verify, ensure_ascii=False, indent=2))
    if result.export_dir:
        print(f"[RAG] export: {result.export_dir}")
    print(f"[RAG] ledger: {ledger.ledger_path}")
    print(
        f"[RAG] llm_mode={result.llm_mode} "
        f"nodes={len(result.nodes)} edges={len(result.edges)}"
    )

    if args.load_graph_distiller:
        if not result.export_dir:
            print("[FATAL] --load-graph_distiller needs export (omit --no-export).", file=sys.stderr)
            ledger.finish(2)
            return 2
        from graph_distiller.rag_load import count_graph_distiller_rows, load_export_dir

        print("[RAG] Phase D1: loading into schema graph_distiller (public.* untouched)…")
        load = load_export_dir(uri, result.export_dir, replace=True)
        counts = count_graph_distiller_rows(uri, load.run_id)
        print(json.dumps({"load": load.__dict__, "db_counts": counts}, ensure_ascii=False, indent=2))
        arts = dict(ledger.artifacts)
        arts["graph_distiller_run_id"] = load.run_id
        arts["graph_distiller_counts"] = counts
        ledger.artifacts = arts
        ledger.write()

    if args.ab_probe:
        if not result.export_dir:
            print("[FATAL] --ab-probe needs export.", file=sys.stderr)
            return 2
        from graph_distiller.config import get_settings
        from graph_distiller.rag_ab_probe import run_ab_probe, write_ab_report

        print("[RAG] Phase D2b: offline A/B probe ($0)…")
        ab = run_ab_probe(result.export_dir)
        ab_path = write_ab_report(
            ab,
            reports_dir=get_settings().reports_dir,
            export_dir=Path(result.export_dir),
        )
        print(
            json.dumps(
                {
                    "pass_overall": ab.pass_overall,
                    "mean_graph_recall": ab.mean_graph_recall,
                    "mean_random_recall": ab.mean_random_recall,
                    "report": str(ab_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    if args.synthesize:
        if not result.export_dir:
            print("[FATAL] --synthesize needs export.", file=sys.stderr)
            return 2
        from graph_distiller.rag_synthesize import run_rag_synthesize

        print(
            f"[RAG] Phase D2a: synthesize max_chapters={args.max_chapters} "
            "(SYNTHESIS_MODEL)…"
        )
        syn = run_rag_synthesize(
            result.export_dir, max_chapters=max(1, int(args.max_chapters))
        )
        print(
            json.dumps(
                {
                    "codex_dir": str(syn.codex_dir),
                    "drafts": syn.drafts,
                    "poisoned": syn.poisoned,
                    "phase4_usage": syn.phase4_usage,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
