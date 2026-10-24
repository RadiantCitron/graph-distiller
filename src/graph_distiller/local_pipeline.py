"""Local-source pipeline (`--source local`) — Phase 1 -> 2 -> 3 -> 4 -> Living Codex.

Migrated from the former top-level `src/main.py` shim (ADR-053 follow-through):
this module is the actual orchestration body that `cli_compiler.py`'s
`_run_local()` calls, and that `tests/e2e/run_e2e_eval.py` drives directly.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from graph_distiller.agents.codex_writer import save_codex
from graph_distiller.agents.workflow import run_phase4
from graph_distiller.budget import (
    DEFAULT_BUDGET_USD,
    DEFAULT_TOKEN_BUDGET,
    clamp_budget_usd,
    clamp_token_budget,
    slice_chapters_by_budget,
    slice_chunks_by_budget,
)
from graph_distiller.cli_compiler import _sum_repo_map_tokens
from graph_distiller.cli_ui import confirm_proceed, print_line, print_table, render_cost_table
from graph_distiller.config import get_settings
from graph_distiller.cost import (
    estimate_phase2_cost,
    estimate_phase4_cost,
    format_cost_status,
    merge_cost_status,
)
from graph_distiller.graph.build import run_phase3
from graph_distiller.graph.checkpoint import load_extraction_checkpoint
from graph_distiller.graph.extraction import run_phase2
from graph_distiller.graph.health import compute_graph_health
from graph_distiller.graph.scale import guard_graph_scale
from graph_distiller.ingestion.checkpoint import (
    PHASE1_CHUNKS_FILENAME,
    load_chunks_checkpoint,
)
from graph_distiller.ingestion.ingestion import SUPPORTED_SUFFIXES
from graph_distiller.ingestion.pipeline import run_phase1
from graph_distiller.observability import PhaseUsage, configure_pipeline_logging
from graph_distiller.pricing import (
    active_rate_for,
    fetch_openrouter_rate_book,
    set_active_rate_book,
)
from graph_distiller.rules import RulesValidationError, load_rules, set_runtime_rules
from graph_distiller.run_ledger import (
    DEFAULT_RUNS_ROOT,
    RunLedger,
    attach_run_log,
    control_room_snapshot,
    settings_public_snapshot,
)
from graph_distiller.schema import GraphDistillerState, Phase3Stats, empty_phase3_stats, empty_state

configure_pipeline_logging(get_settings().errors_log_path)
logger = logging.getLogger("graph_distiller.local_pipeline")


@dataclass
class PipelineOutcome:
    """Last run details for E2E / tooling (set by run_pipeline)."""

    exit_code: int
    state: Optional[GraphDistillerState] = None
    usage2: Optional[PhaseUsage] = None
    usage4: Optional[PhaseUsage] = None
    est2: Optional[Dict[str, Any]] = None
    est4: Optional[Dict[str, Any]] = None


LAST_PIPELINE_OUTCOME: Optional[PipelineOutcome] = None


def _preview(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _phase3_stats_from_state(state: GraphDistillerState) -> Phase3Stats:
    stats = empty_phase3_stats()
    raw = state.get("phase3_stats") or {}
    stats.update(raw)  # type: ignore[typeddict-item]
    return stats


def _algorithm_label(algorithm: str) -> str:
    names = {"leiden": "Leiden", "louvain": "Louvain"}
    return names.get(algorithm, algorithm)


def _print_phase3_summary(stats: Phase3Stats) -> None:
    print("\n[Phase 3] 그래프 / 커뮤니티")
    print(
        f"  추출 노드 {stats['extracted_nodes']} → "
        f"고아 제거 후 {stats['nodes_after_orphan']} "
        f"(엣지 {stats['edges_after_orphan']}, "
        f"고아 {stats['orphans_dropped']}개 제외)"
    )
    algo = _algorithm_label(stats["algorithm"])
    algo_suffix = f" ({algo})" if algo else ""
    print(f"  커뮤니티 감지 {stats['communities_detected']}개{algo_suffix}")
    if stats.get("thin_dropped"):
        print(
            f"  근거 부족 제외(ADR-057): {stats['thin_dropped']}개 "
            "(unique 원문 청크가 MIN_CHAPTER_GROUNDING_CHUNKS 미만)"
        )
    print(
        f"  PoC 캡 POC_TOP_CHAPTERS={stats['poc_top_chapters']} "
        f"({'무캡' if int(stats['poc_top_chapters'] or 0) <= 0 else 'opt-in'}) → "
        f"집필 대상 {stats['communities_kept']}/{stats['communities_detected']} "
        f"(제외 {stats['poc_dropped']}개)"
    )


def _print_ledger_improvement(ledger: RunLedger) -> None:
    """One-line ADR-030 improvement spine (meta only; full JSON in ledger.json)."""
    imp = ledger.improvement or {}
    cost = (imp.get("cost_usd") or {}).get("phase2_plus_4", 0.0)
    print(
        f"  [ledger] improvement orphans={imp.get('orphans_dropped', 0)} "
        f"communities={imp.get('communities_kept', 0)}/"
        f"{imp.get('communities_detected', 0)} "
        f"drafts={imp.get('drafts', 0)} "
        f"poisoned={imp.get('poisoned_count', 0)} "
        f"cost=${float(cost):.6f} "
        f"→ {ledger.ledger_path}"
    )


def _preflight_gate(est: Dict[str, Any], *, auto_confirm: bool, phase_label: str) -> None:
    """Rich pre-flight table + Y/n gate (ADR-053 cli_ui, shared with --source db)."""
    print_table(
        render_cost_table(
            **{"est2" if phase_label == "Phase 2" else "est4": est},
            title=f"Pre-flight Cost Check - {phase_label}",
        )
    )
    if not confirm_proceed(
        auto_confirm=auto_confirm, prompt=f"Proceed with {phase_label}? [Y/n]"
    ):
        print_line(f"[Pre-flight] {phase_label} cancelled.")
        raise SystemExit(0)


def run_pipeline(
    *,
    auto_confirm: bool = False,
    budget_usd: float = DEFAULT_BUDGET_USD,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    runs_root: Path | None = None,
    from_checkpoints: bool = False,
) -> int:
    """
    Full Phase 1->4 compile.

    auto_confirm=True (--yes/-y): confirm_proceed() never blocks on input.
    auto_confirm=False: Rich table + input; n -> exit 0.

    budget_usd / token_budget: hard ceilings (ADR-019). Clamped to
    budget.MAX_BUDGET_USD / budget.MAX_TOKEN_BUDGET. Chunks (post-Phase1) and
    chapters (post-Phase3) are sliced to fit before any LLM call — the run
    never exceeds either cap.

    runs_root: parent of per-run folder (ADR-026 --output-dir). Default
    ``output/runs``.
    """
    global LAST_PIPELINE_OUTCOME
    LAST_PIPELINE_OUTCOME = None

    budget_usd = clamp_budget_usd(budget_usd)
    token_budget = clamp_token_budget(token_budget)

    settings = get_settings()
    settings.ensure_directories()

    try:
        set_runtime_rules(load_rules(settings.rules_dir))
    except RulesValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1

    root = Path(runs_root) if runs_root is not None else settings.runs_dir
    ledger = RunLedger.begin(
        cli={
            "yes": bool(auto_confirm),
            "budget_usd": budget_usd,
            "token_limit": token_budget,
            "output_dir": str(root),
            "from_checkpoints": bool(from_checkpoints),
        },
        settings_snap=settings_public_snapshot(settings),
        control_room_snap=control_room_snapshot(settings),
        runs_root=root,
    )
    ledger.artifacts["errors_log"] = str(settings.errors_log_path)
    ledger.artifacts["codex_output_dir"] = str(settings.codex_output_dir)
    exit_code = 1
    try:
        with attach_run_log(ledger.run_dir) as run_log_path:
            ledger.artifacts["run_log"] = str(run_log_path)
            ledger.write()
            exit_code = _run_pipeline_body(
                ledger,
                auto_confirm=auto_confirm,
                budget_usd=budget_usd,
                token_budget=token_budget,
                from_checkpoints=from_checkpoints,
            )
            return exit_code
    except SystemExit as exc:
        code = exc.code
        if code is None:
            exit_code = 0
        elif isinstance(code, int):
            exit_code = code
        else:
            exit_code = 1
        raise
    finally:
        path = ledger.finish(exit_code)
        print(f"\n[Run] ledger -> {path}")
        run_log = ledger.artifacts.get("run_log")
        if run_log:
            print(f"[Run] log    -> {run_log}")


def _run_pipeline_body(
    ledger: RunLedger,
    *,
    auto_confirm: bool,
    budget_usd: float,
    token_budget: int,
    from_checkpoints: bool = False,
) -> int:
    global LAST_PIPELINE_OUTCOME
    settings = get_settings()

    print("=" * 60)
    print("GraphDistiller compile (Phase 1 -> 2 -> 3 -> 4)")
    print(f"raw_memos : {settings.raw_memos_dir}")
    print(f"codex out : {settings.codex_output_dir}")
    print(f"run dir   : {ledger.run_dir}")
    print(f"base_url  : {settings.openrouter_base_url}")
    print(f"extract   : {settings.extraction_model}")
    print(f"synth     : {settings.synthesis_model}")
    print(f"budget    : ${budget_usd:.2f} / {token_budget:,} tokens (hard ceiling, ADR-019)")
    print(
        f"poc_top   : {settings.poc_top_chapters} "
        "(0=no cap / all communities; >0=PoC top-N, ADR-055)"
    )
    print("=" * 60)

    # ADR-031: one OpenRouter /models fetch; FREE / $rates / UNKNOWN — no local table.
    rate_book = fetch_openrouter_rate_book(
        [
            settings.extraction_model,
            settings.synthesis_model,
            settings.embedding_model,
        ],
        base_url=settings.openrouter_base_url,
    )
    set_active_rate_book(rate_book)
    for line in rate_book.lines_for(
        [
            settings.extraction_model,
            settings.synthesis_model,
            settings.embedding_model,
        ]
    ):
        print(line)
    ledger.cli["pricing"] = {
        "source": rate_book.source,
        "error": rate_book.error,
        "models": {
            mid: {
                "status": rate_book.rate_for(mid).status,
                "input_usd_per_1m": rate_book.rate_for(mid).input_usd_per_1m,
                "output_usd_per_1m": rate_book.rate_for(mid).output_usd_per_1m,
            }
            for mid in (
                settings.extraction_model,
                settings.synthesis_model,
                settings.embedding_model,
            )
        },
    }
    ledger.write()

    source_files = sorted(
        p
        for p in settings.raw_memos_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    ledger.sources = [p.name for p in source_files]
    if not source_files:
        print(
            f"\n[!] `{settings.raw_memos_dir}` 에 .txt/.md/.docx/.json 파일이 없습니다.\n"
            "    예: data/raw_memos/memo_01.txt"
        )
        LAST_PIPELINE_OUTCOME = PipelineOutcome(exit_code=1)
        return 1

    print(f"\nFound {len(source_files)} source file(s):")
    for p in source_files:
        print(f"  - {p.name}")

    if not settings.openrouter_api_key:
        print("\n[!] OPENROUTER_API_KEY 미설정. `.env`를 확인하세요.")
        LAST_PIPELINE_OUTCOME = PipelineOutcome(exit_code=1)
        return 1

    state = empty_state()
    usage2: Optional[PhaseUsage] = None
    usage4: Optional[PhaseUsage] = None
    est2: Optional[Dict[str, Any]] = None
    est4: Optional[Dict[str, Any]] = None

    resumed = False
    if from_checkpoints:
        # Checkpoint resume: Phase 1/2 LLM work is already paid for and
        # persisted (ADR-049). Reload and jump straight to the Phase 3 gate.
        p1 = settings.checkpoint_dir / PHASE1_CHUNKS_FILENAME
        p2 = settings.checkpoint_dir / "phase2_extraction.json"
        if not p1.is_file() or not p2.is_file():
            print(f"\n[FATAL] --from-checkpoints but missing: {p1} / {p2}")
            LAST_PIPELINE_OUTCOME = PipelineOutcome(exit_code=1)
            return 1
        state["chunks"] = list(
            load_chunks_checkpoint(p1.parent, filename=p1.name)
        )
        nodes2, edges2 = load_extraction_checkpoint(p2.parent, filename=p2.name)
        state["nodes"] = list(nodes2)
        state["edges"] = list(edges2)
        usage2 = PhaseUsage()
        resumed = True
        print(
            f"\n[Resume] checkpoints loaded: chunks={len(state['chunks'])} "
            f"nodes={len(state['nodes'])} edges={len(state['edges'])} "
            "(Phase 1/2 LLM skipped, $0)"
        )
        ledger.record_phase(
            "phase1",
            duration_s=0.0,
            counts={"resumed_from": p1.name, "chunks": len(state["chunks"])},
        )
        ledger.record_phase(
            "phase2",
            duration_s=0.0,
            counts={
                "resumed_from": p2.name,
                "nodes": len(state["nodes"]),
                "edges": len(state["edges"]),
            },
            usage=usage2,
        )

    if not resumed:
        # --- Phase 1 ---
        print("\n[Phase 1] Semantic Chunking ...")
        t1 = time.perf_counter()
        state = run_phase1(state)
        n_files = len(state.get("raw_documents") or [])
        n_chunks = len(state.get("chunks") or [])
        print(f"총 {n_files}개 파일, {n_chunks}개 청크 생성")
        p1_ckpt = settings.checkpoint_dir / "phase1_chunks.json"
        if p1_ckpt.is_file():
            print(f"  checkpoint: {p1_ckpt.name} ({n_chunks} chunks)")
        ledger.record_phase(
            "phase1",
            duration_s=time.perf_counter() - t1,
            counts={"files": n_files, "chunks": n_chunks},
        )

        # --- Budget slice: Phase 2 input (ADR-019, never exceeds the hard ceiling) ---
        kept_chunks, slice2_info = slice_chunks_by_budget(
            list(state.get("chunks") or []),
            settings.extraction_model,
            budget_usd=budget_usd,
            token_budget=token_budget,
        )
        ledger.record_budget_slice("phase2_chunks", slice2_info)
        if slice2_info["dropped"]:
            print(
                f"[Budget] chunks {slice2_info['dropped']}개 제외 "
                f"(reason={slice2_info['stop_reason']}, "
                f"kept={slice2_info['kept']}/{n_chunks})"
            )
        state["chunks"] = kept_chunks

        # --- Pre-flight: Phase 2 ---
        est2 = estimate_phase2_cost(
            list(state.get("chunks") or []),
            settings.extraction_model,
            rate=active_rate_for(settings.extraction_model),
        )
        _preflight_gate(est2, auto_confirm=auto_confirm, phase_label="Phase 2")

        # --- Phase 2 ---
        print("\n[Phase 2] LLM Graph Extraction ...")
        t2 = time.perf_counter()
        state, usage2 = run_phase2(state)
        print(
            f"  nodes={len(state['nodes'])}  edges={len(state['edges'])}  "
            f"errors/dlq={len(state['errors'])}"
        )
        health = compute_graph_health(
            list(state.get("nodes") or []),
            list(state.get("edges") or []),
        )
        # Full by_prov dict can be huge (quote provenance) — console gets the
        # summary; the complete dict lives in ledger.json (ADR-030 drawer).
        print(
            f"  graph_health: orphan_ratio={health['orphan_ratio']:.4f} "
            f"avg_w={health['avg_edge_weight']:.4f} "
            f"by_prov_keys={list(health['avg_edge_weight_by_provenance'])}"
        )
        print(usage2.summary_line("Phase 2"))
        ckpt = settings.checkpoint_dir / "phase2_extraction.json"
        if ckpt.exists():
            payload = json.loads(ckpt.read_text(encoding="utf-8"))
            print(
                f"  checkpoint: {ckpt.name} "
                f"({len(payload.get('nodes', []))} nodes / "
                f"{len(payload.get('edges', []))} edges)"
            )
        p1_ckpt = settings.checkpoint_dir / "phase1_chunks.json"
        if p1_ckpt.is_file():
            print(f"  checkpoint: {p1_ckpt.name} (aligned with Phase 2 ids)")
        ledger.record_phase(
            "phase2",
            duration_s=time.perf_counter() - t2,
            counts={
                "nodes": len(state["nodes"]),
                "edges": len(state["edges"]),
                "errors": len(state["errors"]),
                "chunks_in": len(state.get("chunks") or []),
                "graph_health": health,
            },
            usage=usage2,
            estimate=est2,
        )

    # --- W3-A: Phase 2→3 graph scale preflight (OOM guard, $0) ---
    kept_nodes, kept_edges, graph_slice = guard_graph_scale(
        list(state.get("nodes") or []),
        list(state.get("edges") or []),
        max_nodes=settings.graph_max_nodes,
        max_edges=settings.graph_max_edges,
        max_est_mb=settings.graph_max_est_mb,
        on_exceed=(
            "slice" if settings.graph_scale_on_exceed == "slice" else "abort"
        ),
    )
    ledger.record_graph_slice(graph_slice)
    action = graph_slice.get("action")
    if action == "abort":
        print(
            f"[GraphScale] abort reason={graph_slice.get('stop_reason')} "
            f"nodes={graph_slice.get('nodes_before')} "
            f"edges={graph_slice.get('edges_before')} "
            f"est_mb={graph_slice.get('est_mb_before')} "
            f"limits={graph_slice.get('limits')} "
            f"(set GRAPH_SCALE_ON_EXCEED=slice to truncate, or raise limits)"
        )
        LAST_PIPELINE_OUTCOME = PipelineOutcome(
            exit_code=1, state=state, usage2=usage2, est2=est2
        )
        return 1
    if action == "slice":
        print(
            f"[GraphScale] slice reason={graph_slice.get('stop_reason')} "
            f"nodes {graph_slice.get('nodes_before')}→{graph_slice.get('nodes_after')} "
            f"edges {graph_slice.get('edges_before')}→{graph_slice.get('edges_after')} "
            f"est_mb {graph_slice.get('est_mb_before')}→{graph_slice.get('est_mb_after')}"
        )
        state["nodes"] = kept_nodes
        state["edges"] = kept_edges
    else:
        print(
            f"[GraphScale] pass "
            f"nodes={graph_slice.get('nodes_after')} "
            f"edges={graph_slice.get('edges_after')} "
            f"est_mb={graph_slice.get('est_mb_after')}"
        )

    # --- Phase 3 ---
    t3 = time.perf_counter()
    state = run_phase3(state)
    stats3 = _phase3_stats_from_state(state)
    _print_phase3_summary(stats3)
    ledger.record_phase(
        "phase3",
        duration_s=time.perf_counter() - t3,
        counts={
            "extracted_nodes": stats3["extracted_nodes"],
            "nodes_after_orphan": stats3["nodes_after_orphan"],
            "edges_after_orphan": stats3["edges_after_orphan"],
            "orphans_dropped": stats3["orphans_dropped"],
            "communities_detected": stats3["communities_detected"],
            "communities_kept": stats3["communities_kept"],
            "thin_dropped": stats3["thin_dropped"],
            "poc_dropped": stats3["poc_dropped"],
        },
        extras={
            "algorithm": stats3["algorithm"],
            "poc_top_chapters": stats3["poc_top_chapters"],
        },
    )

    # --- Budget slice: Phase 4 input (ADR-019, shares the same overall ceiling
    #     with Phase 2 — remaining budget = ceiling minus actual Phase 2 spend) ---
    remaining_budget_usd = max(0.0, budget_usd - usage2.total_cost_usd)
    remaining_token_budget = max(0, token_budget - usage2.total_tokens)
    kept_toc, slice4_info = slice_chapters_by_budget(
        state.get("toc") or {},
        list(state.get("nodes") or []),
        list(state.get("chunks") or []),
        synthesis_model=settings.synthesis_model,
        budget_usd=remaining_budget_usd,
        token_budget=remaining_token_budget,
    )
    ledger.record_budget_slice("phase4_chapters", slice4_info)
    if slice4_info["dropped"]:
        n_before_budget = len(state.get("toc") or {})
        print(
            f"[Budget] chapters {slice4_info['dropped']}개 제외 "
            f"(reason={slice4_info['stop_reason']}, "
            f"kept={slice4_info['kept']}/{n_before_budget}, "
            f"remaining=${remaining_budget_usd:.6f}/{remaining_token_budget:,}tok)"
        )
    state["toc"] = kept_toc

    if not state["toc"]:
        print("\n[!] TOC empty - Phase 4 skipped. Check Phase 2/3 outputs.")
        ckpt = settings.checkpoint_dir / "phase2_extraction.json"
        ledger.record_outcome_meta(
            state,
            checkpoint_path=ckpt if ckpt.exists() else None,
        )
        _print_ledger_improvement(ledger)
        LAST_PIPELINE_OUTCOME = PipelineOutcome(
            exit_code=1, state=state, usage2=usage2, est2=est2
        )
        return 1

    # --- Pre-flight: Phase 4 ---
    repo_tokens = _sum_repo_map_tokens(state)
    est4 = estimate_phase4_cost(
        n_chapters=len(state["toc"]),
        repo_map_tokens=repo_tokens,
        synthesis_model=settings.synthesis_model,
        planner_model=settings.extraction_model,
        synthesis_rate=active_rate_for(settings.synthesis_model),
        planner_rate=active_rate_for(settings.extraction_model),
    )
    _preflight_gate(est4, auto_confirm=auto_confirm, phase_label="Phase 4")

    # --- Phase 4 ---
    n_write = len(state["toc"])
    n_detected = stats3["communities_detected"]
    n_poc = stats3["communities_kept"]
    print("\n[Phase 4] LangGraph Synthesis (Planner->RepoMap->Synthesizer->Verifier) ...")
    if n_write != n_poc:
        print(
            f"  챕터 {n_write}/{n_detected} 집필 예정 "
            f"(전체 {n_detected} 중 PoC 상위 {n_poc}, "
            f"예산으로 {n_poc - n_write}개 제외)"
        )
    else:
        print(
            f"  챕터 {n_write}/{n_detected} 집필 예정 "
            f"(전체 {n_detected} 중 PoC 상위 {n_poc})"
        )
    t4 = time.perf_counter()
    state, usage4 = run_phase4(state)
    print(
        f"  drafts={len(state['drafts'])}  "
        f"poisoned={state.get('poisoned_chapters') or []}"
    )
    print(usage4.summary_line("Phase 4"))
    for draft in state["drafts"]:
        print(
            f"  DRAFT chapter={draft.chapter_id} title={draft.title!r} "
            f"citations={draft.cited_doc_ids} chars={len(draft.content)}"
        )
        print(f"         {_preview(draft.content)}")
    ledger.record_phase(
        "phase4",
        duration_s=time.perf_counter() - t4,
        counts={
            "chapters_planned": n_write,
            "drafts": len(state["drafts"]),
            "poisoned": len(state.get("poisoned_chapters") or []),
        },
        usage=usage4,
        estimate=est4,
        extras={"communities_detected": n_detected, "poc_kept": n_poc},
    )

    total_cost = usage2.total_cost_usd + usage4.total_cost_usd
    total_status = merge_cost_status(usage2.cost_status, usage4.cost_status)
    print(
        f"\n[Token totals] "
        f"Phase2+4 input={usage2.prompt_tokens + usage4.prompt_tokens}  "
        f"output={usage2.completion_tokens + usage4.completion_tokens}  "
        f"est_cost={format_cost_status(total_cost, total_status)}"
    )

    # --- Persist Living Codex ---
    print(f"\n[Output] Writing Living Codex -> {settings.codex_output_dir}")
    written, archived = save_codex(state, settings.codex_output_dir)
    if archived is not None:
        print(f"  Archived previous Codex -> {archived}")
    for path in written:
        print(f"  OK {path}")

    ckpt = settings.checkpoint_dir / "phase2_extraction.json"
    ledger.record_outcome_meta(
        state,
        codex_paths=written,
        archive_dir=archived,
        checkpoint_path=ckpt if ckpt.exists() else None,
    )
    _print_ledger_improvement(ledger)

    print("\nDone.")
    LAST_PIPELINE_OUTCOME = PipelineOutcome(
        exit_code=0,
        state=state,
        usage2=usage2,
        usage4=usage4,
        est2=est2,
        est4=est4,
    )
    return 0
