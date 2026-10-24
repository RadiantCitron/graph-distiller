"""Phase E5: interrupted-work detection + Resume / Wipe-graph_distiller choices (ADR-040).

Idempotent load = replace=True re-INSERT (same as D1). Partial DB rows vs export
expectations trigger: resume (replace load), wipe-run, wipe-schema, abort.

Never touches public.*.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from graph_distiller.rag_load import count_graph_distiller_rows
from graph_distiller.rag_status import (
    GraphDistillerInventory,
    GraphDistillerRunStatus,
    StatusReport,
    collect_status,
    scan_export_dir,
)
from graph_distiller.rag_tent import (
    TentPlan,
    plan_abort,
    plan_cleanup_run,
    plan_cleanup_schema,
    plan_load,
    plan_rework,
)

HEALTH_OK = "ok"
HEALTH_INCOMPLETE_EXPORT = "incomplete_export"
HEALTH_PARTIAL_LOAD = "partial_load"
HEALTH_MISSING_RUN = "missing_run"
HEALTH_ORPHAN_RUN = "orphan_run"
HEALTH_SCHEMA_BROKEN = "schema_broken"

RESOLVE_RESUME = "resume"
RESOLVE_WIPE_RUN = "wipe-run"
RESOLVE_WIPE_SCHEMA = "wipe-schema"
RESOLVE_ABORT = "abort"
RESOLVE_NONE = "none"

RESOLVE_CHOICES: Sequence[str] = (
    RESOLVE_RESUME,
    RESOLVE_WIPE_RUN,
    RESOLVE_WIPE_SCHEMA,
    RESOLVE_ABORT,
    RESOLVE_NONE,
)


@dataclass(frozen=True)
class ExpectedCounts:
    nodes: Optional[int]
    edges: Optional[int]
    chapters: Optional[int]
    source_map: Optional[int]


@dataclass(frozen=True)
class WorkAssessment:
    run_id: str
    export_path: str
    export_status: str
    health: str
    resume_possible: bool
    expected: Dict[str, Any]
    actual: Dict[str, Any]
    mismatches: List[str] = field(default_factory=list)
    suggested: List[str] = field(default_factory=list)
    message: str = ""


def expected_counts_from_export(export_dir: Path | str) -> ExpectedCounts:
    root = Path(export_dir)
    st = scan_export_dir(root)
    if st.status != "complete":
        return ExpectedCounts(None, None, None, None)
    try:
        nodes = json.loads((root / "nodes.json").read_text(encoding="utf-8"))
        edges = json.loads((root / "edges.json").read_text(encoding="utf-8"))
        toc = json.loads((root / "toc.json").read_text(encoding="utf-8"))
        sm = json.loads((root / "source_map.json").read_text(encoding="utf-8"))
        sm_n = 0
        if isinstance(sm, dict):
            for rows in sm.values():
                if isinstance(rows, list):
                    sm_n += len(rows)
        return ExpectedCounts(
            nodes=len(nodes) if isinstance(nodes, list) else None,
            edges=len(edges) if isinstance(edges, list) else None,
            chapters=len(toc) if isinstance(toc, dict) else None,
            source_map=sm_n,
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return ExpectedCounts(None, None, None, None)


def _run_lookup(
    inventory: Optional[GraphDistillerInventory], run_id: str
) -> Optional[GraphDistillerRunStatus]:
    if inventory is None or not inventory.schema_exists:
        return None
    for r in inventory.runs:
        if r.run_id == run_id:
            return r
    return None


def assess_export_vs_db(
    export_dir: Path | str,
    *,
    inventory: Optional[GraphDistillerInventory],
    db_counts: Optional[Dict[str, int]] = None,
) -> WorkAssessment:
    root = Path(export_dir)
    st = scan_export_dir(root)
    rid = st.run_id
    exp = expected_counts_from_export(root)
    expected = {
        "nodes": exp.nodes,
        "edges": exp.edges,
        "chapters": exp.chapters,
        "source_map": exp.source_map,
    }

    if st.status == "incomplete":
        return WorkAssessment(
            run_id=rid,
            export_path=st.path,
            export_status=st.status,
            health=HEALTH_INCOMPLETE_EXPORT,
            resume_possible=False,
            expected=expected,
            actual={},
            mismatches=[f"missing files: {st.missing}"],
            suggested=[RESOLVE_ABORT, RESOLVE_WIPE_RUN, RESOLVE_WIPE_SCHEMA],
            message=(
                "Local export incomplete  -  cannot Resume load. "
                "Fix export files, or wipe graph_distiller (run/schema) only."
            ),
        )

    if inventory is not None and inventory.schema_exists and inventory.missing_tables:
        return WorkAssessment(
            run_id=rid,
            export_path=st.path,
            export_status=st.status,
            health=HEALTH_SCHEMA_BROKEN,
            resume_possible=False,
            expected=expected,
            actual={"missing_tables": list(inventory.missing_tables)},
            mismatches=[f"missing tables: {inventory.missing_tables}"],
            suggested=[RESOLVE_WIPE_SCHEMA, RESOLVE_ABORT],
            message="graph_distiller schema incomplete  -  prefer wipe-schema then load.",
        )

    run = _run_lookup(inventory, rid)
    if run is None:
        schema_exists = bool(inventory and inventory.schema_exists)
        return WorkAssessment(
            run_id=rid,
            export_path=st.path,
            export_status=st.status,
            health=HEALTH_MISSING_RUN if schema_exists else HEALTH_OK,
            resume_possible=True,
            expected=expected,
            actual={},
            mismatches=[] if not schema_exists else ["run not in graph_distiller.runs"],
            suggested=(
                [RESOLVE_RESUME, RESOLVE_ABORT]
                if schema_exists
                else [RESOLVE_NONE, RESOLVE_RESUME]
            ),
            message=(
                "Export complete; no DB run yet  -  Load/Resume (replace) is safe."
                if schema_exists
                else "Export complete; no tent yet  -  healthy idle (Load when ready)."
            ),
        )

    actual = db_counts or {
        "nodes": run.nodes,
        "edges": run.edges,
        "chapters": run.chapters,
        "source_map": run.source_map,
    }
    mismatches: List[str] = []
    for key in ("nodes", "edges", "chapters", "source_map"):
        e_val, a_val = expected.get(key), actual.get(key)
        if e_val is None or a_val is None:
            continue
        if int(e_val) != int(a_val):
            mismatches.append(f"{key}: expected {e_val}, db {a_val}")

    if mismatches:
        return WorkAssessment(
            run_id=rid,
            export_path=st.path,
            export_status=st.status,
            health=HEALTH_PARTIAL_LOAD,
            resume_possible=True,
            expected=expected,
            actual=dict(actual),
            mismatches=mismatches,
            suggested=[
                RESOLVE_RESUME,
                RESOLVE_WIPE_RUN,
                RESOLVE_WIPE_SCHEMA,
                RESOLVE_ABORT,
            ],
            message=(
                "Partial/interrupted load detected (counts != export). "
                "Resume = idempotent replace load; Wipe = graph_distiller only."
            ),
        )

    return WorkAssessment(
        run_id=rid,
        export_path=st.path,
        export_status=st.status,
        health=HEALTH_OK,
        resume_possible=True,
        expected=expected,
        actual=dict(actual),
        mismatches=[],
        suggested=[RESOLVE_NONE],
        message="Export and DB counts match  -  idempotent re-load still safe.",
    )


def assessments_from_status(
    report: StatusReport,
    *,
    uri: str | None = None,
) -> List[WorkAssessment]:
    out: List[WorkAssessment] = []
    seen: set[str] = set()
    for exp in report.exports:
        db_counts = None
        if uri and exp.status == "complete" and report.graph_distiller:
            run = _run_lookup(report.graph_distiller, exp.run_id)
            if run is not None:
                try:
                    db_counts = count_graph_distiller_rows(uri, exp.run_id)
                except Exception:
                    db_counts = {
                        "nodes": run.nodes,
                        "edges": run.edges,
                        "chapters": run.chapters,
                        "source_map": run.source_map,
                    }
        a = assess_export_vs_db(
            exp.path, inventory=report.graph_distiller, db_counts=db_counts
        )
        out.append(a)
        seen.add(exp.run_id)

    if report.graph_distiller and report.graph_distiller.schema_exists:
        export_by_id = {e.run_id: e for e in report.exports}
        for run in report.graph_distiller.runs:
            if run.run_id in seen:
                continue
            exp = export_by_id.get(run.run_id)
            if exp is None or exp.status != "complete":
                out.append(
                    WorkAssessment(
                        run_id=run.run_id,
                        export_path=run.export_dir or "",
                        export_status="missing" if exp is None else exp.status,
                        health=HEALTH_ORPHAN_RUN,
                        resume_possible=False,
                        expected={},
                        actual={
                            "nodes": run.nodes,
                            "edges": run.edges,
                            "chapters": run.chapters,
                            "source_map": run.source_map,
                        },
                        mismatches=["no complete local export for this run"],
                        suggested=[
                            RESOLVE_WIPE_RUN,
                            RESOLVE_WIPE_SCHEMA,
                            RESOLVE_ABORT,
                        ],
                        message=(
                            "DB run without complete local export  -  "
                            "wipe graph_distiller run/schema only (cannot Resume from files)."
                        ),
                    )
                )
    return out


def assess_rebuild_tree(
    *,
    uri: str | None,
    rebuild_dir: Path | str,
    skip_db: bool = False,
) -> List[WorkAssessment]:
    report = collect_status(
        uri=uri,
        rebuild_dir=rebuild_dir,
        skip_db=skip_db or not (uri or "").strip(),
    )
    return assessments_from_status(
        report, uri=uri if report.db_queried else None
    )


def interrupted_only(assessments: Sequence[WorkAssessment]) -> List[WorkAssessment]:
    """Filter to items that need a user choice (not merely 'not loaded yet')."""
    return [
        a
        for a in assessments
        if a.health
        in (
            HEALTH_INCOMPLETE_EXPORT,
            HEALTH_PARTIAL_LOAD,
            HEALTH_ORPHAN_RUN,
            HEALTH_SCHEMA_BROKEN,
        )
    ]


def format_assessments(assessments: Sequence[WorkAssessment]) -> str:
    lines = ["== interrupted / health =="]
    if not assessments:
        lines.append("  (none)")
        return "\n".join(lines)
    for a in assessments:
        lines.append(
            f"  {a.run_id}: health={a.health} resume_possible={a.resume_possible}"
        )
        lines.append(f"    {a.message}")
        if a.mismatches:
            lines.append(f"    mismatches={a.mismatches}")
        lines.append(f"    suggested={list(a.suggested)}")
    return "\n".join(lines)


def assessment_to_dict(a: WorkAssessment) -> Dict[str, Any]:
    return asdict(a)


def plan_from_resolution(
    assessment: WorkAssessment,
    resolve: str,
    *,
    also_export_dir: bool = False,
) -> TentPlan:
    r = (resolve or "").strip().lower()
    if r not in RESOLVE_CHOICES:
        raise ValueError(f"unknown resolve {resolve!r}; choose from {RESOLVE_CHOICES}")

    if r in (RESOLVE_ABORT, RESOLVE_NONE):
        return plan_abort()

    export = assessment.export_path
    if r == RESOLVE_RESUME:
        if not assessment.resume_possible:
            raise ValueError(
                f"Resume not possible for health={assessment.health}: {assessment.message}"
            )
        if assessment.export_status != "complete":
            raise ValueError("Resume requires a complete export directory")
        plan = plan_load(export, replace=True)
        return TentPlan(
            option=plan.option,
            title="Resume (idempotent replace load)",
            mutates_db=True,
            lines=[
                *plan.lines,
                f"E5 resume for health={assessment.health}",
                "replace=True  -  same as re-running load for this run_id",
            ],
            drop_target=None,
            run_id=assessment.run_id,
            export_dir=plan.export_dir,
            cleanup_mode=None,
            also_export_dir=False,
            replace_load=True,
            estimated=plan.estimated,
        )

    if r == RESOLVE_WIPE_RUN:
        if assessment.export_status != "complete":
            return plan_cleanup_run(
                assessment.run_id, also_export_dir=also_export_dir
            )
        return plan_rework(
            cleanup_mode="run",
            export_dir=export,
            run_id=assessment.run_id,
            also_export_dir=also_export_dir,
        )

    if r == RESOLVE_WIPE_SCHEMA:
        if assessment.export_status != "complete":
            return plan_cleanup_schema(also_export_dir=also_export_dir)
        return plan_rework(
            cleanup_mode="schema",
            export_dir=export,
            run_id=assessment.run_id,
            also_export_dir=also_export_dir,
        )

    raise ValueError(f"unhandled resolve {r!r}")


def skeleton_rework_hint() -> str:
    return (
        "To rebuild export files from RAG sample, run separately: "
        "python cli_rag_skeleton.py --n-jo N --m-qna M -y "
        "(then tent Load/Rework). Phase4 not required."
    )
