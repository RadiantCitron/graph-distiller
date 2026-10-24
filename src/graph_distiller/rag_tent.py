"""Phase E4: leave-no-trace tent console — STATUS → OPTIONS → PLAN → confirm.

Wires E1 status, D1 load, E3 cleanup. Does not invent new WRITE paths.
Resume/partial detection is E5; Rework here = wipe graph_distiller (+ optional load).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from graph_distiller.rag_cleanup import cleanup_with_optional_exports
from graph_distiller.rag_load import SCHEMA, count_graph_distiller_rows, load_export_dir
from graph_distiller.rag_status import (
    StatusReport,
    collect_status,
    format_status_text,
    scan_export_dir,
    status_to_dict,
)

OPTION_STATUS = "S"
OPTION_LOAD = "L"
OPTION_REWORK = "R"
OPTION_CLEANUP = "C"
OPTION_INTERRUPT = "I"
OPTION_ABORT = "X"

MENU_LINES: Sequence[str] = (
    "[S] Status only (read-only)",
    "[L] Load export → graph_distiller",
    "[R] Rework: wipe graph_distiller (schema|run) then load export",
    "[C] Cleanup: DROP SCHEMA graph_distiller | delete run",
    "[I] Interrupted work: Resume / Wipe-graph_distiller / Abort",
    "[X] Abort (no DB change)",
)


@dataclass(frozen=True)
class TentPlan:
    """What will happen if the user confirms. No secrets."""

    option: str
    title: str
    mutates_db: bool
    lines: List[str] = field(default_factory=list)
    drop_target: Optional[str] = None  # always SCHEMA when wiping
    run_id: Optional[str] = None
    export_dir: Optional[str] = None
    cleanup_mode: Optional[str] = None  # "schema" | "run" | None
    also_export_dir: bool = False
    replace_load: bool = True
    estimated: Dict[str, Any] = field(default_factory=dict)

    def format_text(self) -> str:
        bits = [
            f"[PLAN] option={self.option} title={self.title}",
            f"  mutates_db={self.mutates_db}",
        ]
        if self.drop_target:
            bits.append(f"  drop_target={self.drop_target}")
        if self.cleanup_mode:
            bits.append(f"  cleanup_mode={self.cleanup_mode}")
        if self.run_id:
            bits.append(f"  run_id={self.run_id}")
        if self.export_dir:
            bits.append(f"  export_dir={self.export_dir}")
        if self.estimated:
            bits.append(f"  estimated={self.estimated}")
        for line in self.lines:
            bits.append(f"  - {line}")
        bits.append("  Will NOT touch public.labor_law / public.labor_qna.")
        return "\n".join(bits)


def format_menu() -> str:
    return "\n".join(["[OPTIONS]", *MENU_LINES])


def _estimate_export(export_dir: Path) -> Dict[str, Any]:
    st = scan_export_dir(export_dir)
    est: Dict[str, Any] = {
        "export_status": st.status,
        "missing": list(st.missing),
    }
    if st.status != "complete":
        return est
    root = Path(export_dir)
    try:
        nodes = json.loads((root / "nodes.json").read_text(encoding="utf-8"))
        edges = json.loads((root / "edges.json").read_text(encoding="utf-8"))
        toc = json.loads((root / "toc.json").read_text(encoding="utf-8"))
        est["nodes"] = len(nodes) if isinstance(nodes, list) else None
        est["edges"] = len(edges) if isinstance(edges, list) else None
        est["chapters"] = len(toc) if isinstance(toc, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        est["read_error"] = str(exc)
    return est


def plan_status() -> TentPlan:
    return TentPlan(
        option=OPTION_STATUS,
        title="STATUS (read-only)",
        mutates_db=False,
        lines=["Refresh public COUNT + graph_distiller inventory + local exports."],
    )


def plan_abort() -> TentPlan:
    return TentPlan(
        option=OPTION_ABORT,
        title="Abort",
        mutates_db=False,
        lines=["No database or filesystem changes."],
    )


def plan_load(
    export_dir: Path | str,
    *,
    replace: bool = True,
) -> TentPlan:
    root = Path(export_dir)
    est = _estimate_export(root)
    return TentPlan(
        option=OPTION_LOAD,
        title="Load export → graph_distiller",
        mutates_db=True,
        export_dir=str(root),
        run_id=root.name,
        replace_load=replace,
        estimated=est,
        lines=[
            f"INSERT/UPSERT into schema `{SCHEMA}` from export JSON.",
            f"replace_prior_run={replace}",
        ],
    )


def plan_cleanup_schema(*, also_export_dir: bool = False) -> TentPlan:
    return TentPlan(
        option=OPTION_CLEANUP,
        title="Cleanup DROP SCHEMA",
        mutates_db=True,
        drop_target=SCHEMA,
        cleanup_mode="schema",
        also_export_dir=also_export_dir,
        lines=[
            f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE",
            "Neighbor public.* untouched.",
            f"also_export_dir={also_export_dir}",
        ],
    )


def plan_cleanup_run(
    run_id: str,
    *,
    also_export_dir: bool = False,
) -> TentPlan:
    rid = run_id.strip()
    return TentPlan(
        option=OPTION_CLEANUP,
        title="Cleanup DELETE run",
        mutates_db=True,
        drop_target=SCHEMA,
        cleanup_mode="run",
        run_id=rid,
        also_export_dir=also_export_dir,
        lines=[
            f"DELETE FROM {SCHEMA}.runs WHERE run_id={rid!r} (CASCADE children)",
            f"also_export_dir={also_export_dir}",
        ],
    )


def plan_rework(
    *,
    cleanup_mode: str,
    export_dir: Path | str,
    run_id: str | None = None,
    also_export_dir: bool = False,
) -> TentPlan:
    """Wipe tent (schema or run) then load export. Skeleton re-run is E5+/manual."""
    root = Path(export_dir)
    est = _estimate_export(root)
    mode = cleanup_mode.strip().lower()
    if mode not in ("schema", "run"):
        raise ValueError("cleanup_mode must be 'schema' or 'run'")
    rid = (run_id or root.name).strip()
    lines = [
        "Step 1: wipe graph_distiller only "
        + (f"(DROP SCHEMA {SCHEMA})" if mode == "schema" else f"(DELETE run {rid!r})"),
        f"Step 2: load_export_dir({root})",
        "Does not re-run Phase2/3 skeleton here (file export is source of truth).",
        "Hint: python cli_rag_skeleton.py --n-jo N --m-qna M -y to rebuild files.",
    ]
    return TentPlan(
        option=OPTION_REWORK,
        title="Rework wipe+load",
        mutates_db=True,
        drop_target=SCHEMA,
        cleanup_mode=mode,
        run_id=rid if mode == "run" else None,
        export_dir=str(root),
        also_export_dir=also_export_dir,
        estimated=est,
        lines=lines,
    )


def execute_plan(
    plan: TentPlan,
    *,
    uri: str,
    rebuild_dir: Path | str,
) -> Dict[str, Any]:
    """
    Execute a confirmed plan. Abort/Status do not mutate.

    Returns a JSON-serializable result dict (no URI).
    """
    out: Dict[str, Any] = {"plan": asdict(plan), "ok": True}

    if plan.option == OPTION_ABORT:
        out["action"] = "abort"
        out["db_changed"] = False
        return out

    if plan.option == OPTION_STATUS:
        report = collect_status(uri=uri, rebuild_dir=rebuild_dir, skip_db=False)
        out["action"] = "status"
        out["db_changed"] = False
        out["status"] = status_to_dict(report)
        out["status_text"] = format_status_text(report)
        return out

    if plan.option == OPTION_LOAD:
        if not plan.export_dir:
            raise ValueError("export_dir required for load")
        result = load_export_dir(
            uri, plan.export_dir, replace=plan.replace_load
        )
        out["action"] = "load"
        out["db_changed"] = True
        out["load"] = {
            "run_id": result.run_id,
            "export_dir": result.export_dir,
            "nodes": result.nodes,
            "edges": result.edges,
            "chapters": result.chapters,
            "source_map_rows": result.source_map_rows,
        }
        out["db_counts"] = count_graph_distiller_rows(uri, result.run_id)
        return out

    if plan.option == OPTION_CLEANUP:
        if plan.cleanup_mode == "schema":
            cleanup = cleanup_with_optional_exports(
                uri,
                drop_schema=True,
                also_export_dirs=plan.also_export_dir,
                rebuild_root=rebuild_dir if plan.also_export_dir else None,
            )
        elif plan.cleanup_mode == "run":
            if not plan.run_id:
                raise ValueError("run_id required for cleanup run")
            cleanup = cleanup_with_optional_exports(
                uri,
                run_id=plan.run_id,
                also_export_dirs=plan.also_export_dir,
                rebuild_root=rebuild_dir if plan.also_export_dir else None,
            )
        else:
            raise ValueError("cleanup_mode required")
        out["action"] = "cleanup"
        out["db_changed"] = True
        out["cleanup"] = {
            "action": cleanup.action,
            "schema": cleanup.schema,
            "run_id": cleanup.run_id,
            "export_dirs_removed": list(cleanup.export_dirs_removed),
            "ok": cleanup.ok,
        }
        return out

    if plan.option == OPTION_REWORK:
        if not plan.export_dir:
            raise ValueError("export_dir required for rework")
        if plan.cleanup_mode == "schema":
            cleanup = cleanup_with_optional_exports(
                uri,
                drop_schema=True,
                also_export_dirs=plan.also_export_dir,
                rebuild_root=rebuild_dir if plan.also_export_dir else None,
            )
        elif plan.cleanup_mode == "run":
            rid = plan.run_id or Path(plan.export_dir).name
            cleanup = cleanup_with_optional_exports(
                uri,
                run_id=rid,
                also_export_dirs=plan.also_export_dir,
                rebuild_root=rebuild_dir if plan.also_export_dir else None,
            )
        else:
            raise ValueError("cleanup_mode required for rework")
        result = load_export_dir(uri, plan.export_dir, replace=True)
        out["action"] = "rework"
        out["db_changed"] = True
        out["cleanup"] = {
            "action": cleanup.action,
            "schema": cleanup.schema,
            "run_id": cleanup.run_id,
            "export_dirs_removed": list(cleanup.export_dirs_removed),
            "ok": cleanup.ok,
        }
        out["load"] = {
            "run_id": result.run_id,
            "export_dir": result.export_dir,
            "nodes": result.nodes,
            "edges": result.edges,
            "chapters": result.chapters,
            "source_map_rows": result.source_map_rows,
        }
        out["db_counts"] = count_graph_distiller_rows(uri, result.run_id)
        return out

    raise ValueError(f"unknown option {plan.option!r}")


def render_opening_status(
    *,
    uri: str | None,
    rebuild_dir: Path | str,
    offline: bool = False,
) -> tuple[StatusReport, str]:
    report = collect_status(
        uri=uri, rebuild_dir=rebuild_dir, skip_db=offline or not (uri or "").strip()
    )
    text = format_status_text(report)
    # E5: append health / interrupted section (avoid circular import at module top)
    from graph_distiller.rag_resume import (
        assessments_from_status,
        format_assessments,
        interrupted_only,
        skeleton_rework_hint,
    )

    assessments = assessments_from_status(
        report, uri=uri if report.db_queried else None
    )
    text = text + "\n\n" + format_assessments(assessments)
    bad = interrupted_only(assessments)
    if bad:
        text += f"\n  interrupted_count={len(bad)}"
        text += f"\n  {skeleton_rework_hint()}"
    return report, text


ConfirmFn = Callable[[str], bool]


def default_confirm(prompt: str) -> bool:
    ans = input(prompt).strip().lower()
    return ans in ("y", "yes")
