"""Rich terminal UI for the compiler (ADR-053).

cost.py stays pure — this module only renders dicts / counts.
Never print URIs, API keys, or passwords.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from rich.console import Console
from rich.table import Table

console = Console(highlight=False, soft_wrap=False, emoji=False)

BAR_COMPLETE = "#a67c52"
BAR_REMAIN = "grey50"


def render_db_status_table(
    counts: Mapping[str, Any],
    *,
    target_jo: int,
    target_qna: int,
    mode: str,
) -> Table:
    """DB totals vs compiler target. No URI."""
    table = Table(
        title="RAG DB status",
        show_header=True,
        header_style="bold",
        title_style="bold",
    )
    table.add_column("metric", style="dim")
    table.add_column("db_total", justify="right")
    table.add_column("target", justify="right")
    table.add_column("mode")

    rows = (
        ("labor_law_rows", counts.get("labor_law_rows", "?"), "-", mode),
        ("labor_law_jo", counts.get("labor_law_jo", "?"), target_jo, mode),
        ("labor_qna_rows", counts.get("labor_qna_rows", "?"), "-", mode),
        ("labor_qna_cases", counts.get("labor_qna_cases", "?"), target_qna, mode),
    )
    for metric, total, target, label in rows:
        table.add_row(str(metric), str(total), str(target), str(label))
    return table


def render_cost_table(
    *,
    est2: Optional[Mapping[str, Any]] = None,
    est4: Optional[Mapping[str, Any]] = None,
    actual2: Optional[Mapping[str, Any]] = None,
    actual4: Optional[Mapping[str, Any]] = None,
    title: str = "Token / cost dashboard",
) -> Table:
    """EST vs Actual side-by-side. Dicts come from cost.py / PhaseUsage."""
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("phase")
    table.add_column("kind", style="dim")
    table.add_column("input", justify="right")
    table.add_column("output", justify="right")
    table.add_column("total", justify="right")
    table.add_column("usd", justify="right")

    def _row(phase: str, kind: str, blob: Optional[Mapping[str, Any]]) -> None:
        if not blob:
            table.add_row(phase, kind, "-", "-", "-", "-")
            return
        inp = blob.get("input_tokens", blob.get("prompt_tokens", 0))
        out = blob.get("output_tokens", blob.get("completion_tokens", 0))
        tot = blob.get("total_tokens", int(inp or 0) + int(out or 0))
        usd = blob.get("cost_usd", blob.get("total_cost_usd", 0.0))
        table.add_row(
            phase,
            kind,
            f"{int(inp):,}",
            f"{int(out):,}",
            f"{int(tot):,}",
            f"${float(usd):.6f}",
        )

    _row("Phase 2", "EST", est2)
    _row("Phase 2", "Actual", actual2)
    _row("Phase 4", "EST", est4)
    _row("Phase 4", "Actual", actual4)
    return table


def usage_as_dict(usage: Any) -> Dict[str, Any]:
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "total_cost_usd": float(getattr(usage, "total_cost_usd", 0.0) or 0.0),
    }


def print_table(table: Table) -> None:
    console.print(table)


def print_line(message: str) -> None:
    console.print(message, overflow="ignore")


def confirm_proceed(*, auto_confirm: bool, prompt: str = "Proceed? [Y/n]") -> bool:
    if auto_confirm:
        return True
    try:
        answer = console.input(f"{prompt} ").strip().lower()
    except EOFError:
        print_line("[Pre-flight] No TTY input; aborting (use --yes for batch).")
        return False
    return answer in ("", "y", "yes")


def render_explain_tables(
    *,
    general_info: Mapping[str, tuple[str, str]],
    policy_files: Sequence[Mapping[str, str]],
    knobs: Sequence[Mapping[str, str]],
    paths: Sequence[Mapping[str, str]],
    secrets_models: Sequence[Mapping[str, str]],
) -> list[Table]:
    """Render Rich tables for --explain output. Mask secrets completely."""
    from rich.markup import escape

    tables: list[Table] = []

    # 1. General / Workspace Overview
    t_gen = Table(title="Configuration Overview", show_header=True, header_style="bold")
    t_gen.add_column("Property", style="bold")
    t_gen.add_column("Value")
    t_gen.add_column("Layer / Source", style="dim")
    for prop, (val, src) in general_info.items():
        t_gen.add_row(escape(str(prop)), escape(str(val)), escape(str(src)))
    tables.append(t_gen)

    # 2. Policy files (D5)
    t_pol = Table(title="Policy Files (D5 File-by-file)", show_header=True, header_style="bold")
    t_pol.add_column("Policy File", style="bold")
    t_pol.add_column("Layer (Source)", justify="center")
    t_pol.add_column("Status", justify="center")
    t_pol.add_column("Resolved Path", style="dim")
    for row in policy_files:
        t_pol.add_row(
            escape(str(row["file"])),
            escape(str(row["layer"])),
            escape(str(row["status"])),
            escape(str(row["path"])),
        )
    tables.append(t_pol)

    # 3. Pipeline Knobs
    t_knob = Table(title="Pipeline Knobs (19 Knobs)", show_header=True, header_style="bold")
    t_knob.add_column("Knob", style="bold")
    t_knob.add_column("Value", justify="right")
    t_knob.add_column("Layer (Source)", justify="center")
    t_knob.add_column("Note", style="dim")
    for row in knobs:
        t_knob.add_row(
            escape(str(row["knob"])),
            escape(str(row["value"])),
            escape(str(row["layer"])),
            escape(str(row.get("note", ""))),
        )
    tables.append(t_knob)

    # 4. Paths
    t_paths = Table(title="Paths", show_header=True, header_style="bold")
    t_paths.add_column("Path Name", style="bold")
    t_paths.add_column("Resolved Path")
    t_paths.add_column("Layer (Source)", style="dim", justify="center")
    for row in paths:
        t_paths.add_row(
            escape(str(row["name"])),
            escape(str(row["path"])),
            escape(str(row["layer"])),
        )
    tables.append(t_paths)

    # 5. Secrets & CLI Models
    t_sec = Table(title="Secrets & CLI Models", show_header=True, header_style="bold")
    t_sec.add_column("Item", style="bold")
    t_sec.add_column("Status / Value")
    t_sec.add_column("Source", style="dim", justify="center")
    for row in secrets_models:
        t_sec.add_row(
            escape(str(row["item"])),
            escape(str(row["status"])),
            escape(str(row["source"])),
        )
    tables.append(t_sec)

    return tables

