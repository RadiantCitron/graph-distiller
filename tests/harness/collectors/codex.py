"""Re-verify output/codex/*.md against the ledger of the run that wrote it.

Full Part-2-exact re-verification would require re-running Phase 3 (community
detection) from the Phase 2 checkpoint and re-deriving which nodes belong to
which chapter — a parallel pipeline, which the observability placement rules
forbid. Instead this collector re-runs the same deterministic, pure checks
Verifier itself used (`detect_text_corruption`, `_markdown_broken`, citation
density, `verify_draft`) against the file **as it exists on disk now**, using
the DOC_IDs the run's own ledger recorded as valid for that chapter
(`chapters[].cited_doc_ids`) as the allowlist. This is strictly weaker than a
live Verifier pass but catches the two things an audit actually needs to
catch between runs: (a) someone hand-edited a chapter file and broke its
citations/markdown/encoding, (b) the file no longer matches what the last
successful run produced (`status=stale`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph_distiller.agents.codex_writer import _slugify
from graph_distiller.agents.verifier import detect_text_corruption, verify_draft
from graph_distiller.contracts import DOC_ID_RE
from graph_distiller.schema import ChapterDraft

from harness.collectors import Collected, CollectorUnavailable, rel

SCOPE = "codex"


def _latest_ledger_for(codex_dir: Path, runs_dir: Path) -> Optional[Dict[str, Any]]:
    """Most recent ledger whose artifacts.codex_output_dir matches codex_dir."""
    target = str(Path(codex_dir).resolve())
    best: Optional[Dict[str, Any]] = None
    for path in sorted(Path(runs_dir).glob("*/ledger.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CollectorUnavailable(f"unreadable ledger {rel(path)}: {exc}") from exc
        codex_output_dir = (data.get("artifacts") or {}).get("codex_output_dir")
        if codex_output_dir and str(Path(codex_output_dir).resolve()) == target:
            data["_path"] = rel(path)
            best = data  # globbed sorted by run_id timestamp → last write wins
    return best


def collect(codex_dir: Path, runs_dir: Path) -> Collected:
    out = Collected()
    if not out.require_input(SCOPE, codex_dir, severity="error"):
        return out

    ledger = _latest_ledger_for(codex_dir, runs_dir)
    out.check(
        "codex.matching_ledger_found",
        SCOPE,
        ledger is not None,
        severity="warn",
        message="" if ledger else "no ledger points its codex_output_dir at this directory; skipping re-verify",
    )
    if ledger is None:
        out.metric(SCOPE, "chapters_on_disk", len(list(Path(codex_dir).glob("chapter_*.md"))), unit="count")
        return out

    run_id = str(ledger.get("run_id") or Path(ledger["_path"]).parent.name)
    drafted: List[Dict[str, Any]] = [
        c for c in (ledger.get("chapters") or []) if c.get("drafted")
    ]
    disk_count = len(list(Path(codex_dir).glob("chapter_*.md")))
    stale = disk_count != len(drafted)
    out.metric(SCOPE, "chapters_expected", len(drafted), run_id=run_id, unit="count", source=ledger["_path"])
    out.metric(SCOPE, "chapters_on_disk", disk_count, run_id=run_id, unit="count")
    out.check(
        "codex.chapter_count_matches_ledger",
        SCOPE,
        not stale,
        run_id=run_id,
        severity="warn",
        message=f"ledger drafted={len(drafted)} vs on-disk={disk_count}",
        evidence=ledger["_path"],
    )

    for index, ch in enumerate(drafted, start=1):
        subject = f"chapter_{index:02d}"
        filename = f"chapter_{index:02d}_{_slugify(str(ch.get('title') or ''))}.md"
        path = Path(codex_dir) / filename
        status = "stale" if stale else "ok"

        present = path.is_file()
        out.metric(SCOPE, "file_present", present, subject=subject, run_id=run_id, status=status, source=rel(path))
        if not present:
            out.check("codex.chapter_file_present", SCOPE, False, subject=subject, run_id=run_id,
                      severity="warn", message=f"expected {filename}", evidence=rel(path))
            continue

        content = path.read_text(encoding="utf-8")
        ledger_ids = list(ch.get("cited_doc_ids") or [])
        current_ids = DOC_ID_RE.findall(content)
        out.metric(SCOPE, "citations", len(current_ids), subject=subject, run_id=run_id, status=status, source=rel(path))

        matches_recorded = sorted(set(current_ids)) == sorted(set(ledger_ids))
        out.metric(SCOPE, "matches_ledger_citations", matches_recorded, subject=subject, run_id=run_id, status=status, source=rel(path))
        out.check(
            "codex.matches_last_verified_citations",
            SCOPE,
            matches_recorded,
            subject=subject,
            run_id=run_id,
            severity="warn" if stale else "error",
            message="" if matches_recorded else f"on-disk DOC_IDs {sorted(set(current_ids))} != last verified {sorted(set(ledger_ids))}",
            evidence=rel(path),
        )

        corruption = detect_text_corruption(content)
        out.check("codex.no_text_corruption", SCOPE, not corruption, subject=subject, run_id=run_id,
                  severity="warn" if stale else "error", message="; ".join(corruption), evidence=rel(path))

        draft = ChapterDraft(
            chapter_id=int(ch.get("chapter_id", index - 1)),
            title=str(ch.get("title") or ""),
            content=content,
            cited_doc_ids=current_ids,
        )
        ok, reason = verify_draft(draft, set(ledger_ids))
        out.check(
            "codex.verify_against_last_known_valid_ids",
            SCOPE,
            ok,
            subject=subject,
            run_id=run_id,
            severity="warn" if stale else "error",
            message="ok" if ok else reason,
            evidence=rel(path),
        )
    return out
