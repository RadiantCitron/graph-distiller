"""Persist Living Codex markdown under output/codex/ (SPEC Level 3.5 §2)."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

from graph_distiller.schema import ChapterDraft, GraphDistillerState


def _slugify(title: str) -> str:
    text = title.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return text or "chapter"


def _prior_codex_paths(output_dir: Path) -> List[Path]:
    """Return chapter_*.md and non-empty README.md under output_dir (not .gitkeep)."""
    if not output_dir.is_dir():
        return []
    paths: List[Path] = sorted(output_dir.glob("chapter_*.md"))
    readme = output_dir / "README.md"
    if readme.is_file() and readme.stat().st_size > 0:
        paths.append(readme)
    return paths


def _archive_existing_codex(output_dir: Path) -> Optional[Path]:
    """
    Move prior Codex markdown to output/codex_archive/run_YYYYMMDD_HHMMSS/.
    Returns archive dir, or None if there was nothing to preserve.
    """
    to_move = _prior_codex_paths(output_dir)
    if not to_move:
        return None

    archive_root = output_dir.parent / "codex_archive"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = archive_root / f"run_{stamp}"
    # Collision-safe if two saves land in the same second
    while dest.exists():
        time.sleep(0.05)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = archive_root / f"run_{stamp}"

    dest.mkdir(parents=True, exist_ok=False)
    for src in to_move:
        src.rename(dest / src.name)
    return dest


def save_codex(
    state: GraphDistillerState, output_dir: Path
) -> Tuple[List[Path], Optional[Path]]:
    """
    Archive any previous Codex, then write:
      output/codex/README.md
      output/codex/chapter_01_<slug>.md
      ...

    Returns (written_paths, archive_dir_or_none).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archived = _archive_existing_codex(output_dir)

    # Safety: clear any leftover chapter_*.md in the active dir
    for old in output_dir.glob("chapter_*.md"):
        old.unlink()

    drafts: List[ChapterDraft] = sorted(
        state.get("drafts") or [], key=lambda d: d.chapter_id
    )
    written: List[Path] = []
    toc_lines: List[str] = []

    for index, draft in enumerate(drafts, start=1):
        slug = _slugify(draft.title)
        filename = f"chapter_{index:02d}_{slug}.md"
        path = output_dir / filename
        body = draft.content.strip() + "\n"
        path.write_text(body, encoding="utf-8")
        written.append(path)
        toc_lines.append(f"{index}. [{draft.title}](./{filename})")

    poisoned = state.get("poisoned_chapters") or []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    readme = [
        "# Living Codex",
        "",
        f"- Generated: {now}",
        f"- Chapters: {len(drafts)}",
        f"- Poisoned chapters skipped: {poisoned if poisoned else 'none'}",
        "",
        "## Table of Contents",
        "",
    ]
    if toc_lines:
        readme.extend(toc_lines)
    else:
        readme.append("_No chapters were successfully synthesized._")
    readme.append("")

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(readme), encoding="utf-8")
    written.insert(0, readme_path)
    return written, archived
