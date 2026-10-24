"""Cross-phase contracts shared by product code and the audit harness.

These strings are wire formats, not tuning knobs: RepoMap renders them,
Synthesizer is instructed to copy them, Verifier and the harness parse them.
Changing one side alone silently breaks the others, so they live in code
(one place) and are deliberately NOT exposed in data/rules/ (ADR-060).
"""

from __future__ import annotations

import re

# --- Citation contract (RepoMap Part 2 → Synthesizer → Verifier) ---
DOC_ID_RE = re.compile(r"\[DOC_ID:\s*([^\]]+?)\]")
DOC_ID_ONLY_BLOCK_RE = re.compile(r"(?:\[DOC_ID:\s*[^\]]+\]\s*)+")
PART2_HEADER = "## Part 2: Grounding Context"


def format_doc_id(doc_id: str) -> str:
    return f"[DOC_ID: {doc_id}]"


# --- Error-line tags written to state["errors"] / errors.log ---
TAG_DLQ = "[DLQ]"
TAG_VERIFY_FAIL = "[VERIFY_FAIL]"
TAG_FATAL = "[FATAL]"
