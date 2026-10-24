"""Control-room audit: data/prompts/*.md, data/lexicon/*.json (and data/rules/*.toml once present).

Answers "what is the pipeline configured to do right now" with hashes so two
audits can tell exactly which control-room file changed between them.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from harness.collectors import Collected, rel, sha256_file

SCOPE = "config"

# prompt file → (module, attribute) holding the in-code fallback (ADR-059).
PROMPT_FALLBACKS: Dict[str, Optional[Tuple[str, str]]] = {
    "phase2_extraction.md": ("graph_distiller.graph.extraction", "EXTRACTION_SYSTEM_PROMPT"),
    "phase4_planner.md": ("graph_distiller.agents.planner", "TITLE_SYSTEM"),
    "phase4_synthesizer.md": ("graph_distiller.agents.synthesizer", "SYNTH_SYSTEM"),
    "eval_judge.md": ("harness.collectors.judge", "JUDGE_SYSTEM_FALLBACK"),
}

# Wording the downstream parser depends on. Losing it makes Verifier poison every chapter.
_SYNTH_CONTRACT_TOKENS = ("[DOC_ID:", "Part 2")


def _fallback_text(spec: Optional[Tuple[str, str]]) -> Optional[str]:
    if spec is None:
        return None
    module, attr = spec
    return getattr(importlib.import_module(module), attr)


def collect_prompts(out: Collected, prompts_dir: Path) -> None:
    if not out.require_input(SCOPE, prompts_dir, severity="error"):
        return
    for name, spec in PROMPT_FALLBACKS.items():
        path = Path(prompts_dir) / name
        subject = f"prompts/{name}"
        present = path.is_file()
        text = path.read_text(encoding="utf-8").strip() if present else ""
        out.metric(SCOPE, "present", present, subject=subject, source=rel(path))
        out.check("config.prompt_present", SCOPE, present and bool(text), subject=subject, severity="warn",
                  message="" if text else "missing or empty → pipeline uses in-code fallback", evidence=rel(path))
        if not present:
            continue
        out.metric(SCOPE, "chars", len(text), subject=subject, unit="chars", source=rel(path))
        out.metric(SCOPE, "sha256", sha256_file(path), subject=subject, source=rel(path))
        fallback = _fallback_text(spec)
        out.metric(
            SCOPE,
            "matches_fallback",
            None if fallback is None else text == fallback.strip(),
            subject=subject,
            status="unknown" if fallback is None else "ok",
            note="false = edited in the control room",
            source=rel(path),
        )

    synth = Path(prompts_dir) / "phase4_synthesizer.md"
    if synth.is_file():
        text = synth.read_text(encoding="utf-8")
        missing = [tok for tok in _SYNTH_CONTRACT_TOKENS if tok not in text]
        out.check(
            "config.synthesizer_prompt_keeps_citation_contract",
            SCOPE,
            not missing,
            subject="prompts/phase4_synthesizer.md",
            severity="error",
            message="" if not missing else f"citation contract wording removed: {missing}",
            evidence=rel(synth),
        )


def collect_lexicon(out: Collected, lexicon_path: Path) -> None:
    subject = f"lexicon/{Path(lexicon_path).name}"
    if not Path(lexicon_path).is_file():
        out.metric(SCOPE, "present", False, subject=subject, note="missing = no-op", source=rel(lexicon_path))
        return
    out.metric(SCOPE, "present", True, subject=subject, source=rel(lexicon_path))
    out.metric(SCOPE, "sha256", sha256_file(lexicon_path), subject=subject, source=rel(lexicon_path))
    try:
        raw = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        valid = isinstance(raw, dict)
        reason = "" if valid else "root must be a JSON object"
    except json.JSONDecodeError as exc:
        raw, valid, reason = {}, False, f"invalid JSON: {exc}"
    out.check("config.lexicon_valid", SCOPE, valid, subject=subject, severity="error",
              message=reason or "", evidence=rel(lexicon_path))
    if valid:
        aliases = sum(len(v) if isinstance(v, list) else (1 if isinstance(v, str) else 0) for v in raw.values())
        out.metric(SCOPE, "canon_ids", len(raw), subject=subject, unit="count", source=rel(lexicon_path))
        out.metric(SCOPE, "aliases", aliases, subject=subject, unit="count", source=rel(lexicon_path))


def collect(prompts_dir: Path, lexicon_path: Path) -> Collected:
    out = Collected()
    collect_prompts(out, prompts_dir)
    collect_lexicon(out, lexicon_path)
    return out
