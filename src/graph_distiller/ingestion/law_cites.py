"""Deterministic Korean statute citation parser (ADR-037 cites-first, $0).

Extracts 「제N조」 / 「제N조의M」 references from free text. No LLM.
Callers set GraphEdge.provenance="cites" (W2-B) and/or EdgeMeta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

# 제23조, 제 23 조, 제23조의2, 제23조의 2
_CITE_RE = re.compile(r"제\s*(\d+)\s*조(?:의\s*(\d+))?")


@dataclass(frozen=True)
class JoRef:
    """One statute article reference."""

    jo: int
    sub: int | None = None  # 조의M → M; None if plain 제N조

    @property
    def node_id(self) -> str:
        return law_node_id(self.jo, self.sub)

    def label_ko(self) -> str:
        if self.sub is None:
            return f"제{self.jo}조"
        return f"제{self.jo}조의{self.sub}"


def law_node_id(jo: int, sub: int | None = None) -> str:
    """Stable ASCII node id for a 조 (avoid Hangul↔latin alias noise)."""
    if jo <= 0:
        raise ValueError(f"jo must be positive, got {jo}")
    if sub is None:
        return f"LAW_JO_{jo}"
    if sub <= 0:
        raise ValueError(f"sub must be positive when set, got {sub}")
    return f"LAW_JO_{jo}_{sub}"


def qna_node_id(case_pk: int) -> str:
    if case_pk <= 0:
        raise ValueError(f"case_pk must be positive, got {case_pk}")
    return f"QNA_{case_pk}"


def parse_jo_refs(text: str) -> List[JoRef]:
    """All 제N조(의M) hits in order of appearance (deduped, first wins)."""
    if not text:
        return []
    seen: set[Tuple[int, int | None]] = set()
    out: List[JoRef] = []
    for m in _CITE_RE.finditer(text):
        jo = int(m.group(1))
        sub = int(m.group(2)) if m.group(2) else None
        key = (jo, sub)
        if key in seen:
            continue
        seen.add(key)
        out.append(JoRef(jo=jo, sub=sub))
    return out


def primary_jo_ref_from_article_text(text: str) -> JoRef | None:
    """
    Prefer the first citation in a 「조」 row body (usually the article's own number).
    """
    refs = parse_jo_refs(text or "")
    return refs[0] if refs else None


def cite_pairs(
    text: str,
    *,
    source_node_id: str,
    exclude_self: bool = True,
) -> List[Tuple[str, str, JoRef]]:
    """
    (source_node_id, target_law_node_id, JoRef) for each cited article.

    Used for QnA→law and law→law citation edges.
    """
    pairs: List[Tuple[str, str, JoRef]] = []
    for ref in parse_jo_refs(text):
        tgt = ref.node_id
        if exclude_self and tgt == source_node_id:
            continue
        pairs.append((source_node_id, tgt, ref))
    return pairs


def count_selected_jo_hits(text: str, selected: Sequence[JoRef] | Iterable[JoRef]) -> int:
    """How many of the selected article refs appear in text (for QnA ranking)."""
    selected_ids = {r.node_id for r in selected}
    if not selected_ids:
        return 0
    return sum(1 for ref in parse_jo_refs(text) if ref.node_id in selected_ids)
