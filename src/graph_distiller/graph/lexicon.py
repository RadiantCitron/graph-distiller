"""W2-D: domain entity alias lexicon (deterministic Reduce 3).

JSON shape: { "CANON_ID": ["alias1", "alias2"], ... }
Ids are normalized like Phase 2 (spaces → _, UPPER).
Missing/empty file → no-op.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence

from graph_distiller.schema import GraphEdge, GraphNode

logger = logging.getLogger(__name__)


def _normalize_node_id(raw_id: str) -> str:
    return re.sub(r"\s+", "_", raw_id.strip()).upper()


def load_entity_alias_lexicon(path: Path | None) -> Dict[str, str]:
    """
    Return alias_id → canon_id (both normalized). Canon maps to itself.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("alias lexicon load failed (%s): %s", p, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("alias lexicon root must be object: %s", p)
        return {}

    mapping: Dict[str, str] = {}
    for canon_raw, aliases in raw.items():
        canon = _normalize_node_id(str(canon_raw))
        if not canon:
            continue
        mapping[canon] = canon
        if aliases is None:
            continue
        if isinstance(aliases, str):
            alias_list: List[str] = [aliases]
        elif isinstance(aliases, list):
            alias_list = [str(a) for a in aliases]
        else:
            continue
        for alias in alias_list:
            a = _normalize_node_id(alias)
            if a:
                mapping[a] = canon
    return mapping


def rewrite_ids_with_lexicon(
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    alias_to_canon: Mapping[str, str],
) -> tuple[List[GraphNode], List[GraphEdge]]:
    """Rewrite node/edge ids through lexicon before spelling canonicalize."""
    if not alias_to_canon:
        return list(nodes), list(edges)

    new_nodes: List[GraphNode] = []
    for node in nodes:
        nid = _normalize_node_id(node.id)
        canon = alias_to_canon.get(nid, nid)
        if canon == nid:
            new_nodes.append(node)
        else:
            new_nodes.append(node.model_copy(update={"id": canon}))

    new_edges: List[GraphEdge] = []
    for edge in edges:
        src = _normalize_node_id(edge.source)
        tgt = _normalize_node_id(edge.target)
        ns = alias_to_canon.get(src, src)
        nt = alias_to_canon.get(tgt, tgt)
        if ns == src and nt == tgt:
            new_edges.append(edge)
        else:
            new_edges.append(
                edge.model_copy(update={"source": ns, "target": nt})
            )
    return new_nodes, new_edges


def lexicon_prompt_hint(alias_to_canon: Mapping[str, str]) -> str:
    """Compact preferred-id block for EXTRACTION_SYSTEM_PROMPT."""
    if not alias_to_canon:
        return ""
    groups: MutableMapping[str, List[str]] = {}
    for alias, canon in alias_to_canon.items():
        groups.setdefault(canon, [])
        if alias != canon:
            groups[canon].append(alias)
    lines = [
        "",
        "Preferred canonical node ids (use these spellings when the concept matches):",
    ]
    for canon in sorted(groups.keys()):
        aliases = sorted(set(groups[canon]))
        if aliases:
            lines.append(f"- {canon} (aliases: {', '.join(aliases)})")
        else:
            lines.append(f"- {canon}")
    return "\n".join(lines) + "\n"
