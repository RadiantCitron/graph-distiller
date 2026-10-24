"""Bootstrap Reduce 3 lexicon from a Phase 2 checkpoint (LLM, 1-shot).

Reads node id+description, asks a chat model to cluster synonymous concepts,
filters hallucinated ids against the checkpoint, writes
``data/lexicon/entity_aliases.json``.

ADR-030: executable lives under tests/tools. Product defaults unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.messages import HumanMessage, SystemMessage

from graph_distiller.config import PROJECT_ROOT, settings as default_settings
from graph_distiller.cost import (
    calculate_cost,
    count_tokens,
    format_cost_status,
)
from graph_distiller.graph.checkpoint import load_extraction_checkpoint
from graph_distiller.pricing import fetch_openrouter_rate_book
from graph_distiller.schema import GraphNode

DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "checkpoints" / "phase2_extraction.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "lexicon" / "entity_aliases.json"
DEFAULT_BATCH_SIZE = 80
DEFAULT_OUTPUT_TOKENS_PER_BATCH = 1200
DESC_CHARS = 180

SYSTEM_PROMPT = """\
You cluster knowledge-graph entity nodes that mean the SAME real-world concept.

Rules:
1. Only group nodes that a domain expert would treat as one term (synonym,
   abbreviation, spelling variant, translation of the same concept).
2. NEVER group antonyms or opposing concepts (e.g. 자기계발/자기파괴,
   장점/단점, gain/loss, success/failure). Opposite meaning is NOT the
   same concept.
3. NEVER group hypernym/hyponym pairs (e.g. 과일/사과, framework/tool,
   disease/diabetes). Group ONLY exact synonyms, abbreviations, and
   spelling/translation variants of the SAME concept.
4. Do NOT group related-but-distinct concepts (e.g. a law vs a ministry that
   enforces it; a person vs their role; PROGRAMMER vs PROGRAMMING;
   WRITER vs WRITING; TALENT vs MUSICAL_TALENT).
   An alias must be INTERCHANGEABLE with canon in any sentence without
   changing meaning. "질문력" and "이력서" are not interchangeable.
   When in doubt, do NOT group.
5. canon MUST be copied from the given id list (prefer the most official /
   complete spelling). aliases are the other ids in that group.
6. Skip singleton nodes. Return only groups with at least one alias.
7. Never invent ids that are not in the input list.
8. Prefer precision over recall. A typical batch has 5-25 groups, not dozens.
   Do not force weak associations.
"""


class AliasGroup(BaseModel):
    canon: str = Field(description="Representative node id copied from the list")
    aliases: List[str] = Field(
        default_factory=list,
        description="Other node ids that mean the same concept",
    )


class LexiconGroups(BaseModel):
    groups: List[AliasGroup] = Field(default_factory=list)


AliasGroup.model_rebuild()
LexiconGroups.model_rebuild()


def _normalize_node_id(raw_id: str) -> str:
    return re.sub(r"\s+", "_", (raw_id or "").strip()).upper()


def _fingerprint(node_id: str) -> str:
    return re.sub(r"[^A-Z0-9가-힣]", "", _normalize_node_id(node_id))


def known_id_index(nodes: Sequence[GraphNode]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (normalized_id → stored id, fingerprint → stored id)."""
    by_norm: Dict[str, str] = {}
    by_fp: Dict[str, str] = {}
    for node in nodes:
        stored = node.id
        norm = _normalize_node_id(stored)
        by_norm[norm] = stored
        fp = _fingerprint(stored)
        if fp and fp not in by_fp:
            by_fp[fp] = stored
    return by_norm, by_fp


def resolve_to_known(
    raw: str,
    by_norm: Mapping[str, str],
    by_fp: Mapping[str, str],
) -> Optional[str]:
    """Map an LLM-emitted id onto a checkpoint id, or None if hallucinated."""
    norm = _normalize_node_id(raw)
    if not norm:
        return None
    if norm in by_norm:
        return by_norm[norm]
    fp = _fingerprint(norm)
    if fp and fp in by_fp:
        return by_fp[fp]
    return None


def catalog_rows(nodes: Sequence[GraphNode]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for node in nodes:
        desc = (node.description or "").strip().replace("\n", " ")
        if len(desc) > DESC_CHARS:
            desc = desc[: DESC_CHARS - 1] + "..."
        rows.append({"id": node.id, "description": desc})
    return rows


def batched(items: Sequence[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def build_user_prompt(rows: Sequence[Mapping[str, str]]) -> str:
    payload = json.dumps(list(rows), ensure_ascii=False, indent=2)
    return (
        "Cluster synonymous nodes from this list. "
        "Return groups only.\n\n"
        f"{payload}\n"
    )


def groups_from_payload(payload: Any) -> List[AliasGroup]:
    """Accept structured LexiconGroups, a groups list, or canon→aliases dict."""
    if payload is None:
        return []
    if isinstance(payload, LexiconGroups):
        return list(payload.groups)
    if isinstance(payload, AliasGroup):
        return [payload]
    if isinstance(payload, str):
        text = payload.strip()
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            text = fence.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return []
    if isinstance(payload, list):
        out: List[AliasGroup] = []
        for item in payload:
            out.extend(groups_from_payload(item))
        return out
    if isinstance(payload, dict):
        if "groups" in payload:
            return groups_from_payload(payload["groups"])
        if "canon" in payload:
            aliases = payload.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            return [AliasGroup(canon=str(payload["canon"]), aliases=[str(a) for a in aliases])]
        # Final lexicon shape: {canon: [aliases]}
        groups: List[AliasGroup] = []
        for canon, aliases in payload.items():
            if aliases is None:
                alias_list: List[str] = []
            elif isinstance(aliases, str):
                alias_list = [aliases]
            elif isinstance(aliases, list):
                alias_list = [str(a) for a in aliases]
            else:
                continue
            groups.append(AliasGroup(canon=str(canon), aliases=alias_list))
        return groups
    return []


def filter_groups(
    groups: Sequence[AliasGroup],
    by_norm: Mapping[str, str],
    by_fp: Mapping[str, str],
) -> List[Tuple[str, List[str]]]:
    """Drop hallucinated ids. Keep groups with ≥1 real alias distinct from canon."""
    kept: List[Tuple[str, List[str]]] = []
    for group in groups:
        canon = resolve_to_known(group.canon, by_norm, by_fp)
        aliases: List[str] = []
        seen = set()
        for raw in group.aliases:
            resolved = resolve_to_known(raw, by_norm, by_fp)
            if resolved is None or resolved == canon:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            aliases.append(resolved)
        if canon is None:
            if not aliases:
                continue
            canon = aliases[0]
            aliases = aliases[1:]
        if aliases:
            kept.append((canon, aliases))
    return kept


def merge_alias_groups(
    pairs: Iterable[Tuple[str, Sequence[str]]],
) -> Dict[str, List[str]]:
    """Union-find merge overlapping synonym groups into lexicon JSON shape."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    preferred_canon: Dict[str, str] = {}
    for canon, aliases in pairs:
        members = [canon, *aliases]
        for m in members:
            parent.setdefault(m, m)
        for m in members[1:]:
            union(canon, m)
        root = find(canon)
        preferred_canon.setdefault(root, canon)

    clusters: Dict[str, List[str]] = {}
    for nid in parent:
        clusters.setdefault(find(nid), []).append(nid)

    out: Dict[str, List[str]] = {}
    for root, members in clusters.items():
        uniq = list(dict.fromkeys(members))
        if len(uniq) < 2:
            continue
        canon = preferred_canon.get(root, uniq[0])
        if canon not in uniq:
            canon = uniq[0]
        aliases = [m for m in uniq if m != canon]
        out[canon] = aliases
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def lexicon_from_groups(
    groups: Sequence[AliasGroup],
    nodes: Sequence[GraphNode],
) -> Dict[str, List[str]]:
    by_norm, by_fp = known_id_index(nodes)
    filtered = filter_groups(groups, by_norm, by_fp)
    return merge_alias_groups(filtered)


def preview_cost(
    prompts: Sequence[str],
    *,
    model: str,
    base_url: str,
    output_tokens_per_batch: int = DEFAULT_OUTPUT_TOKENS_PER_BATCH,
) -> str:
    in_tokens = sum(count_tokens(SYSTEM_PROMPT) + count_tokens(p) for p in prompts)
    out_tokens = output_tokens_per_batch * max(1, len(prompts))
    book = fetch_openrouter_rate_book([model], base_url=base_url)
    rate = book.rate_for(model)
    usd = calculate_cost(model, in_tokens, out_tokens, rate=rate)
    status = format_cost_status(usd, rate.status)
    return (
        f"[cost preview] model={model} batches={len(prompts)} "
        f"in_tokens~={in_tokens} out_tokens~={out_tokens} -> {status}"
    )


def _invoke_structured(llm: Any, user_prompt: str) -> Any:
    structured = llm.with_structured_output(LexiconGroups)
    return structured.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )


def collect_groups_from_llm(
    rows: Sequence[Mapping[str, str]],
    llm: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[AliasGroup]:
    groups: List[AliasGroup] = []

    def run_batch(batch: List[Mapping[str, str]], depth: int = 0) -> None:
        prompt = build_user_prompt(batch)
        print(f"[bootstrap] LLM batch size={len(batch)} depth={depth}...")
        try:
            parsed = _invoke_structured(llm, prompt)
        except Exception as exc:  # noqa: BLE001 — split on output-length / parse fail
            name = type(exc).__name__
            if len(batch) <= 20 or depth > 4:
                print(f"[bootstrap] skip batch ({name}): {exc}", file=sys.stderr)
                return
            mid = len(batch) // 2
            print(f"[bootstrap] {name}; splitting {len(batch)} -> {mid}+{len(batch)-mid}")
            run_batch(batch[:mid], depth + 1)
            run_batch(batch[mid:], depth + 1)
            return
        groups.extend(groups_from_payload(parsed))

    for batch in batched(list(rows), batch_size):
        run_batch(batch)
    return groups


def resolve_model(cli_model: Optional[str]) -> str:
    """CLI > LEXICON_BOOTSTRAP_MODEL env > SYNTHESIS_MODEL."""
    if cli_model and cli_model.strip():
        return cli_model.strip()
    env = (os.getenv("LEXICON_BOOTSTRAP_MODEL") or "").strip().strip('"').strip("'")
    if env:
        return env
    return default_settings.synthesis_model


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap entity_aliases.json via LLM")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--model", default=None, help="Override LEXICON_BOOTSTRAP_MODEL")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file",
    )
    p.add_argument(
        "--print",
        dest="print_only",
        action="store_true",
        help="Print JSON; do not write the output file",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Cost preview + prompt sizes only (no LLM, no write)",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    path: Path = args.checkpoint
    if not path.is_file():
        print(f"[FATAL] checkpoint not found: {path}", file=sys.stderr)
        return 1
    nodes, _edges = load_extraction_checkpoint(path.parent, filename=path.name)
    rows = catalog_rows(nodes)
    model = resolve_model(args.model)
    batches = batched(rows, args.batch_size)
    prompts = [build_user_prompt(b) for b in batches]
    print(
        f"[bootstrap] nodes={len(nodes)} batches={len(batches)} "
        f"batch_size={args.batch_size} model={model}"
    )
    print(
        preview_cost(
            prompts,
            model=model,
            base_url=default_settings.openrouter_base_url,
        )
    )
    if args.dry_run:
        return 0

    out_path: Path = args.output
    if out_path.is_file() and out_path.stat().st_size > 4 and not args.force and not args.print_only:
        print(
            f"[FATAL] output exists ({out_path}); pass --force to overwrite",
            file=sys.stderr,
        )
        return 1

    from graph_distiller.adapters.openrouter import default_llm_chat

    llm = default_llm_chat(default_settings, model=model)
    groups = collect_groups_from_llm(rows, llm, batch_size=args.batch_size)
    lexicon = lexicon_from_groups(groups, nodes)
    text = json.dumps(lexicon, ensure_ascii=False, indent=2) + "\n"
    n_alias = sum(len(v) for v in lexicon.values())
    print(
        f"[bootstrap] groups={len(lexicon)} aliases={n_alias} "
        f"(LLM raw groups={len(groups)})"
    )
    if args.print_only:
        sys.stdout.write(text)
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
