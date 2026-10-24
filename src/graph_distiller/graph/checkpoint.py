"""Persist Phase 2 Node/Edge checkpoint (SPEC Level 2 §4 MUST)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

from graph_distiller.schema import GraphEdge, GraphNode


def save_extraction_checkpoint(
    nodes: List[GraphNode],
    edges: List[GraphEdge],
    checkpoint_dir: Path,
    filename: str = "phase2_extraction.json",
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    payload = {
        "nodes": [n.model_dump() for n in nodes],
        "edges": [e.model_dump() for e in edges],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_extraction_checkpoint(
    checkpoint_dir: Path,
    filename: str = "phase2_extraction.json",
) -> Tuple[List[GraphNode], List[GraphEdge]]:
    path = checkpoint_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = [GraphNode.model_validate(n) for n in data.get("nodes", [])]
    edges = [GraphEdge.model_validate(e) for e in data.get("edges", [])]
    return nodes, edges
