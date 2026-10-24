"""
Ingestion loaders for data/raw_memos/ (SPEC Level 3.5 §1 + .docx/.json).

Extension routing:
  .txt / .md  -> plain UTF-8 text read
  .docx       -> langchain_community Docx2txtLoader
                (fallback: UnstructuredWordDocumentLoader)
  .json       -> parse JSON and flatten key-value pairs to plain text
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".docx", ".json"}


def sanitize_source_doc_id(stem: str) -> str:
    """
    ADR-056: collapse whitespace in file stems for DOC_ID stability.

    Example: ``어떻게 살 것인가`` -> ``어떻게_살_것인가``.
    Special-character scrubbing is out of this slice (blueprint 1.1 remainder).
    """
    return re.sub(r"\s+", "_", stem)


def _load_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _load_docx(path: Path) -> str:
    """
    Prefer Docx2txtLoader; fall back to UnstructuredWordDocumentLoader.
    """
    try:
        from langchain_community.document_loaders import Docx2txtLoader

        docs = Docx2txtLoader(str(path)).load()
        return "\n\n".join(d.page_content for d in docs).strip()
    except Exception as primary_exc:  # noqa: BLE001
        logger.warning(
            "Docx2txtLoader failed for %s (%s); trying UnstructuredWordDocumentLoader",
            path.name,
            primary_exc,
        )
        from langchain_community.document_loaders import (
            UnstructuredWordDocumentLoader,
        )

        docs = UnstructuredWordDocumentLoader(str(path)).load()
        return "\n\n".join(d.page_content for d in docs).strip()


def flatten_json(data: Any, prefix: str = "") -> List[str]:
    """
    Recursively flatten JSON into `key: value` text lines for the chunker.

    - dict  -> prefix.key
    - list  -> prefix[i]
    - scalar -> "prefix: value" (or str(value) if no prefix)
    """
    lines: List[str] = []

    if isinstance(data, dict):
        if not data and prefix:
            lines.append(f"{prefix}: {{}}")
            return lines
        for key, value in data.items():
            key_str = str(key)
            next_prefix = f"{prefix}.{key_str}" if prefix else key_str
            lines.extend(flatten_json(value, next_prefix))
        return lines

    if isinstance(data, list):
        if not data and prefix:
            lines.append(f"{prefix}: []")
            return lines
        for index, item in enumerate(data):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            lines.extend(flatten_json(item, next_prefix))
        return lines

    # bool/int/float/str/None
    if data is None:
        rendered = "null"
    elif isinstance(data, bool):
        rendered = "true" if data else "false"
    else:
        rendered = str(data).strip()

    if not rendered and not prefix:
        return lines
    if prefix:
        lines.append(f"{prefix}: {rendered}")
    else:
        lines.append(rendered)
    return lines


def _load_json(path: Path) -> str:
    """Parse a .json file and flatten key-value pairs into chunker-ready text."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    lines = flatten_json(data)
    return "\n".join(lines).strip()


def load_file_content(path: Path) -> str:
    """Dispatch to the appropriate loader by file extension."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _load_plain_text(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix == ".json":
        return _load_json(path)
    raise ValueError(f"Unsupported memo extension: {suffix} ({path})")


def load_raw_memos(raw_dir: Path) -> List[Dict[str, Any]]:
    """
    Walk raw_dir and load each supported memo file.

    source_doc_id = filename stem with whitespace collapsed to ``_`` (ADR-056).
    예: memo_20231012.json -> source_doc_id="memo_20231012"
    예: ``어떻게 살 것인가.txt`` -> source_doc_id="어떻게_살_것인가"
    """
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw memos directory not found: {raw_dir}\n"
            f"Create it and drop .txt / .md / .docx / .json files there, then re-run."
        )

    documents: List[Dict[str, Any]] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            content = load_file_content(path)
        except Exception as exc:  # noqa: BLE001 — skip bad file, continue batch
            logger.error("Failed to load %s: %s", path, exc)
            continue
        if not content:
            continue
        documents.append(
            {
                "source_doc_id": sanitize_source_doc_id(path.stem),
                "path": str(path.resolve()),
                "content": content,
                "suffix": path.suffix.lower(),
            }
        )
    return documents
