"""Phase 2/4 프롬프트 통제실 로더 (ADR-059).

`data/prompts/*.md`를 읽어 런타임에 시스템 프롬프트를 주입한다. 파일이 없거나
비어 있으면 호출부가 넘긴 fallback(과거 하드코딩 상수)으로 안전하게 강등된다 —
도메인 전문가가 편집하는 .md가 지워지거나 깨져도 파이프라인은 항상 끝까지 돈다.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_prompt(prompts_dir: Path, filename: str, fallback: str) -> str:
    """`prompts_dir/filename`을 읽어 반환한다. 없거나 비면 fallback을 반환한다."""
    path = Path(prompts_dir) / filename
    if not path.is_file():
        logger.warning("Prompt file not found: %s — using in-code fallback.", path)
        return fallback
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        logger.warning("Prompt file empty: %s — using in-code fallback.", path)
        return fallback
    return text
