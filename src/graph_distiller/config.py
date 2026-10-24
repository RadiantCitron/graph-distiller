"""Runtime configuration — OpenRouter LLM wiring (SPEC scaffolding §5).

초심자: 환경변수 의미·Phase 맵은 `.env` 주석과
`docs/design/attachments/env-beginner.md` (ADR-029)를 본다.
이 모듈은 값을 읽어 Settings로 묶을 뿐, 암묵 모델 폴백은 하지 않는다.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from graph_distiller.workspace import (
    _warn_deprecated_knob,
    reset_deprecated_knobs_warnings,
    resolve_workspace,
)

# config.py lives at src/graph_distiller/config.py → parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"

# CWD-independent: always load repo-root .env (works from src/, IDE, etc.)
_dotenv_loaded = load_dotenv(dotenv_path=ENV_PATH, override=False)
if not ENV_PATH.is_file():
    # Still allow OS env injection, but warn loudly
    logging.getLogger(__name__).warning(
        ".env not found at %s (CWD=%s). Relying on process environment only.",
        ENV_PATH,
        Path.cwd(),
    )

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Canonical PoC defaults (must also be set explicitly in .env)
DEFAULT_EXTRACTION_MODEL = "openai/gpt-4o-mini"
DEFAULT_SYNTHESIS_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

logger = logging.getLogger(__name__)


def _env_get(name: str, default: str = "") -> str:
    """Read env var and strip optional surrounding quotes."""
    return (os.getenv(name, default) or "").strip().strip('"').strip("'")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_get(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def _env_choice(name: str, allowed: set[str], default: str) -> str:
    """Return lowercased env value if in allowed; else default."""
    raw = _env_get(name, default).lower()
    if raw in allowed:
        return raw
    logger.warning(
        "%s=%r invalid (allowed=%s); using %s",
        name,
        raw,
        sorted(allowed),
        default,
    )
    return default


def _env_knob_str(name: str, default: str) -> str:
    """Read knob env var, emitting deprecation warning if present (D2, ADR-077)."""
    raw = os.getenv(name)
    if raw is not None and raw.strip() != "":
        _warn_deprecated_knob(name)
        return raw.strip().strip('"').strip("'")
    return default


def _env_knob_bool(name: str, default: bool = False) -> bool:
    """Read boolean knob env var, emitting deprecation warning if present."""
    raw = os.getenv(name)
    if raw is not None and raw.strip() != "":
        _warn_deprecated_knob(name)
        val = raw.strip().strip('"').strip("'").lower()
        return val in {"1", "true", "yes", "on"}
    return default


def _env_knob_choice(name: str, allowed: set[str], default: str) -> str:
    """Read choice knob env var, emitting deprecation warning if present."""
    raw = os.getenv(name)
    if raw is not None and raw.strip() != "":
        _warn_deprecated_knob(name)
        val = raw.strip().strip('"').strip("'").lower()
        if val in allowed:
            return val
        logger.warning(
            "%s=%r invalid (allowed=%s); using %s",
            name,
            val,
            sorted(allowed),
            default,
        )
    return default


def _require_env(name: str, *, example: str) -> str:
    """Return env value or exit with a clear message (no silent fallback)."""
    value = _env_get(name)
    if value:
        return value
    logger.error(
        "%s is not set (looked for .env at %s). Add e.g. %s=%s and re-run.",
        name,
        ENV_PATH,
        name,
        example,
    )
    print(
        f"[FATAL] {name} 미설정. 프로젝트 루트 `{ENV_PATH}` 에 "
        f"`{name}={example}` 를 넣고 다시 실행하세요.",
        file=sys.stderr,
    )
    raise SystemExit(1)


@dataclass(frozen=True)
class Settings:
    # --- 자격증명 / 엔드포인트 ---
    openrouter_api_key: str  # 유료 API 키 (.env only, never commit)
    openrouter_base_url: str  # 보통 https://openrouter.ai/api/v1
    # --- 모델 (Phase 역할이 다름 — env-beginner.md) ---
    embedding_model: str  # Phase 1 SemanticChunker
    extraction_model: str  # Phase 2 추출 + Phase 4 제목 (필수 env)
    synthesis_model: str  # Phase 4 챕터 본문 집필 (필수 env)
    # --- 입출력 경로 ---
    raw_memos_dir: Path
    checkpoint_dir: Path
    codex_output_dir: Path
    errors_log_path: Path
    # --- 프롬프트 통제실 (ADR-059) — Phase 2/4 시스템 프롬프트 .md 위치 ---
    prompts_dir: Path
    # --- 규칙 통제실 (ADR-060) — Phase 2/4 판정 규칙 .toml 위치 ---
    rules_dir: Path
    # --- 실 RAG (ADR-033) — 읽기 전용 URI; 값은 로그/스냅샷에 넣지 않음 ---
    rag_db_uri: str  # 비어 있으면 파일 PoC만; CLI skeleton은 필수
    rag_rebuild_dir: Path  # output/rag_rebuild 기본
    # --- 실행 장부 / 보고서 경로 (ROADMAP_0001 step 030) ---
    runs_dir: Path
    reports_dir: Path
    # --- Phase 1 / 2 / 3 / 4 노브 ---
    max_chunk_tokens: int
    semantic_breakpoint_percentile: int
    llm_concurrency: int  # Phase 2 동시 호출 상한
    max_chapter_retries: int  # Verifier FAIL 시 챕터당 재시도 한도 → 초과 시 poison
    poc_top_chapters: int  # ADR-055: 0=전 커뮤니티(기본). >0이면 PoC 상위 N만 집필

    louvain_resolution: float
    min_community_size: int
    min_chapter_grounding_chunks: int  # ADR-057: <이 값 unique 원문 청크는 챕터 제외
    repomap_small_keep_all_below: int
    repomap_snippet_chars: int
    verify_citation_paragraph_ratio: float
    verify_citation_sentence_ratio: float
    # Phase 2 Reduce 2 (ADR-042) — default off so spelling-only golden stays
    canon_embed_merge: bool
    canon_embed_sim_threshold: float  # unmeasured placeholder, not a gate
    canon_embed_max_nodes: int
    # Phase 2 Reduce 3 (W2-D) — domain alias lexicon JSON; missing = no-op
    canon_alias_lexicon_path: Path
    # Phase 2→3 scale guard (W3-A) — in-memory OOM preflight
    graph_max_nodes: int
    graph_max_edges: int
    graph_max_est_mb: float
    graph_scale_on_exceed: str  # abort | slice
    # --- 워크스페이스 (ROADMAP_0001 step 040) ---
    workspace_dir: Optional[Path] = None

    def ensure_directories(self) -> None:
        """Create I/O dirs so Phase 2/4 never die on FileNotFoundError."""
        if self.workspace_dir is not None:
            if not self.raw_memos_dir.is_dir():
                raise FileNotFoundError(
                    f"Workspace input directory not found: {self.raw_memos_dir}"
                )
        else:
            self.raw_memos_dir.mkdir(parents=True, exist_ok=True)
            self.rag_rebuild_dir.mkdir(parents=True, exist_ok=True)
            self.canon_alias_lexicon_path.parent.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.codex_output_dir.mkdir(parents=True, exist_ok=True)
        self.errors_log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Ensured dirs: raw_memos=%s checkpoint=%s codex=%s rag_rebuild=%s "
            "lexicon_parent=%s errors_parent=%s",
            self.raw_memos_dir,
            self.checkpoint_dir,
            self.codex_output_dir,
            self.rag_rebuild_dir,
            self.canon_alias_lexicon_path.parent,
            self.errors_log_path.parent,
        )

    @classmethod
    def from_env(
        cls,
        *,
        models_required: bool = True,
        workspace_dir: Optional[Path] = None,
    ) -> "Settings":
        root = PROJECT_ROOT

        if models_required:
            extraction_model = _require_env(
                "EXTRACTION_MODEL", example=DEFAULT_EXTRACTION_MODEL
            )
            synthesis_model = _require_env(
                "SYNTHESIS_MODEL", example=DEFAULT_SYNTHESIS_MODEL
            )
        else:
            # Compiler injects models from CLI (ADR-053). Env slugs are ignored.
            extraction_model = _env_get("EXTRACTION_MODEL")
            synthesis_model = _env_get("SYNTHESIS_MODEL")

        settings = cls(
            openrouter_api_key=_env_get("OPENROUTER_API_KEY"),
            openrouter_base_url=_env_get(
                "OPENROUTER_BASE_URL", OPENROUTER_BASE_URL
            )
            or OPENROUTER_BASE_URL,
            embedding_model=_env_get(
                "EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            )
            or DEFAULT_EMBEDDING_MODEL,
            extraction_model=extraction_model,
            synthesis_model=synthesis_model,
            raw_memos_dir=Path(
                _env_get("RAW_MEMOS_DIR", str(root / "data" / "raw_memos"))
                or str(root / "data" / "raw_memos")
            ),
            checkpoint_dir=Path(
                _env_get("CHECKPOINT_DIR", str(root / "data" / "checkpoints"))
                or str(root / "data" / "checkpoints")
            ),
            codex_output_dir=Path(
                _env_get("CODEX_OUTPUT_DIR", str(root / "output" / "codex"))
                or str(root / "output" / "codex")
            ),
            errors_log_path=Path(
                _env_get("ERRORS_LOG_PATH", str(root / "output" / "errors.log"))
                or str(root / "output" / "errors.log")
            ),
            prompts_dir=Path(
                _env_get("PROMPTS_DIR", str(root / "data" / "prompts"))
                or str(root / "data" / "prompts")
            ),
            rules_dir=Path(
                _env_get("RULES_DIR", str(root / "data" / "rules"))
                or str(root / "data" / "rules")
            ),
            rag_db_uri=_env_get("RAG_DB_URI") or _env_get("DB_URI"),
            rag_rebuild_dir=Path(
                _env_get(
                    "RAG_REBUILD_DIR", str(root / "output" / "rag_rebuild")
                )
                or str(root / "output" / "rag_rebuild")
            ),
            runs_dir=Path(
                _env_get("RUNS_DIR", str(root / "output" / "runs"))
                or str(root / "output" / "runs")
            ),
            reports_dir=Path(
                _env_get("REPORTS_DIR", str(root / "tests" / "reports"))
                or str(root / "tests" / "reports")
            ),
            max_chunk_tokens=int(_env_knob_str("MAX_CHUNK_TOKENS", "1000") or "1000"),
            semantic_breakpoint_percentile=int(
                _env_knob_str("SEMANTIC_BREAKPOINT_PERCENTILE", "80") or "80"
            ),
            llm_concurrency=int(_env_knob_str("LLM_CONCURRENCY", "20") or "20"),
            max_chapter_retries=int(
                _env_knob_str("MAX_CHAPTER_RETRIES", "3") or "3"
            ),
            poc_top_chapters=int(_env_knob_str("POC_TOP_CHAPTERS", "0") or "0"),
            louvain_resolution=float(
                _env_knob_str("LOUVAIN_RESOLUTION", "0.5") or "0.5"
            ),
            min_community_size=int(
                _env_knob_str("MIN_COMMUNITY_SIZE", "8") or "8"
            ),
            min_chapter_grounding_chunks=int(
                _env_knob_str("MIN_CHAPTER_GROUNDING_CHUNKS", "2") or "2"
            ),
            repomap_small_keep_all_below=int(
                _env_knob_str("REPOMAP_SMALL_KEEP_ALL_BELOW", "10") or "10"
            ),
            repomap_snippet_chars=int(
                _env_knob_str("REPOMAP_SNIPPET_CHARS", "2000") or "2000"
            ),
            verify_citation_paragraph_ratio=float(
                _env_knob_str("VERIFY_CITATION_PARAGRAPH_RATIO", "1.0") or "1.0"
            ),
            verify_citation_sentence_ratio=float(
                _env_knob_str("VERIFY_CITATION_SENTENCE_RATIO", "0.25") or "0.25"
            ),
            canon_embed_merge=_env_knob_bool("CANON_EMBED_MERGE", False),
            canon_embed_sim_threshold=float(
                _env_knob_str("CANON_EMBED_SIM_THRESHOLD", "0.90") or "0.90"
            ),
            canon_embed_max_nodes=int(
                _env_knob_str("CANON_EMBED_MAX_NODES", "2000") or "2000"
            ),
            canon_alias_lexicon_path=Path(
                _env_get(
                    "CANON_ALIAS_LEXICON_PATH",
                    str(root / "data" / "lexicon" / "entity_aliases.json"),
                )
                or str(root / "data" / "lexicon" / "entity_aliases.json")
            ),
            graph_max_nodes=int(
                _env_knob_str("GRAPH_MAX_NODES", "100000") or "100000"
            ),
            graph_max_edges=int(
                _env_knob_str("GRAPH_MAX_EDGES", "500000") or "500000"
            ),
            graph_max_est_mb=float(
                _env_knob_str("GRAPH_MAX_EST_MB", "1024") or "1024"
            ),
            graph_scale_on_exceed=_env_knob_choice(
                "GRAPH_SCALE_ON_EXCEED",
                {"abort", "slice"},
                "abort",
            ),
            workspace_dir=workspace_dir,
        )
        if workspace_dir is not None:
            resolved = resolve_workspace(workspace_dir=workspace_dir)
            settings = replace(
                settings,
                workspace_dir=resolved.workspace_dir,
                raw_memos_dir=resolved.raw_memos_dir,
                checkpoint_dir=resolved.checkpoint_dir,
                codex_output_dir=resolved.codex_output_dir,
                errors_log_path=resolved.errors_log_path,
                rag_rebuild_dir=resolved.rag_rebuild_dir,
                runs_dir=resolved.runs_dir,
                prompts_dir=resolved.prompts_dir,
                rules_dir=resolved.rules_dir,
                canon_alias_lexicon_path=resolved.canon_alias_lexicon_path,
                **resolved.knobs,
            )
        if _dotenv_loaded:
            logger.info("Loaded environment from %s", ENV_PATH)
        return settings

    def build_chat_model(self, model: str | None = None) -> ChatOpenAI:
        """
        LangChain ChatOpenAI bound to OpenRouter.
        MUST inject base_url + api_key explicitly.
        """
        if not self.openrouter_api_key:
            raise ValueError(
                f"OPENROUTER_API_KEY is not set "
                f"(expected in {ENV_PATH} or process env)"
            )
        return ChatOpenAI(
            model=model or self.extraction_model,
            api_key=self.openrouter_api_key,
            base_url=self.openrouter_base_url,
            temperature=0,
        )

    def build_embeddings(self) -> OpenAIEmbeddings:
        """Embeddings via OpenRouter-compatible OpenAI SDK endpoint."""
        if not self.openrouter_api_key:
            raise ValueError(
                f"OPENROUTER_API_KEY is not set "
                f"(expected in {ENV_PATH} or process env)"
            )
        return OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.openrouter_api_key,
            base_url=self.openrouter_base_url,
        )


settings = Settings.from_env(models_required=False)
_runtime_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-scoped Settings (CLI compiler may replace env models)."""
    return _runtime_settings if _runtime_settings is not None else settings


def set_runtime_settings(cfg: Settings) -> None:
    """Bind Settings for this process. Does not write .env."""
    global settings, _runtime_settings
    _runtime_settings = cfg
    settings = cfg
