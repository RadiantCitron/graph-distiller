"""Run ledger — meta-only execution journal (ADR-026).

Persists under output/runs/codex_YYYYMMDD_HHMMSS/ledger.json.
No chunk bodies, triples, chapter prose, or API keys.
Does not extend GraphDistillerState. cost.py stays pure.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, TextIO

from graph_distiller.config import PROJECT_ROOT, Settings, get_settings
from graph_distiller.rules import Rules, get_rules

DEFAULT_RUNS_ROOT = PROJECT_ROOT / "output" / "runs"
LEDGER_FILENAME = "ledger.json"
RUN_LOG_FILENAME = "run.log"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime) -> str:
    return ts.isoformat()


def make_run_id(when: Optional[datetime] = None) -> str:
    ts = when or _utc_now()
    return f"codex_{ts.strftime('%Y%m%d_%H%M%S')}"


def create_run_directory(
    runs_root: Path | None = None,
    *,
    when: Optional[datetime] = None,
) -> Path:
    """
    Create output/runs/codex_YYYYMMDD_HHMMSS/ (uses get_settings().runs_dir by default).

    ``runs_root`` is the parent of the per-run folder (ADR-026: --output-dir
    will override this in a later stage).
    """
    root = Path(runs_root) if runs_root is not None else get_settings().runs_dir
    run_id = make_run_id(when)
    path = root / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def settings_public_snapshot(cfg: Settings) -> Dict[str, Any]:
    """Model slugs and paths only — never API keys."""
    return {
        "openrouter_base_url": cfg.openrouter_base_url,
        "embedding_model": cfg.embedding_model,
        "extraction_model": cfg.extraction_model,
        "synthesis_model": cfg.synthesis_model,
        "raw_memos_dir": str(cfg.raw_memos_dir),
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "codex_output_dir": str(cfg.codex_output_dir),
        "errors_log_path": str(cfg.errors_log_path),
        "prompts_dir": str(cfg.prompts_dir),
        "rules_dir": str(cfg.rules_dir),
        "rag_db_uri_set": bool(getattr(cfg, "rag_db_uri", "") or ""),
        "rag_rebuild_dir": str(getattr(cfg, "rag_rebuild_dir", "") or ""),
        "runs_dir": str(getattr(cfg, "runs_dir", "") or ""),
        "reports_dir": str(getattr(cfg, "reports_dir", "") or ""),
        "canon_alias_lexicon_path": str(getattr(cfg, "canon_alias_lexicon_path", "") or ""),
        "workspace_dir": str(cfg.workspace_dir) if getattr(cfg, "workspace_dir", None) else None,
        "max_chunk_tokens": cfg.max_chunk_tokens,
        "semantic_breakpoint_percentile": cfg.semantic_breakpoint_percentile,
        "llm_concurrency": cfg.llm_concurrency,
        "max_chapter_retries": cfg.max_chapter_retries,
        "poc_top_chapters": cfg.poc_top_chapters,
        "louvain_resolution": cfg.louvain_resolution,
        "min_community_size": cfg.min_community_size,
        "min_chapter_grounding_chunks": getattr(cfg, "min_chapter_grounding_chunks", 2),
        "repomap_small_keep_all_below": cfg.repomap_small_keep_all_below,
        "repomap_snippet_chars": cfg.repomap_snippet_chars,
        "verify_citation_paragraph_ratio": cfg.verify_citation_paragraph_ratio,
        "verify_citation_sentence_ratio": cfg.verify_citation_sentence_ratio,
        "canon_embed_merge": getattr(cfg, "canon_embed_merge", False),
        "canon_embed_sim_threshold": getattr(cfg, "canon_embed_sim_threshold", 0.9),
        "canon_embed_max_nodes": getattr(cfg, "canon_embed_max_nodes", 2000),
        "graph_max_nodes": getattr(cfg, "graph_max_nodes", None),
        "graph_max_edges": getattr(cfg, "graph_max_edges", None),
        "graph_max_est_mb": getattr(cfg, "graph_max_est_mb", None),
        "graph_scale_on_exceed": getattr(cfg, "graph_scale_on_exceed", None),
        "openrouter_api_key_set": bool(cfg.openrouter_api_key),
    }


def _file_sha256(path: Path | None) -> Optional[str]:
    import hashlib

    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel_path_str(path: Path | None, *, base_dir: Path | None = None) -> Optional[str]:
    if path is None:
        return None
    p = Path(path)
    if base_dir is not None:
        try:
            return p.relative_to(base_dir).as_posix()
        except ValueError:
            pass
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.name


def control_room_snapshot(cfg: Settings, *, rules: Optional[Rules] = None) -> Dict[str, Any]:
    """Applied prompt/rule control-room state (ADR-059/060) + workspace provenance (ADR-075, Step 060).

    Records what actually ran, hashes of all prompt/rule/lexicon files, applied rules,
    and workspace metadata with file-by-file provenance (D5). Relative paths only for new fields.
    """
    import os
    from graph_distiller.workspace import (
        KNOB_ENV_TO_FIELD,
        LEXICON_FILENAME,
        PROMPT_FILES,
        RULE_FILES,
        load_profile_file,
        load_workspace_file,
        resolve_workspace,
    )

    rules = rules if rules is not None else get_rules(cfg.rules_dir)
    prompts_dir = Path(cfg.prompts_dir)
    rules_dir = Path(cfg.rules_dir)

    prompt_files = {
        name: _file_sha256(prompts_dir / name)
        for name in (
            "phase2_extraction.md",
            "phase4_planner.md",
            "phase4_synthesizer.md",
            "eval_judge.md",
        )
    }
    rule_files = {
        name: _file_sha256(rules_dir / name)
        for name in (
            "phase2_canonicalize.toml",
            "phase4_repomap.toml",
            "phase4_planner.toml",
            "phase4_verifier.toml",
            "eval_scoring.toml",
        )
    }

    lexicon_path = (
        Path(cfg.canon_alias_lexicon_path)
        if getattr(cfg, "canon_alias_lexicon_path", None)
        else None
    )
    lexicon_sha256 = _file_sha256(lexicon_path) if lexicon_path else None

    ws_meta: Optional[Dict[str, Any]] = None
    policy_files_meta: Dict[str, Dict[str, Any]] = {}
    knobs_meta: Dict[str, Dict[str, Any]] = {}
    field_to_env = {v: k for k, v in KNOB_ENV_TO_FIELD.items()}

    if getattr(cfg, "workspace_dir", None) is not None:
        ws_path = Path(cfg.workspace_dir)
        ws_file = ws_path / "workspace.toml"
        ws_name = ws_path.name
        corpus_path: Optional[str] = None
        profile_path: Optional[str] = None
        profile_name: Optional[str] = None

        if ws_file.is_file():
            try:
                ws_schema = load_workspace_file(ws_file)
                ws_name = ws_schema.name or ws_path.name
                if ws_schema.corpus and ws_schema.corpus.path:
                    corpus_path = ws_schema.corpus.path
                elif ws_schema.paths and ws_schema.paths.input:
                    corpus_path = ws_schema.paths.input

                if ws_schema.profile and ws_schema.profile.path:
                    profile_path = ws_schema.profile.path
                    prof_dir = (ws_path / profile_path).resolve()
                    prof_file = prof_dir / "profile.toml"
                    if prof_file.is_file():
                        try:
                            prof_schema = load_profile_file(prof_file)
                            profile_name = prof_schema.name
                        except Exception:
                            pass
            except Exception:
                pass

        ws_meta = {
            "name": ws_name,
            "corpus_path": corpus_path,
            "profile_path": profile_path,
            "profile_name": profile_name,
        }

        try:
            resolved = resolve_workspace(ws_path)
            for pf in resolved.policy_files.values():
                policy_files_meta[pf.file_name] = {
                    "source": pf.source,
                    "path": _rel_path_str(pf.resolved_path, base_dir=ws_path),
                    "sha256": _file_sha256(pf.resolved_path),
                }

            for field in sorted(KNOB_ENV_TO_FIELD.values()):
                val = resolved.knobs.get(field)
                prov = resolved.provenance.get(field)
                src = prov.source if prov else "engine_default"
                knobs_meta[field] = {
                    "value": val,
                    "source": src,
                }
        except Exception:
            pass

    else:
        for rf in RULE_FILES:
            rf_p = rules_dir / rf
            policy_files_meta[rf] = {
                "source": "env" if "RULES_DIR" in os.environ else "engine_default",
                "path": _rel_path_str(rf_p),
                "sha256": _file_sha256(rf_p),
            }
        for pf in PROMPT_FILES:
            pf_p = prompts_dir / pf
            policy_files_meta[pf] = {
                "source": "env" if "PROMPTS_DIR" in os.environ else "engine_default",
                "path": _rel_path_str(pf_p),
                "sha256": _file_sha256(pf_p),
            }
        if lexicon_path:
            policy_files_meta[LEXICON_FILENAME] = {
                "source": "env" if "CANON_ALIAS_LEXICON_PATH" in os.environ else "engine_default",
                "path": _rel_path_str(lexicon_path),
                "sha256": lexicon_sha256,
            }

        for field in sorted(KNOB_ENV_TO_FIELD.values()):
            val = getattr(cfg, field, None)
            env_k = field_to_env.get(field, "")
            src = "env" if env_k in os.environ and os.environ[env_k] else "engine_default"
            knobs_meta[field] = {
                "value": val,
                "source": src,
            }

    return {
        "prompts_dir": str(prompts_dir),
        "rules_dir": str(rules_dir),
        "prompt_file_sha256": prompt_files,
        "rule_file_sha256": rule_files,
        "lexicon_file_sha256": lexicon_sha256,
        "workspace": ws_meta,
        "policy_files": policy_files_meta,
        "knobs": knobs_meta,
        "applied_rules": rules.model_dump(),
    }


def usage_public_dict(usage: object | None) -> Optional[Dict[str, Any]]:
    """PhaseUsage → JSON-safe meta (no secrets)."""
    if usage is None:
        return None
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "total_cost_usd": float(getattr(usage, "total_cost_usd", 0.0) or 0.0),
    }


def estimate_public_dict(est: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    """Pre-flight cost dict → JSON-safe subset (counts/tokens/cost/model only)."""
    if not est:
        return None
    keys = (
        "chunk_count",
        "n_chapters",
        "repo_map_tokens",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "model_name",
        "synthesis_model",
        "planner_model",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        if key in est:
            out[key] = est[key]
    return out or None


def chapters_meta_from_state(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Chapter meta only (ADR-026 stage 4) — no chapter body.
    Includes drafts, poisoned, and titled chapters.
    """
    drafts = {d.chapter_id: d for d in (state.get("drafts") or [])}
    retries = dict(state.get("retry_counts") or {})
    poisoned = set(state.get("poisoned_chapters") or [])
    titles = dict(state.get("chapter_titles") or {})
    toc_ids = set(state.get("toc") or {})
    ids = sorted(toc_ids | set(drafts) | poisoned | set(titles))
    rows: List[Dict[str, Any]] = []
    for chapter_id in ids:
        draft = drafts.get(chapter_id)
        title = ""
        if draft is not None and draft.title:
            title = draft.title
        elif chapter_id in titles:
            title = str(titles[chapter_id])
        rows.append(
            {
                "chapter_id": int(chapter_id),
                "title": title,
                "chars": len(draft.content) if draft is not None else 0,
                "cited_doc_ids": list(draft.cited_doc_ids) if draft is not None else [],
                "retries": int(retries.get(chapter_id, 0)),
                "poisoned": chapter_id in poisoned,
                "drafted": draft is not None,
            }
        )
    return rows


def errors_summary(
    messages: List[str] | None,
    *,
    max_items: int = 100,
    max_chars: int = 500,
) -> List[str]:
    """Truncate state errors for ledger (no huge dumps)."""
    out: List[str] = []
    for msg in list(messages or [])[:max_items]:
        text = str(msg)
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        out.append(text)
    return out


def phase3_stats_public(state: Mapping[str, Any] | None) -> Optional[Dict[str, Any]]:
    """Copy Phase3Stats keys only (ADR-030) — no node id lists."""
    if state is None:
        return None
    raw = state.get("phase3_stats") or {}
    if not isinstance(raw, Mapping) or not raw:
        return None
    keys = (
        "extracted_nodes",
        "nodes_after_orphan",
        "edges_after_orphan",
        "orphans_dropped",
        "communities_detected",
        "communities_kept",
        "poc_dropped",
        "poc_top_chapters",
        "algorithm",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        if key in raw:
            out[key] = raw[key]
    return out or None


def improvement_snapshot(
    phases: Mapping[str, Any],
    state: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Cross-run comparison spine (ADR-030).

    Aggregates orphan / poison / chapter counts / cost from phase records + state.
    No bodies, triples, or secrets.
    """
    p2 = phases.get("phase2") or {}
    p3 = phases.get("phase3") or {}
    p4 = phases.get("phase4") or {}
    p2u = p2.get("usage") or {}
    p4u = p4.get("usage") or {}
    p3c = p3.get("counts") or {}
    p4c = p4.get("counts") or {}

    poisoned_ids = sorted(
        int(x) for x in list((state or {}).get("poisoned_chapters") or [])
    )
    toc = (state or {}).get("toc") or {}
    drafts = list((state or {}).get("drafts") or [])

    cost2 = float(p2u.get("total_cost_usd") or 0.0)
    cost4 = float(p4u.get("total_cost_usd") or 0.0)
    tok2 = int(p2u.get("total_tokens") or 0)
    tok4 = int(p4u.get("total_tokens") or 0)

    drafts_n = len(drafts) if state is not None else int(p4c.get("drafts") or 0)
    toc_n = len(toc) if state is not None else int(p4c.get("chapters_planned") or 0)

    snap: Dict[str, Any] = {
        "orphans_dropped": int(p3c.get("orphans_dropped") or 0),
        "extracted_nodes": int(p3c.get("extracted_nodes") or 0),
        "nodes_after_orphan": int(p3c.get("nodes_after_orphan") or 0),
        "communities_detected": int(p3c.get("communities_detected") or 0),
        "communities_kept": int(p3c.get("communities_kept") or 0),
        "chapters_in_toc": toc_n,
        "drafts": drafts_n,
        "poisoned_count": len(poisoned_ids),
        "poisoned_chapter_ids": poisoned_ids,
        "cost_usd": {
            "phase2": cost2,
            "phase4": cost4,
            "phase2_plus_4": round(cost2 + cost4, 6),
        },
        "tokens": {
            "phase2": tok2,
            "phase4": tok4,
            "phase2_plus_4": tok2 + tok4,
        },
    }
    # W2-A: Phase 2 graph_health (orphan_ratio / avg weight by provenance)
    p2_health = (p2.get("counts") or {}).get("graph_health")
    if isinstance(p2_health, Mapping) and p2_health:
        snap["phase2_graph_health"] = {
            k: p2_health[k]
            for k in (
                "orphan_ratio",
                "orphan_count",
                "avg_edge_weight",
                "avg_edge_weight_by_provenance",
                "edge_count_by_provenance",
                "total_nodes",
                "total_edges",
            )
            if k in p2_health
        }
    p3_stats = phase3_stats_public(state)
    if p3_stats is not None:
        snap["phase3_stats"] = p3_stats
    return snap


class _TeeTextIO:
    """Write to console stream and run.log (ADR-026 stage 3)."""

    def __init__(self, primary: TextIO, secondary: TextIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, data: str) -> int:
        n = self._primary.write(data)
        try:
            self._secondary.write(data)
            self._secondary.flush()
        except Exception:  # noqa: BLE001 — never break the pipeline for log I/O
            pass
        return n

    def flush(self) -> None:
        self._primary.flush()
        try:
            self._secondary.flush()
        except Exception:  # noqa: BLE001
            pass

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._primary.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._primary, "encoding", "utf-8") or "utf-8"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._primary, name)


@contextmanager
def attach_run_log(run_dir: Path) -> Iterator[Path]:
    """
    Tee stdout/stderr to run_dir/run.log (ADR-026 stage 3).

    Captures what the operator sees (print + console warnings). Does not
    replace errors.log. Callers must not print API keys.
    """
    path = Path(run_dir) / RUN_LOG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = path.open("w", encoding="utf-8", newline="\n")
    old_out, old_err = sys.stdout, sys.stderr
    # Windows cp949 consoles crash on characters like '—' (U+2014) mid-run;
    # the console is a debug surface, so never let encoding kill the pipeline.
    for stream in (old_out, old_err):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    try:
        sys.stdout = _TeeTextIO(old_out, log_fp)  # type: ignore[assignment]
        sys.stderr = _TeeTextIO(old_err, log_fp)  # type: ignore[assignment]
        yield path
    finally:
        sys.stdout = old_out
        sys.stderr = old_err
        try:
            log_fp.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class RunLedger:
    """In-memory meta ledger; JSON is the canonical on-disk form."""

    run_id: str
    run_dir: Path
    started_at: str
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None
    exit_code: Optional[int] = None
    cli: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    control_room: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)
    budget_slices: Dict[str, Any] = field(default_factory=dict)
    graph_slice: Dict[str, Any] = field(default_factory=dict)
    phases: Dict[str, Any] = field(default_factory=dict)
    chapters: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    # ADR-030: orphan / poison / chapter counts / cost for cross-run compare
    improvement: Dict[str, Any] = field(default_factory=dict)
    _t0: datetime = field(default_factory=_utc_now, repr=False, compare=False)

    @classmethod
    def begin(
        cls,
        *,
        cli: Optional[Dict[str, Any]] = None,
        settings_snap: Optional[Dict[str, Any]] = None,
        control_room_snap: Optional[Dict[str, Any]] = None,
        runs_root: Path | None = None,
        when: Optional[datetime] = None,
    ) -> "RunLedger":
        t0 = when or _utc_now()
        run_dir = create_run_directory(runs_root, when=t0)
        return cls(
            run_id=run_dir.name,
            run_dir=run_dir,
            started_at=_iso(t0),
            cli=dict(cli or {}),
            settings=dict(settings_snap or {}),
            control_room=dict(control_room_snap or {}),
            _t0=t0,
        )

    def record_phase(
        self,
        name: str,
        *,
        duration_s: float,
        counts: Optional[Dict[str, Any]] = None,
        usage: object | None = None,
        estimate: Mapping[str, Any] | None = None,
        extras: Optional[Dict[str, Any]] = None,
        persist: bool = True,
    ) -> None:
        """Store Phase meta (ADR-026 stage 2). No bodies / RepoMap text."""
        entry: Dict[str, Any] = {
            "duration_s": round(float(duration_s), 6),
            "counts": dict(counts or {}),
            "usage": usage_public_dict(usage),
            "estimate": estimate_public_dict(estimate),
        }
        if extras:
            entry["extras"] = dict(extras)
        self.phases[name] = entry
        if persist:
            self.write()

    def record_budget_slice(
        self,
        name: str,
        info: Mapping[str, Any],
        *,
        persist: bool = True,
    ) -> None:
        """Budget slice counters only (kept/dropped/stop_reason/…)."""
        allowed = (
            "kept",
            "dropped",
            "stop_reason",
            "cum_tokens",
            "cum_cost_usd",
        )
        self.budget_slices[name] = {k: info[k] for k in allowed if k in info}
        if persist:
            self.write()

    def record_graph_slice(
        self,
        info: Mapping[str, Any],
        *,
        persist: bool = True,
    ) -> None:
        """W3-A scale preflight meta (pass|slice|abort). No graph bodies."""
        allowed = (
            "action",
            "stop_reason",
            "on_exceed",
            "nodes_before",
            "edges_before",
            "nodes_after",
            "edges_after",
            "nodes_dropped",
            "edges_dropped",
            "est_bytes_before",
            "est_bytes_after",
            "est_mb_before",
            "est_mb_after",
            "limits",
        )
        self.graph_slice = {k: info[k] for k in allowed if k in info}
        if persist:
            self.write()

    def record_outcome_meta(
        self,
        state: Mapping[str, Any] | None,
        *,
        codex_paths: Optional[List[Path]] = None,
        archive_dir: Path | None = None,
        checkpoint_path: Path | None = None,
        persist: bool = True,
    ) -> None:
        """Chapters / artifacts / errors / improvement (ADR-026+030). No bodies."""
        if state is not None:
            self.chapters = chapters_meta_from_state(state)
            self.errors = errors_summary(list(state.get("errors") or []))
        self.improvement = improvement_snapshot(self.phases, state)
        arts = dict(self.artifacts)
        if codex_paths is not None:
            arts["codex_files"] = [str(p) for p in codex_paths]
        if archive_dir is not None:
            arts["archive_dir"] = str(archive_dir)
        if checkpoint_path is not None:
            arts["checkpoint"] = str(checkpoint_path)
        arts["ledger"] = str(self.ledger_path)
        self.artifacts = arts
        if persist:
            self.write()

    def finish(self, exit_code: int, *, when: Optional[datetime] = None) -> Path:
        t1 = when or _utc_now()
        self.finished_at = _iso(t1)
        self.duration_s = max(0.0, (t1 - self._t0).total_seconds())
        self.exit_code = int(exit_code)
        return self.write()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": self.duration_s,
            "exit_code": self.exit_code,
            "cli": self.cli,
            "settings": self.settings,
            "control_room": self.control_room,
            "sources": list(self.sources),
            "budget_slices": self.budget_slices,
            "graph_slice": dict(self.graph_slice),
            "phases": self.phases,
            "chapters": list(self.chapters),
            "artifacts": self.artifacts,
            "errors": list(self.errors),
            "improvement": dict(self.improvement),
        }

    def write(self) -> Path:
        path = self.run_dir / LEDGER_FILENAME
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @property
    def ledger_path(self) -> Path:
        return self.run_dir / LEDGER_FILENAME
