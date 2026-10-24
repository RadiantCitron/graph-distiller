"""GraphDistiller compile CLI — single pipeline entry (ADR-053).

Model slugs come from argv only. .env EXTRACTION_MODEL / SYNTHESIS_MODEL
are ignored here. Secrets (API key, RAG_DB_URI) stay in .env.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph_distiller.adapters.source import RdbSourcePull
from graph_distiller.agents.repomap import build_repo_map, chapter_grounding_budget
from graph_distiller.budget import (
    DEFAULT_BUDGET_USD,
    DEFAULT_TOKEN_BUDGET,
    clamp_budget_usd,
    clamp_token_budget,
)
from graph_distiller.cli_ui import (
    confirm_proceed,
    print_line,
    print_table,
    render_cost_table,
    render_db_status_table,
)
from graph_distiller.config import (
    Settings,
    set_runtime_settings,
)
from graph_distiller.cost import count_tokens, estimate_phase2_cost, estimate_phase4_cost
from graph_distiller.observability import configure_pipeline_logging
from graph_distiller.rules import RulesValidationError, load_rules, set_runtime_rules
from graph_distiller.pricing import active_rate_for, fetch_openrouter_rate_book, set_active_rate_book
from graph_distiller.rag_gates import fatal_if_db_mode_ambiguous, phase2_llm_default_enabled
from graph_distiller.rag_skeleton import run_rag_skeleton
from graph_distiller.rag_synthesize import cap_toc_chapters, load_state_from_export, run_rag_synthesize
from graph_distiller.workspace import resolve_workspace


class CompilerArgumentParser(argparse.ArgumentParser):
    """Custom parser that relaxes required model args when --explain or --check-config is given."""

    def parse_args(self, args=None, namespace=None):
        raw = sys.argv[1:] if args is None else list(args)
        if any(a in ("--explain", "--check-config") for a in raw):
            for action in self._actions:
                if action.dest in ("extraction_model", "synthesis_model"):
                    action.required = False
        return super().parse_args(args=args, namespace=namespace)


def build_parser() -> argparse.ArgumentParser:
    p = CompilerArgumentParser(
        description=(
            "GraphDistiller compiler: Phase 1-4. Models MUST be passed on the CLI "
            "(env model slugs are ignored)."
        )
    )
    p.add_argument(
        "--extraction-model",
        required=True,
        help="Phase 2 extract + Phase 4 titles. Not read from .env.",
    )
    p.add_argument(
        "--synthesis-model",
        required=True,
        help="Phase 4 chapter body. Not read from .env.",
    )
    p.add_argument(
        "--embedding-model",
        default=None,
        help="Optional Phase 1 embedder slug (default: Settings/env embed model).",
    )
    p.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Workspace directory containing workspace.toml (single source of truth).",
    )
    p.add_argument(
        "--explain",
        action="store_true",
        help="Print resolved configuration, source layers, and policy provenance, then exit.",
    )
    p.add_argument(
        "--check-config",
        action="store_true",
        help="Validate workspace configuration and policy schemas, then exit.",
    )
    p.add_argument(
        "--source",
        choices=("db", "local"),
        default=None,
        help="db = ragdb (default without --workspace). local = data/raw_memos/ or workspace input.",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help="--source db: sample n_jo / m_qna (required with this flag).",
    )
    p.add_argument(
        "--full-db",
        action="store_true",
        help="--source db: explicit full corpus (budget-sliced). Not a silent default.",
    )
    p.add_argument("--n-jo", type=int, default=None)
    p.add_argument("--m-qna", type=int, default=None)
    p.add_argument(
        "--llm",
        action="store_true",
        default=phase2_llm_default_enabled(),
        help="Phase 2 LLM on the db graph (opt-in; default cites-only).",
    )
    p.add_argument(
        "--budget-usd",
        type=float,
        default=DEFAULT_BUDGET_USD,
    )
    p.add_argument(
        "--token-limit",
        type=int,
        default=DEFAULT_TOKEN_BUDGET,
    )
    p.add_argument(
        "--max-chapters",
        type=int,
        default=0,
        help="Phase 4 community cap after Phase 3. 0 = all kept communities.",
    )
    p.add_argument(
        "--poc-top",
        type=int,
        default=None,
        help=(
            "ADR-055 opt-in: Phase 3 keep only top-N communities by node count. "
            "Omit to use POC_TOP_CHAPTERS from env (default 0 = no cap)."
        ),
    )
    p.add_argument("--uri", default=None, help="Override RAG_DB_URI (never logged).")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--from-checkpoints", action="store_true")
    p.add_argument("-y", "--yes", action="store_true")
    return p


def validate_args(args: argparse.Namespace) -> Optional[str]:
    is_inspection = bool(getattr(args, "explain", False) or getattr(args, "check_config", False))

    if not is_inspection:
        if not str(args.extraction_model or "").strip():
            return "--extraction-model is required (compiler ignores .env model slugs)."
        if not str(args.synthesis_model or "").strip():
            return "--synthesis-model is required (compiler ignores .env model slugs)."

    # D4 & D8: Workspace handling
    if getattr(args, "workspace", None) is not None:
        if args.source == "db":
            return "--workspace is not supported with --source db (D4: local source only)."
        ws_path = Path(args.workspace)
        if not ws_path.is_dir():
            return f"Workspace directory not found: {ws_path}"
        ws_file = ws_path / "workspace.toml"
        if not ws_file.is_file():
            return f"Workspace configuration file not found: {ws_file}"
        try:
            cli_overlay: Dict[str, Any] = {}
            if getattr(args, "output_dir", None):
                cli_overlay["output_dir"] = args.output_dir
            if getattr(args, "poc_top", None) is not None:
                cli_overlay["poc_top_chapters"] = args.poc_top
            resolve_workspace(ws_path, cli_overlay=cli_overlay)
        except Exception as exc:
            return f"Workspace validation error: {exc}"

        if args.source is None:
            args.source = "local"
            args._source_defaulted_by_workspace = True
    else:
        if args.source is None:
            args.source = "db"

    if is_inspection:
        return None

    if args.source == "local":
        if args.sample or args.full_db:
            return "--sample / --full-db apply only to --source db."
        return None
    return fatal_if_db_mode_ambiguous(
        sample=bool(args.sample),
        full_db=bool(args.full_db),
        n_jo=args.n_jo,
        m_qna=args.m_qna,
        llm=bool(args.llm),
    )


def settings_from_cli(args: argparse.Namespace) -> Settings:
    """Build Settings from env knobs + CLI models. Env model slugs overwritten."""
    base = Settings.from_env(models_required=False)
    extraction = str(args.extraction_model).strip()
    synthesis = str(args.synthesis_model).strip()
    embed = (
        str(args.embedding_model).strip()
        if args.embedding_model
        else base.embedding_model
    )
    cfg = replace(
        base,
        extraction_model=extraction,
        synthesis_model=synthesis,
        embedding_model=embed or base.embedding_model,
    )
    if args.poc_top is not None:
        cfg = replace(cfg, poc_top_chapters=max(0, int(args.poc_top)))

    if getattr(args, "workspace", None) is not None:
        cli_overlay: Dict[str, Any] = {}
        if getattr(args, "output_dir", None):
            cli_overlay["output_dir"] = args.output_dir
        if getattr(args, "poc_top", None) is not None:
            cli_overlay["poc_top_chapters"] = args.poc_top
        resolved = resolve_workspace(
            workspace_dir=Path(args.workspace),
            cli_overlay=cli_overlay,
        )
        cfg = replace(
            cfg,
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

    set_runtime_settings(cfg)
    return cfg


def _resolve_uri(cli_uri: str | None, cfg: Settings) -> str:
    uri = (cli_uri or cfg.rag_db_uri or "").strip()
    if not uri:
        print(
            "[FATAL] RAG_DB_URI unset. Add it to .env (never commit) or pass --uri.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return uri


def _sum_repo_map_tokens(state) -> int:
    toc = state.get("toc") or {}
    nodes = list(state.get("nodes") or [])
    chunks = list(state.get("chunks") or [])
    total = 0
    for chapter_id, member_ids in sorted(toc.items()):
        repo_map = build_repo_map(
            chapter_id=chapter_id,
            member_ids=list(member_ids),
            nodes=nodes,
            chunks=chunks,
            grounding_budget=chapter_grounding_budget(
                list(member_ids), nodes, chunks
            ),
        )
        total += count_tokens(repo_map)
    return total


def _print_est_table(
    *,
    est2: Optional[Dict[str, Any]],
    est4: Optional[Dict[str, Any]],
) -> None:
    print_table(
        render_cost_table(
            est2=est2,
            est4=est4,
            title="Estimated tokens / cost (EST)",
        )
    )


def _print_actual_table(
    *,
    est2: Optional[Dict[str, Any]],
    est4: Optional[Dict[str, Any]],
    actual2: Optional[Dict[str, Any]],
    actual4: Optional[Dict[str, Any]],
) -> None:
    print_table(
        render_cost_table(
            est2=est2,
            est4=est4,
            actual2=actual2,
            actual4=actual4,
            title="EST vs Actual tokens / cost",
        )
    )


def _run_local(args: argparse.Namespace) -> int:
    from graph_distiller.local_pipeline import run_pipeline

    return run_pipeline(
        auto_confirm=bool(args.yes),
        budget_usd=float(args.budget_usd),
        token_budget=int(args.token_limit),
        runs_root=args.output_dir,
        from_checkpoints=bool(args.from_checkpoints),
    )


def _run_db(args: argparse.Namespace, cfg: Settings) -> int:
    source = RdbSourcePull()
    uri = _resolve_uri(args.uri, cfg)
    counts = source.preview_counts(uri)
    mode = "full-db" if args.full_db else "sample"
    target_jo = (
        int(counts.get("labor_law_jo") or 0) if args.full_db else int(args.n_jo)
    )
    target_qna = (
        int(counts.get("labor_qna_cases") or 0) if args.full_db else int(args.m_qna)
    )
    print_table(
        render_db_status_table(
            counts,
            target_jo=target_jo,
            target_qna=target_qna,
            mode=mode,
        )
    )
    if args.full_db:
        if target_jo < 1:
            print("[FATAL] --full-db but labor_law has no 조 rows.", file=sys.stderr)
            return 2
    else:
        live_jo = int(counts.get("labor_law_jo") or 0)
        live_qna = int(counts.get("labor_qna_cases") or 0)
        if args.n_jo > live_jo or args.m_qna > live_qna:
            print(
                f"[FATAL] Sample exceeds DB: n_jo<={live_jo} m_qna<={live_qna}.",
                file=sys.stderr,
            )
            return 2

    if not confirm_proceed(auto_confirm=bool(args.yes), prompt="Proceed with DB compile? [Y/n]"):
        print_line("Aborted.")
        return 1

    budget_usd = clamp_budget_usd(float(args.budget_usd))
    _ = clamp_token_budget(int(args.token_limit))
    rate_book = fetch_openrouter_rate_book(
        [cfg.extraction_model, cfg.synthesis_model, cfg.embedding_model],
        base_url=cfg.openrouter_base_url,
    )
    set_active_rate_book(rate_book)
    for line in rate_book.lines_for(
        [cfg.extraction_model, cfg.synthesis_model, cfg.embedding_model]
    ):
        print_line(line)

    result = run_rag_skeleton(
        n_jo=args.n_jo,
        m_qna=args.m_qna,
        rag_db_uri=uri,
        settings=cfg,
        llm=bool(args.llm),
        budget_usd=budget_usd,
        export=True,
        export_root=cfg.rag_rebuild_dir,
        full_db=bool(args.full_db),
    )
    print_line(
        f"[compiler] skeleton nodes={len(result.nodes)} edges={len(result.edges)} "
        f"llm_mode={result.llm_mode} export={result.export_dir}"
    )

    est2: Optional[Dict[str, Any]] = None
    if args.llm:
        est2 = estimate_phase2_cost(
            list(result.chunks),
            cfg.extraction_model,
            rate=active_rate_for(cfg.extraction_model),
        )
    else:
        est2 = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "chunk_count": len(result.chunks),
        }

    if result.export_dir is None:
        print("[FATAL] export dir missing; cannot synthesize.", file=sys.stderr)
        return 2

    state = load_state_from_export(result.export_dir)
    max_chapters = int(args.max_chapters)
    if max_chapters <= 0:
        max_chapters = max(1, len(state.get("toc") or {}) or 1)
    state = cap_toc_chapters(state, max_chapters=max_chapters)
    repo_tokens = _sum_repo_map_tokens(state)
    est4 = estimate_phase4_cost(
        n_chapters=len(state.get("toc") or {}),
        repo_map_tokens=repo_tokens,
        synthesis_model=cfg.synthesis_model,
        planner_model=cfg.extraction_model,
        synthesis_rate=active_rate_for(cfg.synthesis_model),
        planner_rate=active_rate_for(cfg.extraction_model),
    )
    _print_est_table(est2=est2, est4=est4)
    if not confirm_proceed(
        auto_confirm=bool(args.yes), prompt="Proceed with Phase 4 synthesis? [Y/n]"
    ):
        print_line("Aborted before Phase 4.")
        return 1

    syn = run_rag_synthesize(result.export_dir, max_chapters=max_chapters)
    actual2 = dict(result.phase2_usage) if result.phase2_usage else {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_usd": 0.0,
    }
    actual4 = dict(syn.phase4_usage)
    _print_actual_table(est2=est2, est4=est4, actual2=actual2, actual4=actual4)
    print_line(
        f"[compiler] drafts={syn.drafts} poisoned={syn.poisoned} "
        f"codex={syn.codex_dir}"
    )
    return 0 if syn.drafts > 0 or not syn.poisoned else 1


def run_check_config(args: argparse.Namespace) -> int:
    """Validate workspace / default configuration and policy schemas without running pipeline."""
    import json
    import tomllib
    from graph_distiller.rules import (
        CanonicalizeRules,
        EvalScoringRules,
        PlannerRules,
        RepomapRules,
        RulesValidationError,
        VerifierRules,
        load_rules,
    )
    from graph_distiller.workspace import LEXICON_FILENAME, PROMPT_FILES, RULE_FILES

    if getattr(args, "workspace", None) is not None:
        ws_path = Path(args.workspace)
        cli_overlay: Dict[str, Any] = {}
        if getattr(args, "output_dir", None):
            cli_overlay["output_dir"] = args.output_dir
        if getattr(args, "poc_top", None) is not None:
            cli_overlay["poc_top_chapters"] = args.poc_top
        try:
            resolved = resolve_workspace(ws_path, cli_overlay=cli_overlay)
        except Exception as exc:
            print(f"[FATAL] Workspace validation error: {exc}", file=sys.stderr)
            return 2

        # D11: Validate input directory exists
        if not resolved.raw_memos_dir.is_dir():
            print(f"[FATAL] Input directory not found: {resolved.raw_memos_dir}", file=sys.stderr)
            return 2

        # Validate policy rule files
        rule_schemas = {
            "phase2_canonicalize.toml": CanonicalizeRules,
            "phase4_repomap.toml": RepomapRules,
            "phase4_planner.toml": PlannerRules,
            "phase4_verifier.toml": VerifierRules,
            "eval_scoring.toml": EvalScoringRules,
        }
        for r_name, schema_cls in rule_schemas.items():
            pf = resolved.policy_files.get(r_name)
            if pf and pf.resolved_path.is_file():
                try:
                    raw = tomllib.loads(pf.resolved_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    print(f"[FATAL] {pf.resolved_path}: invalid TOML: {exc}", file=sys.stderr)
                    return 2
                try:
                    schema_cls.model_validate(raw)
                except Exception as exc:
                    print(f"[FATAL] {pf.resolved_path}: {exc}", file=sys.stderr)
                    return 2

        # Validate prompts if present
        for p_name in PROMPT_FILES:
            pf = resolved.policy_files.get(p_name)
            if pf and pf.resolved_path.is_file():
                try:
                    pf.resolved_path.read_text(encoding="utf-8")
                except Exception as exc:
                    print(f"[FATAL] {pf.resolved_path}: failed to read: {exc}", file=sys.stderr)
                    return 2

        # Validate alias lexicon if present
        lex_pf = resolved.policy_files.get(LEXICON_FILENAME)
        if lex_pf and lex_pf.resolved_path.is_file():
            try:
                raw = json.loads(lex_pf.resolved_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    print(f"[FATAL] {lex_pf.resolved_path}: alias lexicon root must be a JSON object", file=sys.stderr)
                    return 2
            except Exception as exc:
                print(f"[FATAL] {lex_pf.resolved_path}: invalid JSON: {exc}", file=sys.stderr)
                return 2

        print(f"[check-config] OK: workspace '{resolved.name}' ({ws_path}) configuration and schemas are valid.")
        return 0

    else:
        # Default non-workspace mode
        cfg = Settings.from_env(models_required=False)
        try:
            load_rules(cfg.rules_dir)
        except RulesValidationError as exc:
            print(f"[FATAL] {exc}", file=sys.stderr)
            return 2

        if cfg.canon_alias_lexicon_path and cfg.canon_alias_lexicon_path.is_file():
            try:
                raw = json.loads(cfg.canon_alias_lexicon_path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    print(f"[FATAL] {cfg.canon_alias_lexicon_path}: alias lexicon root must be a JSON object", file=sys.stderr)
                    return 2
            except Exception as exc:
                print(f"[FATAL] {cfg.canon_alias_lexicon_path}: invalid JSON: {exc}", file=sys.stderr)
                return 2

        print("[check-config] OK: default configuration and schemas are valid.")
        return 0


def run_explain(args: argparse.Namespace) -> int:
    """Print all resolved values, source layers, policy file provenance, masked secrets, and exit."""
    import os
    from graph_distiller.cli_ui import render_explain_tables, print_table
    from graph_distiller.workspace import KNOB_ENV_TO_FIELD

    field_to_env = {v: k for k, v in KNOB_ENV_TO_FIELD.items()}

    if getattr(args, "workspace", None) is not None:
        ws_path = Path(args.workspace)
        cli_overlay: Dict[str, Any] = {}
        if getattr(args, "output_dir", None):
            cli_overlay["output_dir"] = args.output_dir
        if getattr(args, "poc_top", None) is not None:
            cli_overlay["poc_top_chapters"] = args.poc_top
        resolved = resolve_workspace(ws_path, cli_overlay=cli_overlay)

        general_info: Dict[str, tuple[str, str]] = {
            "Workspace Name": (resolved.name, "workspace"),
            "Workspace Dir": (str(resolved.workspace_dir), "workspace"),
        }

        # Policy files (D5)
        policy_files = []
        for pf in resolved.policy_files.values():
            status = "present" if pf.resolved_path.is_file() else "in-code default"
            policy_files.append({
                "file": pf.file_name,
                "layer": pf.source,
                "status": status,
                "path": str(pf.resolved_path),
            })

        # Knobs
        knobs = []
        for field in sorted(KNOB_ENV_TO_FIELD.values()):
            val = resolved.knobs.get(field)
            prov = resolved.provenance.get(field)
            src = prov.source if prov else "engine_default"
            note = ""
            layer_display = src
            if src == "env":
                layer_display = "env (deprecated)"
                env_k = field_to_env.get(field, "")
                note = f"Deprecated env: {env_k}"
            elif src == "cli":
                note = "CLI override"
            knobs.append({
                "knob": field,
                "value": str(val),
                "layer": layer_display,
                "note": note,
            })

        # Paths
        paths = []
        path_fields = [
            ("raw_memos_dir (input)", resolved.raw_memos_dir, resolved.provenance.get("raw_memos_dir")),
            ("checkpoint_dir (work)", resolved.checkpoint_dir, resolved.provenance.get("checkpoint_dir")),
            ("codex_output_dir (output)", resolved.codex_output_dir, resolved.provenance.get("codex_output_dir")),
            ("errors_log_path", resolved.errors_log_path, resolved.provenance.get("errors_log_path")),
            ("rag_rebuild_dir", resolved.rag_rebuild_dir, resolved.provenance.get("rag_rebuild_dir")),
            ("runs_dir", resolved.runs_dir, resolved.provenance.get("runs_dir")),
            ("prompts_dir", resolved.prompts_dir, None),
            ("rules_dir", resolved.rules_dir, None),
            ("canon_alias_lexicon_path", resolved.canon_alias_lexicon_path, None),
        ]
        for name, p_val, prov in path_fields:
            src = prov.source if prov else "workspace"
            paths.append({
                "name": name,
                "path": str(p_val),
                "layer": src,
            })

    else:
        # Default non-workspace mode
        cfg = Settings.from_env(models_required=False)
        general_info = {
            "Mode": ("default (non-workspace)", "engine_default"),
        }

        # Policy files
        from graph_distiller.workspace import LEXICON_FILENAME, PROMPT_FILES, RULE_FILES
        policy_files = []
        for rf in RULE_FILES:
            p = cfg.rules_dir / rf
            policy_files.append({
                "file": rf,
                "layer": "env" if "RULES_DIR" in os.environ else "engine_default",
                "status": "present" if p.is_file() else "in-code default",
                "path": str(p),
            })
        for pf in PROMPT_FILES:
            p = cfg.prompts_dir / pf
            policy_files.append({
                "file": pf,
                "layer": "env" if "PROMPTS_DIR" in os.environ else "engine_default",
                "status": "present" if p.is_file() else "in-code default",
                "path": str(p),
            })
        lex_p = cfg.canon_alias_lexicon_path
        policy_files.append({
            "file": LEXICON_FILENAME,
            "layer": "env" if "CANON_ALIAS_LEXICON_PATH" in os.environ else "engine_default",
            "status": "present" if lex_p.is_file() else "in-code default",
            "path": str(lex_p),
        })

        # Knobs
        knobs = []
        for field in sorted(KNOB_ENV_TO_FIELD.values()):
            val = getattr(cfg, field, None)
            env_k = field_to_env.get(field, "")
            if env_k in os.environ and os.environ[env_k]:
                layer_display = "env (deprecated)"
                note = f"Deprecated env: {env_k}"
            elif field == "poc_top_chapters" and getattr(args, "poc_top", None) is not None:
                layer_display = "cli"
                note = "CLI override"
            else:
                layer_display = "engine_default"
                note = ""
            knobs.append({
                "knob": field,
                "value": str(val),
                "layer": layer_display,
                "note": note,
            })

        # Paths
        env_path_map = {
            "raw_memos_dir (input)": (cfg.raw_memos_dir, "RAW_MEMOS_DIR"),
            "checkpoint_dir (work)": (cfg.checkpoint_dir, "CHECKPOINT_DIR"),
            "codex_output_dir (output)": (cfg.codex_output_dir, "CODEX_OUTPUT_DIR"),
            "errors_log_path": (cfg.errors_log_path, "ERRORS_LOG_PATH"),
            "rag_rebuild_dir": (cfg.rag_rebuild_dir, "RAG_REBUILD_DIR"),
            "runs_dir": (cfg.runs_dir, ""),
            "prompts_dir": (cfg.prompts_dir, "PROMPTS_DIR"),
            "rules_dir": (cfg.rules_dir, "RULES_DIR"),
            "canon_alias_lexicon_path": (cfg.canon_alias_lexicon_path, "CANON_ALIAS_LEXICON_PATH"),
        }
        paths = []
        for name, (p_val, env_k) in env_path_map.items():
            src = "env" if env_k and env_k in os.environ else "engine_default"
            paths.append({
                "name": name,
                "path": str(p_val),
                "layer": src,
            })

    # Secrets & Models (strictly masked!)
    secrets_models = [
        {
            "item": "OPENROUTER_API_KEY",
            "status": "[set]" if bool(os.environ.get("OPENROUTER_API_KEY")) else "[not set]",
            "source": "env" if bool(os.environ.get("OPENROUTER_API_KEY")) else "-",
        },
        {
            "item": "RAG_DB_URI",
            "status": "[set]" if bool(getattr(args, "uri", None) or os.environ.get("RAG_DB_URI")) else "[not set]",
            "source": "cli" if getattr(args, "uri", None) else ("env" if os.environ.get("RAG_DB_URI") else "-"),
        },
        {
            "item": "OPENROUTER_BASE_URL",
            "status": "[set]" if bool(os.environ.get("OPENROUTER_BASE_URL")) else "[default]",
            "source": "env" if bool(os.environ.get("OPENROUTER_BASE_URL")) else "engine_default",
        },
        {
            "item": "EXTRACTION_MODEL",
            "status": str(args.extraction_model) if getattr(args, "extraction_model", None) else "[not set (CLI required for compile)]",
            "source": "cli" if getattr(args, "extraction_model", None) else "-",
        },
        {
            "item": "SYNTHESIS_MODEL",
            "status": str(args.synthesis_model) if getattr(args, "synthesis_model", None) else "[not set (CLI required for compile)]",
            "source": "cli" if getattr(args, "synthesis_model", None) else "-",
        },
        {
            "item": "EMBEDDING_MODEL",
            "status": str(getattr(args, "embedding_model", None) or os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small"),
            "source": "cli" if getattr(args, "embedding_model", None) else ("env" if os.environ.get("EMBEDDING_MODEL") else "engine_default"),
        },
    ]

    for table in render_explain_tables(
        general_info=general_info,
        policy_files=policy_files,
        knobs=knobs,
        paths=paths,
        secrets_models=secrets_models,
    ):
        print_table(table)

    return 0


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reason = validate_args(args)
    if reason:
        print(f"[FATAL] {reason}", file=sys.stderr)
        return 2

    if getattr(args, "check_config", False):
        rc = run_check_config(args)
        if rc != 0 or not getattr(args, "explain", False):
            return rc

    if getattr(args, "explain", False):
        return run_explain(args)

    if getattr(args, "workspace", None) is not None:
        cli_overlay: Dict[str, Any] = {}
        if getattr(args, "output_dir", None):
            cli_overlay["output_dir"] = args.output_dir
        if getattr(args, "poc_top", None) is not None:
            cli_overlay["poc_top_chapters"] = args.poc_top
        resolved = resolve_workspace(
            workspace_dir=Path(args.workspace),
            cli_overlay=cli_overlay,
        )
        print(f"[workspace] Workspace: {resolved.name} ({resolved.workspace_dir})")
        if getattr(args, "_source_defaulted_by_workspace", False):
            print("[workspace] --source omitted with --workspace; defaulting to 'local'")

    cfg = settings_from_cli(args)
    configure_pipeline_logging(cfg.errors_log_path)
    try:
        cfg.ensure_directories()
    except FileNotFoundError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    try:
        set_runtime_rules(load_rules(cfg.rules_dir))
    except RulesValidationError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    if args.source == "local":
        return _run_local(args)
    return _run_db(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
