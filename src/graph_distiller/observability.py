"""Pipeline observability: Rich progress + token/cost summaries."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from graph_distiller.cost import (
    CostStatus,
    ModelRate,
    calculate_cost,
    format_cost_status,
)
from graph_distiller.pricing import active_rate_for

logger = logging.getLogger(__name__)

_FILE_HANDLER_NAME = "graph_distiller.errors_file"
_CONSOLE_HANDLER_NAME = "graph_distiller.console"

PHASE_TQDM_BAR_FORMAT = (
    "{desc}: {n_fmt}/{total_fmt} [{bar}] ETA: {remaining}{postfix}"
)  # kept for log/docs; bars are rich.progress (ADR-053)


def format_token_count(n: int) -> str:
    """Compact token count for tqdm postfix (e.g. 15320 → '15.3K')."""
    n = max(0, int(n))
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        value = n / 1000.0
        text = f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{text}K"
    value = n / 1_000_000.0
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return f"{text}M"


def tokens_postfix(prompt_tokens: int, completion_tokens: int) -> str:
    return (
        f"Tokens: {format_token_count(prompt_tokens)} in / "
        f"{format_token_count(completion_tokens)} out"
    )


@dataclass(frozen=True)
class PhaseUsage:
    """Snapshot from langchain get_openai_callback + OpenRouter RateBook (ADR-031)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_status: CostStatus = "unknown"

    @classmethod
    def from_callback(
        cls,
        cb: object,
        model: Optional[str] = None,
        *,
        rate: Optional[ModelRate] = None,
    ) -> "PhaseUsage":
        """
        langchain's get_openai_callback() only prices literal OpenAI model name
        strings; OpenRouter slugs usually yield total_cost=0 even when tokens
        are accurate.

        Prefer the run-scoped OpenRouter ``RateBook`` (or an explicit ``rate``).
        Never invent a price: unknown → status UNKNOWN and $0.
        """
        prompt_tokens = int(getattr(cb, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(cb, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(cb, "total_tokens", 0) or 0)
        cb_cost = float(getattr(cb, "total_cost", 0.0) or 0.0)

        resolved = rate
        if resolved is None and model:
            resolved = active_rate_for(model)
        if resolved is None:
            resolved = ModelRate.unknown()

        # Trust callback only when it already priced and we have no better book.
        if cb_cost > 0.0 and resolved.status == "unknown":
            return cls(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                total_cost_usd=cb_cost,
                cost_status="known",
            )

        cost_usd = calculate_cost(
            model or "", prompt_tokens, completion_tokens, rate=resolved
        )
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            total_cost_usd=cost_usd,
            cost_status=resolved.status,
        )

    @property
    def postfix(self) -> str:
        return tokens_postfix(self.prompt_tokens, self.completion_tokens)

    @property
    def cost_postfix(self) -> str:
        """Live progress-bar postfix: tokens + FREE | UNKNOWN | $…"""
        label = format_cost_status(self.total_cost_usd, self.cost_status)
        return f"{self.postfix} | {label}"

    def summary_line(self, phase_label: str) -> str:
        label = format_cost_status(self.total_cost_usd, self.cost_status)
        return (
            f"[{phase_label} tokens] "
            f"input={self.prompt_tokens}  output={self.completion_tokens}  "
            f"total={self.total_tokens}  "
            f"est_cost={label}"
        )


class PhaseProgress:
    """tqdm-shaped wrapper over rich.progress (brown/gray bar, ETA)."""

    def __init__(self, total: int, *, desc: str, unit: str) -> None:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeRemainingColumn,
        )

        from graph_distiller.cli_ui import BAR_COMPLETE, BAR_REMAIN, console

        self.unit = unit
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(
                bar_width=None,
                style=BAR_REMAIN,
                complete_style=BAR_COMPLETE,
                finished_style=BAR_COMPLETE,
                pulse_style=BAR_REMAIN,
            ),
            TaskProgressColumn(),
            TimeRemainingColumn(compact=True, elapsed_when_finished=True),
            TextColumn("{task.fields[postfix]}", style="dim"),
            console=console,
            transient=False,
            refresh_per_second=4,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            desc, total=max(0, int(total)), postfix=""
        )

    def update(self, n: int = 1) -> None:
        self._progress.update(self._task_id, advance=n)

    def set_postfix_str(self, s: str) -> None:
        self._progress.update(self._task_id, postfix=str(s or ""))

    def close(self) -> None:
        self._progress.stop()


def make_phase_progress(total: int, *, desc: str, unit: str) -> PhaseProgress:
    """In-place progress bar: Phase name, brown/gray bar, ETA."""
    return PhaseProgress(total, desc=desc, unit=unit)


def make_phase_tqdm(total: int, *, desc: str, unit: str) -> PhaseProgress:
    """Deprecated alias — tqdm is gone; Rich progress is the bar."""
    return make_phase_progress(total, desc=desc, unit=unit)


def configure_pipeline_logging(errors_log_path: Path) -> None:
    """
    Console: WARNING+ only (keeps tqdm intact).
    errors.log: INFO+ for audit without breaking the bar.
    """
    errors_log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler()
    console.set_name(_CONSOLE_HANDLER_NAME)
    console.setLevel(logging.WARNING)
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    file_handler = logging.FileHandler(errors_log_path, encoding="utf-8")
    file_handler.set_name(_FILE_HANDLER_NAME)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    for noisy in ("httpx", "httpcore", "openai", "langchain", "langgraph"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@contextmanager
def quiet_console_logs() -> Iterator[None]:
    """During tqdm loops, hide console logging entirely; file logging stays."""
    root = logging.getLogger()
    console_handlers = [
        h
        for h in root.handlers
        if getattr(h, "name", "") == _CONSOLE_HANDLER_NAME
        or (
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        )
    ]
    previous = [h.level for h in console_handlers]
    try:
        for h in console_handlers:
            h.setLevel(logging.CRITICAL + 1)
        yield
    finally:
        for h, level in zip(console_handlers, previous):
            h.setLevel(level)


def emit_loop_error(message: str, level: int = logging.ERROR) -> None:
    """Progress-safe error line + file logger (does not break the bar)."""
    from graph_distiller.cli_ui import console

    console.print(message, overflow="ignore")
    logger.log(level, message)
