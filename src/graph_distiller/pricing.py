"""OpenRouter model pricing — fetch once per run, then reuse (ADR-031).

Not in cost.py: this module may hit the network. cost.py stays pure math.
Public catalog ``GET {base}/models`` exposes ``pricing.prompt`` /
``pricing.completion`` as USD **per token**. We convert to USD/1M for
``cost.ModelRate``. No API key required for the public list.

House rule: try once; on failure or missing slug → ``unknown`` (never invent
a stale local table as truth). Both rates 0 → ``free``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from graph_distiller.cost import ModelRate, rate_from_per_1m

logger = logging.getLogger(__name__)

DEFAULT_MODELS_PATH = "/models"
DEFAULT_TIMEOUT_S = 15.0

_active: Optional["RateBook"] = None


@dataclass(frozen=True)
class RateBook:
    """Immutable snapshot of rates known at pipeline start."""

    by_id: Mapping[str, ModelRate] = field(default_factory=dict)
    source: str = "empty"
    error: Optional[str] = None

    def rate_for(self, model_id: str) -> ModelRate:
        key = (model_id or "").strip()
        if not key:
            return ModelRate.unknown()
        found = self.by_id.get(key)
        if found is not None:
            return found
        return ModelRate.unknown()

    def lines_for(self, model_ids: Sequence[str]) -> List[str]:
        lines: List[str] = []
        if self.error:
            lines.append(f"[Pricing] fetch failed: {self.error} → UNKNOWN")
        elif self.source:
            lines.append(f"[Pricing] source={self.source}")
        seen = set()
        for mid in model_ids:
            mid = (mid or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            lines.append(f"[Pricing] {mid}: {self.rate_for(mid).label()}")
        return lines


def set_active_rate_book(book: RateBook) -> None:
    """Install the run-scoped book (main calls once after fetch)."""
    global _active
    _active = book


def get_active_rate_book() -> RateBook:
    return _active if _active is not None else RateBook()


def active_rate_for(model_id: str) -> ModelRate:
    return get_active_rate_book().rate_for(model_id)


def _per_token_str_to_1m(raw: object) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(str(raw).strip()) * 1_000_000.0
    except (TypeError, ValueError):
        return None


def rates_from_models_payload(
    payload: Mapping[str, object],
    *,
    wanted: Optional[Iterable[str]] = None,
) -> Dict[str, ModelRate]:
    """
    Parse OpenRouter ``/models`` JSON into ``ModelRate`` map.

    If ``wanted`` is set, only those ids are returned (missing → omitted;
    caller treats omit as unknown via ``rate_for``).
    """
    want = {w.strip() for w in (wanted or []) if w and str(w).strip()}
    data = payload.get("data")
    if not isinstance(data, list):
        return {}

    out: Dict[str, ModelRate] = {}
    for item in data:
        if not isinstance(item, Mapping):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        if want and mid not in want:
            continue
        pricing = item.get("pricing")
        if not isinstance(pricing, Mapping):
            out[mid] = ModelRate.unknown()
            continue
        in_1m = _per_token_str_to_1m(pricing.get("prompt"))
        out_1m = _per_token_str_to_1m(pricing.get("completion"))
        if in_1m is None or out_1m is None:
            out[mid] = ModelRate.unknown()
            continue
        out[mid] = rate_from_per_1m(in_1m, out_1m)
    return out


def fetch_openrouter_rate_book(
    model_ids: Sequence[str],
    *,
    base_url: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> RateBook:
    """
    One HTTP GET to ``{base_url}/models``. Soft-fail → empty book + error.
    """
    base = (base_url or "").rstrip("/")
    url = f"{base}{DEFAULT_MODELS_PATH}"
    wanted = [m.strip() for m in model_ids if m and str(m).strip()]
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "graph_distiller-pricing/1", "Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return RateBook(error="models payload not an object", source=url)
        by_id = rates_from_models_payload(payload, wanted=wanted or None)
        # Ensure every wanted id appears (unknown if absent from catalog)
        filled = dict(by_id)
        for mid in wanted:
            filled.setdefault(mid, ModelRate.unknown())
        return RateBook(by_id=filled, source=url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        logger.warning("OpenRouter pricing fetch failed: %s", exc)
        filled = {mid: ModelRate.unknown() for mid in wanted}
        return RateBook(by_id=filled, source=url, error=str(exc))
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
        logger.warning("OpenRouter pricing parse failed: %s", exc)
        filled = {mid: ModelRate.unknown() for mid in wanted}
        return RateBook(by_id=filled, source=url, error=str(exc))
