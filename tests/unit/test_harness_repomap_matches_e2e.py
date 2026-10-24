"""Unit: tests/e2e/run_e2e_eval.py re-exports harness.collectors.repomap's compression math.

ADR-061 refactor: run_e2e_eval.py no longer keeps its own copy of
compute_compression_ratio_pct / _phase1_source_tokens / _collect_repomap_metrics
— it imports them from harness.collectors.repomap, so the paid E2E judge and
the $0 audit harness measure Prompt Diet identically by construction. This
test guards that re-export (same function object survives future refactors)
rather than comparing two independently-defined implementations.

Importing e2e.run_e2e_eval pulls in graph_distiller.local_pipeline, whose module body
attaches a FileHandler onto output/errors.log as an import-time side effect
(see harness/collectors/repomap.py's docstring for why the harness itself
never imports run_e2e_eval). The autouse `_no_file_logging_leaks_into_tracked_errors_log`
fixture in tests/conftest.py strips it after this test.
"""

from __future__ import annotations

import pytest

from harness.collectors.repomap import compute_compression_ratio_pct as harness_ratio

pytestmark = pytest.mark.unit


def test_compression_ratio_formula_matches_e2e_harness() -> None:
    from e2e.run_e2e_eval import compute_compression_ratio_pct as e2e_ratio

    assert e2e_ratio is harness_ratio  # literally the same function, imported not duplicated

    for phase1, repo in [(0, 0), (1000, 0), (1000, 500), (1000, 2000), (100, 100)]:
        assert harness_ratio(phase1, repo) == e2e_ratio(phase1, repo)
