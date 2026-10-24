"""Unit: Rich CLI tables render without secrets ($0)."""

from __future__ import annotations

import pytest

from graph_distiller.cli_ui import render_cost_table, render_db_status_table

pytestmark = pytest.mark.unit


def test_db_status_table_has_targets() -> None:
    table = render_db_status_table(
        {
            "labor_law_rows": 366,
            "labor_law_jo": 89,
            "labor_qna_rows": 302,
            "labor_qna_cases": 295,
        },
        target_jo=2,
        target_qna=2,
        mode="sample",
    )
    assert table.row_count == 4
    assert "RAG DB" in (table.title or "")


def test_cost_table_est_and_actual_rows() -> None:
    table = render_cost_table(
        est2={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "cost_usd": 0.01},
        actual2={"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14, "total_cost_usd": 0.02},
        est4={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150, "cost_usd": 0.03},
        actual4=None,
    )
    assert table.row_count == 4
    assert "EST" in (table.title or "") or "cost" in (table.title or "").lower()
