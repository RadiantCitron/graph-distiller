"""LLM-as-a-Judge output schema — shared by harness.collectors.judge and
tests/e2e/run_e2e_eval.py so both parse the same structured-output shape."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    compression_score: int = Field(
        description=(
            "How extremely tokens were reduced from Phase1 source to RepoMap "
            "(1=almost no compression, 10=extreme Prompt Diet). "
            "Use the measured compression_ratio_pct as primary evidence."
        ),
        ge=1,
        le=10,
    )
    information_retention: int = Field(
        description=(
            "Whether PageRank core / top nodes are retained in RepoMap without "
            "omission (1=severe missing cores, 10=all critical nodes present). "
            "Missing high-PageRank cores must be scored harshly."
        ),
        ge=1,
        le=10,
    )
    feedback: str = Field(
        description=(
            "Actionable Prompt Diet directives for the next agent loop "
            "(at most 3 short lines; no subjective writing-quality critique)."
        ),
    )
