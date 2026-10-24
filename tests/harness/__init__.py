"""GraphDistiller audit harness — exhaustive, serializable inspection of pipeline artifacts.

Collectors read product artifacts (checkpoints, codex, ledgers, errors.log,
control-room files) and reuse product pure functions; they never run a
parallel pipeline and never write to tests/eval/ (observability placement).
Every collector returns MetricRecord / CheckResult so one AuditReport can be
rendered as JSON, CSV and Markdown.
"""

from harness.model import SCHEMA_VERSION, AuditReport, CheckResult, MetricRecord

__all__ = ["SCHEMA_VERSION", "AuditReport", "CheckResult", "MetricRecord"]
