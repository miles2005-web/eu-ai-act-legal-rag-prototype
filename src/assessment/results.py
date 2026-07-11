"""Structured outputs from an assessment engine execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.assessment.findings import Finding
from src.assessment.models import SerializableModel, utc_now
from src.assessment.requirements import RuleRequirementResult


@dataclass(slots=True)
class RuleExecutionFailure(SerializableModel):
    """Operational failure captured while attempting to execute one rule."""

    rule_id: str
    rule_version: str
    error_type: str
    message: str


@dataclass(slots=True)
class AssessmentResult(SerializableModel):
    """Deterministically ordered output from one engine execution."""

    findings: list[Finding]
    executed_rule_ids: list[str]
    engine_version: str
    timestamp: datetime = field(default_factory=utc_now)
    failures: list[RuleExecutionFailure] = field(default_factory=list)
    missing_fact_requirements: list[RuleRequirementResult] = field(
        default_factory=list
    )
