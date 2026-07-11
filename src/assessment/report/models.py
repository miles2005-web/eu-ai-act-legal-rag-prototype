"""Serializable domain models for final assessment reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.assessment.evidence.models import Evidence, FindingEvidenceBinding
from src.assessment.findings import Finding
from src.assessment.models import SerializableModel
from src.assessment.requirements import MissingFactReason
from src.assessment.results import RuleExecutionFailure


@dataclass(slots=True)
class MissingInformation(SerializableModel):
    """One unresolved fact retained in the final report."""

    rule_id: str
    rule_version: str
    fact_path: str
    reason: MissingFactReason


@dataclass(slots=True)
class RuleVersionMetadata(SerializableModel):
    """Rule identity and version used or considered by the assessment."""

    rule_id: str
    version: str


@dataclass(slots=True)
class AssessmentReport(SerializableModel):
    """Traceable, deterministic compliance assessment output."""

    report_id: str
    assessment_run_reference: str
    generated_at: datetime
    summary: str
    findings: list[Finding]
    evidence: list[Evidence]
    evidence_bindings: list[FindingEvidenceBinding]
    missing_information: list[MissingInformation]
    recommendations: list[str]
    engine_version: str
    rule_versions: list[RuleVersionMetadata]
    execution_failures: list[RuleExecutionFailure]
    report_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "report_id",
            "assessment_run_reference",
            "summary",
            "engine_version",
            "report_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.generated_at, datetime):
            raise TypeError("generated_at must be a datetime")
        if any(
            not isinstance(recommendation, str) or not recommendation.strip()
            for recommendation in self.recommendations
        ):
            raise ValueError("recommendations must contain non-empty strings")

