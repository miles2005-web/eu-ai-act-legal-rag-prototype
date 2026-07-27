"""Serializable domain models for final assessment reports."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime
from typing import Any, ClassVar

from src.assessment.baselines import AssessmentBaseline
from src.assessment.evidence.models import Evidence, FindingEvidenceBinding
from src.assessment.execution import RuleExecutionRecord
from src.assessment.findings import Finding
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.invocations import AuthorizedRuleInvocation, RuleInvocation
from src.assessment.models import (
    SerializableModel,
    model_from_dict,
    validate_stable_identifier,
)
from src.assessment.recruitment_models import (
    ComplianceArtefactMetadata,
    RoleHypothesis,
    normalized_identifier_list,
)
from src.assessment.requirements import MissingFactReason
from src.assessment.results import RuleExecutionFailure
from src.assessment.scope import (
    ActorId,
    ProcessingOperationId,
    SystemId,
    WorkflowId,
)


@dataclass(slots=True)
class MissingInformation(SerializableModel):
    """One unresolved fact retained in the final report."""

    rule_id: str
    rule_version: str
    fact_path: str
    reason: MissingFactReason
    framework: RegulatoryFramework = RegulatoryFramework.UNKNOWN


@dataclass(slots=True)
class RuleVersionMetadata(SerializableModel):
    """Rule identity and version used or considered by the assessment."""

    rule_id: str
    version: str
    framework: RegulatoryFramework = RegulatoryFramework.UNKNOWN


@dataclass(slots=True)
class FrameworkFindings(SerializableModel):
    """Findings belonging to one regulatory framework."""

    framework: RegulatoryFramework
    findings: list[Finding]

    def __post_init__(self) -> None:
        if not isinstance(self.framework, RegulatoryFramework):
            raise TypeError("framework must be a RegulatoryFramework")
        if any(not isinstance(finding, Finding) for finding in self.findings):
            raise TypeError("findings must contain Finding instances")
        if any(finding.framework is not self.framework for finding in self.findings):
            raise ValueError("all findings must match the group framework")


@dataclass(slots=True)
class InformationalGap(SerializableModel):
    gap_id: str
    scope_description: str
    reason_codes: list[str] = field(default_factory=list)
    missing_fact_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ApplicabilityLimitation(SerializableModel):
    temporal_assessment_complete: bool = False
    territorial_assessment_complete: bool = False
    explanation: str | None = None


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
    findings_by_framework: list[FrameworkFindings] = field(default_factory=list)
    assessed_frameworks: list[RegulatoryFramework] = field(default_factory=list)
    authorized_rule_ids: list[str] = field(default_factory=list)
    rule_invocations: list[RuleInvocation] = field(default_factory=list)
    authorized_rule_invocations: list[AuthorizedRuleInvocation] = field(
        default_factory=list
    )
    rule_execution_records: list[RuleExecutionRecord] = field(default_factory=list)
    actor_references: list[ActorId] = field(default_factory=list)
    system_references: list[SystemId] = field(default_factory=list)
    workflow_references: list[WorkflowId] = field(default_factory=list)
    processing_operation_references: list[ProcessingOperationId] = field(
        default_factory=list
    )
    role_hypotheses: list[RoleHypothesis] = field(default_factory=list)
    compliance_artefacts: list[ComplianceArtefactMetadata] = field(
        default_factory=list
    )
    unresolved_informational_gaps: list[InformationalGap] = field(
        default_factory=list
    )
    applicability_limitation: ApplicabilityLimitation | None = None
    assessment_baseline: AssessmentBaseline | None = None
    _serialized_fields: frozenset[str] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _report_version_at_construction: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"report_id"}
    )

    _V2_ONLY_FIELDS: ClassVar[tuple[str, ...]] = (
        "rule_invocations",
        "authorized_rule_invocations",
        "rule_execution_records",
        "actor_references",
        "system_references",
        "workflow_references",
        "processing_operation_references",
        "role_hypotheses",
        "compliance_artefacts",
        "unresolved_informational_gaps",
        "applicability_limitation",
        "assessment_baseline",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_report_version_at_construction", self.report_version
        )
        self._validate_contract()

    def _validate_contract(self) -> None:
        if self.report_version != self._report_version_at_construction:
            raise ValueError(
                "report_version cannot be changed in place; use an explicit "
                "report migration boundary"
            )
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
        validate_stable_identifier(self.report_id, field_name="report_id")
        validate_stable_identifier(
            self.assessment_run_reference,
            field_name="assessment_run_reference",
        )
        if not isinstance(self.generated_at, datetime):
            raise TypeError("generated_at must be a datetime")
        if any(
            not isinstance(recommendation, str) or not recommendation.strip()
            for recommendation in self.recommendations
        ):
            raise ValueError("recommendations must contain non-empty strings")
        if any(
            not isinstance(rule_id, str) or not rule_id.strip()
            for rule_id in self.authorized_rule_ids
        ):
            raise ValueError("authorized_rule_ids must contain non-empty strings")
        if len(set(self.authorized_rule_ids)) != len(self.authorized_rule_ids):
            raise ValueError("authorized_rule_ids must not contain duplicates")
        if self.report_version not in ("1.0.0", "2.0.0"):
            raise ValueError("unsupported report_version")
        if self.report_version == "1.0.0":
            populated = [
                field_name
                for field_name in self._V2_ONLY_FIELDS
                if getattr(self, field_name) not in (None, [], ())
            ]
            if populated:
                raise ValueError(
                    "Report 1.0.0 cannot contain Report 2.0.0 fields: "
                    + ", ".join(populated)
                )
        else:
            self._normalize_v2_references()
        self._validate_source_shape()

    def _validate_source_shape(self) -> None:
        if self._serialized_fields is None:
            return
        for model_field in fields(self):
            name = model_field.name
            if name.startswith("_") or name in self._serialized_fields:
                continue
            if model_field.default is not MISSING:
                default = model_field.default
            elif model_field.default_factory is not MISSING:
                default = model_field.default_factory()
            else:
                continue
            if getattr(self, name) != default:
                raise ValueError(
                    f"report field {name!r} was absent from the source payload; "
                    "use an explicit report migration boundary before populating it"
                )

    def _normalize_v2_references(self) -> None:
        typed_collections = (
            ("rule_invocations", RuleInvocation),
            ("authorized_rule_invocations", AuthorizedRuleInvocation),
            ("rule_execution_records", RuleExecutionRecord),
            ("role_hypotheses", RoleHypothesis),
            ("compliance_artefacts", ComplianceArtefactMetadata),
            ("unresolved_informational_gaps", InformationalGap),
        )
        for field_name, model_type in typed_collections:
            normalized = []
            for item in getattr(self, field_name):
                if isinstance(item, model_type):
                    normalized.append(item)
                elif isinstance(item, dict):
                    normalized.append(model_type.from_dict(item))
                else:
                    raise TypeError(
                        f"{field_name} must contain {model_type.__name__} instances"
                    )
            setattr(self, field_name, normalized)
        if isinstance(self.applicability_limitation, dict):
            self.applicability_limitation = ApplicabilityLimitation.from_dict(
                self.applicability_limitation
            )
        if (
            self.applicability_limitation is not None
            and not isinstance(
                self.applicability_limitation, ApplicabilityLimitation
            )
        ):
            raise TypeError(
                "applicability_limitation must be an ApplicabilityLimitation"
            )
        if isinstance(self.assessment_baseline, dict):
            self.assessment_baseline = AssessmentBaseline.from_dict(
                self.assessment_baseline
            )
        if self.assessment_baseline is not None and not isinstance(
            self.assessment_baseline, AssessmentBaseline
        ):
            raise TypeError("assessment_baseline must be an AssessmentBaseline")
        for field_name, identifier_type in (
            ("actor_references", ActorId),
            ("system_references", SystemId),
            ("workflow_references", WorkflowId),
            ("processing_operation_references", ProcessingOperationId),
        ):
            setattr(
                self,
                field_name,
                normalized_identifier_list(
                    getattr(self, field_name),
                    identifier_type,
                    field_name=field_name,
                )
                or [],
            )

    def to_dict(self) -> dict[str, object]:
        """Keep Report 1.0.0 serialization byte-shape compatible."""

        self._validate_contract()
        payload = SerializableModel.to_dict(self)
        if self.report_version == "1.0.0":
            for field_name in self._V2_ONLY_FIELDS:
                payload.pop(field_name, None)
        if self._serialized_fields is not None:
            payload = {
                key: value
                for key, value in payload.items()
                if key in self._serialized_fields
            }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AssessmentReport:
        """Read reports in place; never upgrade a historical snapshot."""

        restored = model_from_dict(cls, payload)
        restored._serialized_fields = frozenset(payload)
        return restored
