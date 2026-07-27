"""Observable recruitment facts and informational v0.6 domain contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from collections.abc import Mapping
from typing import ClassVar, TypeVar

from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import SerializableModel, TriState
from src.assessment.scope import (
    ActorId,
    AssessmentScope,
    ProcessingOperationId,
    StableIdentifier,
    SystemId,
    WorkflowId,
)


_IdentifierT = TypeVar("_IdentifierT", bound=StableIdentifier)
_EnumT = TypeVar("_EnumT", bound=Enum)


def normalize_enum(
    value: object,
    enum_type: type[_EnumT],
    *,
    field_name: str,
) -> _EnumT:
    """Normalize a declared enum without accepting arbitrary scalar types."""

    if isinstance(value, enum_type):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a {enum_type.__name__}")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} is not a valid {enum_type.__name__}: {value!r}"
        ) from exc


def normalized_string_collection(
    values: list[str] | tuple[str, ...],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Normalize an unordered collection of non-empty strings."""

    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must contain non-empty strings")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return tuple(sorted(normalized))


def normalized_string_mapping(
    values: Mapping[object, object],
    *,
    field_name: str,
) -> dict[str, str]:
    """Validate informational metadata without dereferencing its values."""

    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string-to-string mapping")
        normalized[key] = value
    return dict(sorted(normalized.items()))


def normalized_identifier_list(
    values: list[str] | tuple[str, ...] | None,
    identifier_type: type[_IdentifierT],
    *,
    field_name: str,
) -> list[_IdentifierT] | None:
    """Normalize a set-like stable-reference collection deterministically."""

    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise TypeError(f"{field_name} must be a list or tuple")
    normalized = [identifier_type(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicate identifiers")
    return sorted(normalized, key=str)


def normalized_identifier_mapping(
    values: Mapping[str, list[str] | tuple[str, ...]] | None,
    key_type: type[_IdentifierT],
    value_type: type[StableIdentifier],
    *,
    field_name: str,
) -> dict[_IdentifierT, list[StableIdentifier]] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized: dict[_IdentifierT, list[StableIdentifier]] = {}
    for raw_key, raw_values in values.items():
        key = key_type(raw_key)
        if key in normalized:
            raise ValueError(f"{field_name} contains duplicate normalized keys")
        normalized[key] = normalized_identifier_list(
            raw_values,
            value_type,
            field_name=f"{field_name}[{key}]",
        ) or []
    return dict(sorted(normalized.items(), key=lambda item: str(item[0])))


def normalized_key_mapping(
    values: Mapping[str, object] | None,
    key_type: type[_IdentifierT],
    *,
    field_name: str,
) -> dict[_IdentifierT, object] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized: dict[_IdentifierT, object] = {}
    for raw_key, value in values.items():
        key = key_type(raw_key)
        if key in normalized:
            raise ValueError(f"{field_name} contains duplicate normalized keys")
        normalized[key] = value
    return dict(sorted(normalized.items(), key=lambda item: str(item[0])))


class _StableIdentityRecord:
    """Permit record edits while preventing silent identity replacement."""

    _identity_field: str

    def __setattr__(self, name: str, value: object) -> None:
        if name == self._identity_field and hasattr(self, name):
            raise AttributeError(
                f"{self._identity_field} is immutable; create a new record and "
                "retire the prior identity"
            )
        object.__setattr__(self, name, value)


class ActorKind(str, Enum):
    EMPLOYER = "employer"
    RECRUITER = "recruiter"
    HEADHUNTER = "headhunter"
    AI_VENDOR = "ai_vendor"
    DATA_SUPPLIER = "data_supplier"
    APPLICANT = "applicant"
    OTHER = "other"
    UNKNOWN = "unknown"


class ClassificationReviewState(str, Enum):
    REVIEWED = "reviewed"
    NOT_REVIEWED = "not_reviewed"
    UNKNOWN = "unknown"


class ArtefactSupplyState(str, Enum):
    SUPPLIED = "supplied"
    NOT_SUPPLIED = "not_supplied"
    UNKNOWN = "unknown"


class RoleHypothesisStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"
    NOT_ASSESSED = "not_assessed"


@dataclass(slots=True)
class RecordMetadata(SerializableModel):
    record_version: str = "1.0.0"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    supersedes_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.supersedes_ids = normalized_identifier_list(
            self.supersedes_ids,
            StableIdentifier,
            field_name="supersedes_ids",
        ) or []


@dataclass(slots=True)
class ActorFacts(_StableIdentityRecord, SerializableModel):
    _identity_field = "actor_id"
    actor_id: ActorId
    display_name: str | None = None
    actor_kind: ActorKind = ActorKind.UNKNOWN
    legal_form: str | None = None
    establishment_locations: list[str] | None = None
    acts_in_own_name: TriState = TriState.UNKNOWN
    operates_system_ids: list[SystemId] | None = None
    develops_or_commissions_system_ids: list[SystemId] | None = None
    markets_system_ids_under_own_name: list[SystemId] | None = None
    uses_system_ids_in_own_organisation: list[SystemId] | None = None
    uses_system_ids_on_behalf_of_actor_ids: dict[str, list[ActorId]] | None = None
    metadata: RecordMetadata = field(default_factory=RecordMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", ActorId(self.actor_id))
        self.actor_kind = normalize_enum(
            self.actor_kind, ActorKind, field_name="actor_kind"
        )
        self.acts_in_own_name = normalize_enum(
            self.acts_in_own_name, TriState, field_name="acts_in_own_name"
        )
        self.normalize_references()

    def normalize_references(self) -> None:
        for field_name in (
            "operates_system_ids",
            "develops_or_commissions_system_ids",
            "markets_system_ids_under_own_name",
            "uses_system_ids_in_own_organisation",
        ):
            setattr(
                self,
                field_name,
                normalized_identifier_list(
                    getattr(self, field_name),
                    SystemId,
                    field_name=field_name,
                ),
            )
        self.uses_system_ids_on_behalf_of_actor_ids = (
            normalized_identifier_mapping(
                self.uses_system_ids_on_behalf_of_actor_ids,
                SystemId,
                ActorId,
                field_name="uses_system_ids_on_behalf_of_actor_ids",
            )
        )


@dataclass(slots=True)
class AISystemFacts(_StableIdentityRecord, SerializableModel):
    _identity_field = "system_id"
    system_id: SystemId
    name: str | None = None
    description: str | None = None
    intended_purpose: str | None = None
    lifecycle_status: str = "unknown"
    outputs: list[str] | None = None
    degree_of_autonomy: str = "unknown"
    vendor_actor_ids: list[ActorId] | None = None
    commissioning_actor_ids: list[ActorId] | None = None
    branding_actor_ids: list[ActorId] | None = None
    selected_by_actor_ids: list[ActorId] | None = None
    configured_by_actor_ids: list[ActorId] | None = None
    put_into_service_date: date | None = None
    system_use_locations: list[str] | None = None
    metadata: RecordMetadata = field(default_factory=RecordMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "system_id", SystemId(self.system_id))
        self.normalize_references()

    def normalize_references(self) -> None:
        for field_name in (
            "vendor_actor_ids",
            "commissioning_actor_ids",
            "branding_actor_ids",
            "selected_by_actor_ids",
            "configured_by_actor_ids",
        ):
            setattr(
                self,
                field_name,
                normalized_identifier_list(
                    getattr(self, field_name),
                    ActorId,
                    field_name=field_name,
                ),
            )


@dataclass(slots=True)
class RecruitmentWorkflowFacts(_StableIdentityRecord, SerializableModel):
    _identity_field = "workflow_id"
    workflow_id: WorkflowId
    title: str | None = None
    recruitment_objective: str | None = None
    employer_actor_ids: list[ActorId] | None = None
    recruiter_actor_ids: list[ActorId] | None = None
    system_ids: list[SystemId] | None = None
    candidate_population: str | None = None
    recruitment_stages: list[str] | None = None
    output_recipient_actor_ids: list[ActorId] | None = None
    final_decision_actor_ids: list[ActorId] | None = None
    intended_use_date: date | None = None
    system_use_locations: list[str] | None = None
    output_use_locations: list[str] | None = None
    affected_person_locations: list[str] | None = None
    processing_operation_ids: list[ProcessingOperationId] | None = None
    metadata: RecordMetadata = field(default_factory=RecordMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow_id", WorkflowId(self.workflow_id))
        self.normalize_references()

    def normalize_references(self) -> None:
        for field_name in (
            "employer_actor_ids",
            "recruiter_actor_ids",
            "output_recipient_actor_ids",
            "final_decision_actor_ids",
        ):
            setattr(
                self,
                field_name,
                normalized_identifier_list(
                    getattr(self, field_name),
                    ActorId,
                    field_name=field_name,
                ),
            )
        self.system_ids = normalized_identifier_list(
            self.system_ids, SystemId, field_name="system_ids"
        )
        self.processing_operation_ids = normalized_identifier_list(
            self.processing_operation_ids,
            ProcessingOperationId,
            field_name="processing_operation_ids",
        )


@dataclass(slots=True)
class ProcessingOperationFacts(_StableIdentityRecord, SerializableModel):
    _identity_field = "processing_operation_id"
    processing_operation_id: ProcessingOperationId
    workflow_id: WorkflowId | None = None
    system_ids: list[SystemId] | None = None
    participating_actor_ids: list[ActorId] | None = None
    reported_purpose: str | None = None
    candidate_population: str | None = None
    data_categories: list[str] | None = None
    data_sources: list[str] | None = None
    recipients: list[ActorId] | None = None
    within_documented_instructions: TriState = TriState.UNKNOWN
    outside_documented_instructions: TriState = TriState.UNKNOWN
    independent_reuse_purpose: TriState = TriState.UNKNOWN
    operation_start_date: date | None = None
    operation_end_date: date | None = None
    territorial_context: list[str] | None = None
    affected_person_locations: list[str] | None = None
    metadata: RecordMetadata = field(default_factory=RecordMetadata)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "processing_operation_id",
            ProcessingOperationId(self.processing_operation_id),
        )
        if self.workflow_id is not None:
            self.workflow_id = WorkflowId(self.workflow_id)
        for field_name in (
            "within_documented_instructions",
            "outside_documented_instructions",
            "independent_reuse_purpose",
        ):
            setattr(
                self,
                field_name,
                normalize_enum(
                    getattr(self, field_name),
                    TriState,
                    field_name=field_name,
                ),
            )
        self.normalize_references()

    def normalize_references(self) -> None:
        if self.workflow_id is not None:
            self.workflow_id = WorkflowId(self.workflow_id)
        self.system_ids = normalized_identifier_list(
            self.system_ids, SystemId, field_name="system_ids"
        )
        self.participating_actor_ids = normalized_identifier_list(
            self.participating_actor_ids,
            ActorId,
            field_name="participating_actor_ids",
        )
        self.recipients = normalized_identifier_list(
            self.recipients, ActorId, field_name="recipients"
        )


@dataclass(slots=True)
class ScreeningCriterionFacts(_StableIdentityRecord, SerializableModel):
    _identity_field = "criterion_id"
    criterion_id: StableIdentifier
    category: str = "unknown"
    description: str | None = None
    selecting_actor_ids: list[ActorId] | None = None
    configuring_actor_ids: list[ActorId] | None = None
    used_for_ranking: TriState = TriState.UNKNOWN
    used_for_filtering: TriState = TriState.UNKNOWN
    used_for_recommendation: TriState = TriState.UNKNOWN
    used_for_exclusion: TriState = TriState.UNKNOWN
    gdpr_special_category_data: TriState = TriState.UNKNOWN
    employment_equality_protected_characteristic: TriState = TriState.UNKNOWN
    proxy_for_protected_characteristic: TriState = TriState.UNKNOWN
    classification_review_state: ClassificationReviewState = (
        ClassificationReviewState.UNKNOWN
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "criterion_id", StableIdentifier(self.criterion_id)
        )
        for field_name in (
            "used_for_ranking",
            "used_for_filtering",
            "used_for_recommendation",
            "used_for_exclusion",
            "gdpr_special_category_data",
            "employment_equality_protected_characteristic",
            "proxy_for_protected_characteristic",
        ):
            setattr(
                self,
                field_name,
                normalize_enum(
                    getattr(self, field_name),
                    TriState,
                    field_name=field_name,
                ),
            )
        self.classification_review_state = normalize_enum(
            self.classification_review_state,
            ClassificationReviewState,
            field_name="classification_review_state",
        )
        self.normalize_references()

    def normalize_references(self) -> None:
        self.selecting_actor_ids = normalized_identifier_list(
            self.selecting_actor_ids,
            ActorId,
            field_name="selecting_actor_ids",
        )
        self.configuring_actor_ids = normalized_identifier_list(
            self.configuring_actor_ids,
            ActorId,
            field_name="configuring_actor_ids",
        )


@dataclass(slots=True)
class RecruitmentDecisionProcessFacts(_StableIdentityRecord, SerializableModel):
    _identity_field = "process_id"
    process_id: StableIdentifier
    workflow_id: WorkflowId | None = None
    processing_operation_id: ProcessingOperationId | None = None
    system_id: SystemId | None = None
    ranking: TriState = TriState.UNKNOWN
    filtering: TriState = TriState.UNKNOWN
    recommendation: TriState = TriState.UNKNOWN
    automatic_exclusion: TriState = TriState.UNKNOWN
    materially_influences_decision: TriState = TriState.UNKNOWN
    human_review: TriState = TriState.UNKNOWN
    substantive_basis_review: TriState = TriState.UNKNOWN
    genuine_override_authority: TriState = TriState.UNKNOWN
    routine_following_of_ai_output: TriState = TriState.UNKNOWN
    screening_criteria: list[ScreeningCriterionFacts] | None = None
    documented_instructions: TriState = TriState.UNKNOWN
    system_configuration_authority: TriState = TriState.UNKNOWN
    final_decision_authority: TriState = TriState.UNKNOWN

    def __post_init__(self) -> None:
        object.__setattr__(self, "process_id", StableIdentifier(self.process_id))
        for field_name in (
            "ranking",
            "filtering",
            "recommendation",
            "automatic_exclusion",
            "materially_influences_decision",
            "human_review",
            "substantive_basis_review",
            "genuine_override_authority",
            "routine_following_of_ai_output",
            "documented_instructions",
            "system_configuration_authority",
            "final_decision_authority",
        ):
            setattr(
                self,
                field_name,
                normalize_enum(
                    getattr(self, field_name),
                    TriState,
                    field_name=field_name,
                ),
            )
        self.normalize_references()

    def normalize_references(self) -> None:
        if self.workflow_id is not None:
            self.workflow_id = WorkflowId(self.workflow_id)
        if self.processing_operation_id is not None:
            self.processing_operation_id = ProcessingOperationId(
                self.processing_operation_id
            )
        if self.system_id is not None:
            self.system_id = SystemId(self.system_id)
        if self.screening_criteria is not None:
            seen: set[str] = set()
            for criterion in self.screening_criteria:
                criterion.normalize_references()
                if str(criterion.criterion_id) in seen:
                    raise ValueError(
                        "screening_criteria must not contain duplicate criterion IDs"
                    )
                seen.add(str(criterion.criterion_id))
            self.screening_criteria.sort(key=lambda item: str(item.criterion_id))


@dataclass(slots=True)
class TemporalContextFacts(SerializableModel):
    assessment_date: date | None = None
    intended_use_dates: dict[str, date | None] | None = None
    put_into_service_dates: dict[str, date | None] | None = None
    operation_dates: dict[str, tuple[date | None, date | None]] | None = None
    legal_source_baseline_id: str | None = None

    def __post_init__(self) -> None:
        self.normalize_references()

    def normalize_references(self) -> None:
        self.intended_use_dates = normalized_key_mapping(
            self.intended_use_dates,
            WorkflowId,
            field_name="intended_use_dates",
        )
        self.put_into_service_dates = normalized_key_mapping(
            self.put_into_service_dates,
            SystemId,
            field_name="put_into_service_dates",
        )
        self.operation_dates = normalized_key_mapping(
            self.operation_dates,
            ProcessingOperationId,
            field_name="operation_dates",
        )


@dataclass(slots=True)
class TerritorialContextFacts(SerializableModel):
    actor_establishment_locations: dict[str, list[str] | None] | None = None
    system_use_locations: dict[str, list[str] | None] | None = None
    output_use_locations: dict[str, list[str] | None] | None = None
    affected_person_locations: dict[str, list[str] | None] | None = None
    processing_operation_context: dict[str, list[str] | None] | None = None

    def __post_init__(self) -> None:
        self.normalize_references()

    def normalize_references(self) -> None:
        self.actor_establishment_locations = normalized_key_mapping(
            self.actor_establishment_locations,
            ActorId,
            field_name="actor_establishment_locations",
        )
        self.system_use_locations = normalized_key_mapping(
            self.system_use_locations,
            SystemId,
            field_name="system_use_locations",
        )
        self.output_use_locations = normalized_key_mapping(
            self.output_use_locations,
            WorkflowId,
            field_name="output_use_locations",
        )
        self.affected_person_locations = normalized_key_mapping(
            self.affected_person_locations,
            WorkflowId,
            field_name="affected_person_locations",
        )
        self.processing_operation_context = normalized_key_mapping(
            self.processing_operation_context,
            ProcessingOperationId,
            field_name="processing_operation_context",
        )


@dataclass(slots=True)
class ComplianceArtefactMetadata(_StableIdentityRecord, SerializableModel):
    """Metadata only; ``file_reference`` must never be fetched or opened."""

    _identity_field = "artefact_id"
    artefact_id: StableIdentifier
    artefact_type: str
    scope: AssessmentScope = field(default_factory=AssessmentScope)
    file_reference: str | None = None
    supply_state: ArtefactSupplyState = ArtefactSupplyState.UNKNOWN
    provenance: dict[str, str] = field(default_factory=dict)
    descriptive_metadata: dict[str, str] = field(default_factory=dict)

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"artefact_id"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artefact_id", StableIdentifier(self.artefact_id)
        )
        if not isinstance(self.artefact_type, str) or not self.artefact_type.strip():
            raise ValueError("artefact_type must be a non-empty string")
        self.artefact_type = self.artefact_type.strip()
        if isinstance(self.scope, dict):
            self.scope = AssessmentScope.from_dict(self.scope)
        if not isinstance(self.scope, AssessmentScope):
            raise TypeError("scope must be an AssessmentScope")
        if self.file_reference is not None:
            if (
                not isinstance(self.file_reference, str)
                or not self.file_reference.strip()
            ):
                raise ValueError(
                    "file_reference must be None or a non-empty opaque string"
                )
            self.file_reference = self.file_reference.strip()
        self.supply_state = normalize_enum(
            self.supply_state,
            ArtefactSupplyState,
            field_name="supply_state",
        )
        self.provenance = normalized_string_mapping(
            self.provenance, field_name="provenance"
        )
        self.descriptive_metadata = normalized_string_mapping(
            self.descriptive_metadata,
            field_name="descriptive_metadata",
        )


@dataclass(slots=True)
class RoleHypothesis(_StableIdentityRecord, SerializableModel):
    """Informational projection that is never a formal legal Finding."""

    _identity_field = "hypothesis_id"
    hypothesis_id: StableIdentifier
    framework: RegulatoryFramework
    scope: AssessmentScope
    hypothesis_type: str
    status: RoleHypothesisStatus = RoleHypothesisStatus.UNRESOLVED
    supporting_fact_paths: list[str] = field(default_factory=list)
    contradictory_fact_paths: list[str] = field(default_factory=list)
    missing_fact_paths: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    projection_version: str = "1.0.0"

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"hypothesis_id"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "hypothesis_id", StableIdentifier(self.hypothesis_id)
        )
        self.framework = normalize_enum(
            self.framework,
            RegulatoryFramework,
            field_name="framework",
        )
        if isinstance(self.scope, dict):
            self.scope = AssessmentScope.from_dict(self.scope)
        if not isinstance(self.scope, AssessmentScope):
            raise TypeError("scope must be an AssessmentScope")
        if (
            not isinstance(self.hypothesis_type, str)
            or not self.hypothesis_type.strip()
        ):
            raise ValueError("hypothesis_type must be a non-empty string")
        self.hypothesis_type = self.hypothesis_type.strip()
        self.status = normalize_enum(
            self.status,
            RoleHypothesisStatus,
            field_name="status",
        )
        if self.projection_version != "1.0.0":
            raise ValueError("unsupported RoleHypothesis projection_version")
        for field_name in (
            "supporting_fact_paths",
            "contradictory_fact_paths",
            "missing_fact_paths",
            "reason_codes",
        ):
            setattr(
                self,
                field_name,
                list(
                    normalized_string_collection(
                        getattr(self, field_name),
                        field_name=field_name,
                    )
                ),
            )
