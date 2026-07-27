"""Typed fact schema for an EU AI Act assessment case."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from copy import deepcopy
from datetime import date, datetime
from enum import Enum
from typing import Any, ClassVar

from src.assessment.models import SerializableModel, TriState, model_from_dict
from src.assessment.recruitment_models import (
    ActorFacts,
    AISystemFacts,
    ComplianceArtefactMetadata,
    ProcessingOperationFacts,
    RecruitmentDecisionProcessFacts,
    RecruitmentWorkflowFacts,
    TemporalContextFacts,
    TerritorialContextFacts,
    normalized_identifier_list,
)
from src.assessment.scope import StableIdentifier


class LifecycleStatus(str, Enum):
    PLANNED = "planned"
    DEVELOPMENT = "development"
    PILOT = "pilot"
    DEPLOYED = "deployed"
    RETIRED = "retired"
    UNKNOWN = "unknown"


class SystemOutput(str, Enum):
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    DECISION = "decision"
    GENERATED_CONTENT = "generated_content"
    OTHER = "other"


class DegreeOfAutonomy(str, Enum):
    NONE = "none"
    LIMITED = "limited"
    SUBSTANTIAL = "substantial"
    UNKNOWN = "unknown"


class UseDomain(str, Enum):
    BIOMETRICS = "biometrics"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    EDUCATION = "education"
    EMPLOYMENT = "employment"
    ESSENTIAL_SERVICES = "essential_services"
    LAW_ENFORCEMENT = "law_enforcement"
    MIGRATION_ASYLUM_BORDER_CONTROL = "migration_asylum_border_control"
    JUSTICE_DEMOCRATIC_PROCESSES = "justice_democratic_processes"
    PRODUCT_SAFETY = "product_safety"
    OTHER = "other"
    UNKNOWN = "unknown"


class AffectedPerson(str, Enum):
    WORKER = "worker"
    JOB_CANDIDATE = "job_candidate"
    STUDENT = "student"
    CONSUMER = "consumer"
    PATIENT = "patient"
    CHILD = "child"
    OTHER = "other"


class AnnexIIIArea(str, Enum):
    BIOMETRICS = "biometrics"
    CRITICAL_INFRASTRUCTURE = "critical_infrastructure"
    EDUCATION = "education"
    EMPLOYMENT = "employment"
    ESSENTIAL_SERVICES = "essential_services"
    LAW_ENFORCEMENT = "law_enforcement"
    MIGRATION_ASYLUM_BORDER_CONTROL = "migration_asylum_border_control"
    JUSTICE_DEMOCRATIC_PROCESSES = "justice_democratic_processes"
    OTHER = "other"


class FactSource(str, Enum):
    QUESTIONNAIRE = "questionnaire"
    CASE_EDIT = "case_edit"
    CONFIRMED_ASSISTANT_SUGGESTION = "confirmed_assistant_suggestion"
    IMPORT = "import"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class CaseFacts(SerializableModel):
    title: str | None = None
    reference: str | None = None
    assessment_date: date | None = None
    notes: str | None = None


@dataclass(slots=True)
class SystemFacts(SerializableModel):
    name: str | None = None
    description: str | None = None
    lifecycle_status: LifecycleStatus = LifecycleStatus.UNKNOWN
    intended_purpose: str | None = None
    # None means unknown; an empty list means confirmed to have no listed output.
    outputs: list[SystemOutput] | None = None
    machine_based_inference: TriState = TriState.UNKNOWN
    degree_of_autonomy: DegreeOfAutonomy = DegreeOfAutonomy.UNKNOWN
    adaptiveness_after_deployment: TriState = TriState.UNKNOWN


@dataclass(slots=True)
class ScopeFacts(SerializableModel):
    provider_established_in_eu: TriState = TriState.UNKNOWN
    deployer_established_in_eu: TriState = TriState.UNKNOWN
    placed_on_eu_market: TriState = TriState.UNKNOWN
    put_into_service_in_eu: TriState = TriState.UNKNOWN
    output_used_in_eu: TriState = TriState.UNKNOWN
    exclusively_military_defence_national_security: TriState = TriState.UNKNOWN
    research_before_market_or_service: TriState = TriState.UNKNOWN
    personal_non_professional_activity: TriState = TriState.UNKNOWN
    open_source_release: TriState = TriState.UNKNOWN


@dataclass(slots=True)
class OrganisationFacts(SerializableModel):
    name: str | None = None
    # None means unknown; an empty list means no establishment countries reported.
    establishment_countries: list[str] | None = None


@dataclass(slots=True)
class SupplyChainFacts(SerializableModel):
    develops_system: TriState = TriState.UNKNOWN
    has_system_developed_for_it: TriState = TriState.UNKNOWN
    markets_under_own_name_or_trademark: TriState = TriState.UNKNOWN
    uses_system_under_its_authority: TriState = TriState.UNKNOWN
    introduces_third_country_system_into_eu_market: TriState = TriState.UNKNOWN
    makes_system_available_in_eu_distribution_chain: TriState = TriState.UNKNOWN
    substantial_modification: TriState = TriState.UNKNOWN
    changed_intended_purpose: TriState = TriState.UNKNOWN


@dataclass(slots=True)
class UseContextFacts(SerializableModel):
    domain: UseDomain = UseDomain.UNKNOWN
    task: str | None = None
    # None means unknown; an empty list means no affected-person group identified.
    affected_persons: list[AffectedPerson] | None = None
    materially_influences_decision: TriState = TriState.UNKNOWN
    human_review_before_effect: TriState = TriState.UNKNOWN
    profiles_natural_persons: TriState = TriState.UNKNOWN


@dataclass(slots=True)
class PracticesFacts(SerializableModel):
    manipulative_or_deceptive_technique: TriState = TriState.UNKNOWN
    exploits_vulnerability: TriState = TriState.UNKNOWN
    causes_or_likely_causes_significant_harm: TriState = TriState.UNKNOWN
    social_scoring: TriState = TriState.UNKNOWN
    criminal_risk_based_solely_on_profiling: TriState = TriState.UNKNOWN
    untargeted_facial_image_scraping: TriState = TriState.UNKNOWN
    emotion_inference: TriState = TriState.UNKNOWN
    emotion_inference_workplace_or_education: TriState = TriState.UNKNOWN
    biometric_categorisation_sensitive_attributes: TriState = TriState.UNKNOWN
    realtime_remote_biometric_identification_public_space_law_enforcement: TriState = (
        TriState.UNKNOWN
    )


@dataclass(slots=True)
class HighRiskFacts(SerializableModel):
    is_safety_component_or_product: TriState = TriState.UNKNOWN
    product_covered_by_annex_i: TriState = TriState.UNKNOWN
    requires_third_party_conformity_assessment: TriState = TriState.UNKNOWN
    annex_iii_area: AnnexIIIArea | None = None
    annex_iii_use_case: str | None = None
    narrow_procedural_task: TriState = TriState.UNKNOWN
    improves_completed_human_activity: TriState = TriState.UNKNOWN
    detects_patterns_without_replacing_or_influencing_human_assessment: TriState = (
        TriState.UNKNOWN
    )
    preparatory_task: TriState = TriState.UNKNOWN


@dataclass(slots=True)
class ProductRegulationFacts(SerializableModel):
    """Facts reserved for the AI Act product-safety classification route."""

    ai_is_product: TriState = TriState.UNKNOWN
    ai_is_safety_component: TriState = TriState.UNKNOWN
    product_type: str | None = None
    annex_i_instrument: str | None = None
    annex_i_instrument_confirmed: TriState = TriState.UNKNOWN
    third_party_conformity_required: TriState = TriState.UNKNOWN

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ProductRegulationFacts:
        """Deserialize the isolated namespace using existing fact conventions."""

        if not isinstance(payload, dict):
            raise TypeError("product_regulation facts must be an object")
        allowed_fields = {
            "ai_is_product",
            "ai_is_safety_component",
            "product_type",
            "annex_i_instrument",
            "annex_i_instrument_confirmed",
            "third_party_conformity_required",
        }
        unknown_fields = set(payload).difference(allowed_fields)
        if unknown_fields:
            raise ValueError(
                "unknown product_regulation fact fields: "
                + ", ".join(sorted(unknown_fields))
            )

        def tri_state(field_name: str) -> TriState:
            value = payload.get(field_name, TriState.UNKNOWN.value)
            return value if isinstance(value, TriState) else TriState(str(value))

        def optional_string(field_name: str) -> str | None:
            value = payload.get(field_name)
            if value is None:
                return None
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string or null")
            normalized = value.strip()
            return normalized or None

        return cls(
            ai_is_product=tri_state("ai_is_product"),
            ai_is_safety_component=tri_state("ai_is_safety_component"),
            product_type=optional_string("product_type"),
            annex_i_instrument=optional_string("annex_i_instrument"),
            annex_i_instrument_confirmed=tri_state(
                "annex_i_instrument_confirmed"
            ),
            third_party_conformity_required=tri_state(
                "third_party_conformity_required"
            ),
        )


@dataclass(slots=True)
class DataProtectionFacts(SerializableModel):
    """Facts reserved for cross-regulation data-protection assessments."""

    personal_data_processed: TriState = TriState.UNKNOWN
    automated_individual_decision: TriState = TriState.UNKNOWN
    special_category_data_processed: TriState = TriState.UNKNOWN


@dataclass(slots=True)
class DataActFacts(SerializableModel):
    """Facts reserved for future EU Data Act relevance assessments."""

    connected_product: TriState = TriState.UNKNOWN
    related_service: TriState = TriState.UNKNOWN
    data_generated: TriState = TriState.UNKNOWN
    data_holder_identified: TriState = TriState.UNKNOWN
    user_or_third_party_access_request: TriState = TriState.UNKNOWN


@dataclass(slots=True)
class FactMetadata(SerializableModel):
    source: FactSource = FactSource.UNKNOWN
    question_id: str | None = None
    recorded_at: datetime | None = None


@dataclass(slots=True)
class AssessmentFacts(SerializableModel):
    """The stable, versioned fact contract consumed by future rules."""

    schema_version: str = "2.0.0"
    case: CaseFacts = field(default_factory=CaseFacts)
    system: SystemFacts = field(default_factory=SystemFacts)
    scope: ScopeFacts = field(default_factory=ScopeFacts)
    organisation: OrganisationFacts = field(default_factory=OrganisationFacts)
    supply_chain: SupplyChainFacts = field(default_factory=SupplyChainFacts)
    use_context: UseContextFacts = field(default_factory=UseContextFacts)
    practices: PracticesFacts = field(default_factory=PracticesFacts)
    high_risk: HighRiskFacts = field(default_factory=HighRiskFacts)
    product_regulation: ProductRegulationFacts = field(
        default_factory=ProductRegulationFacts
    )
    data_protection: DataProtectionFacts = field(
        default_factory=DataProtectionFacts
    )
    data_act: DataActFacts = field(default_factory=DataActFacts)
    fact_metadata: dict[str, FactMetadata] = field(default_factory=dict)
    temporal_context: TemporalContextFacts | None = None
    territorial_context: TerritorialContextFacts | None = None
    actors: list[ActorFacts] | None = None
    ai_systems: list[AISystemFacts] | None = None
    recruitment_workflows: list[RecruitmentWorkflowFacts] | None = None
    processing_operations: list[ProcessingOperationFacts] | None = None
    recruitment_processes: list[RecruitmentDecisionProcessFacts] | None = None
    compliance_artefacts: list[ComplianceArtefactMetadata] | None = None
    retired_entity_ids: list[str] = field(default_factory=list)
    source_schema_version: str | None = None
    _serialized_fields: frozenset[str] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )
    _schema_version_at_construction: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    V2_SCHEMA_VERSION: ClassVar[str] = "2.0.0"
    V3_SCHEMA_VERSION: ClassVar[str] = "3.0.0"
    _V3_FIELDS: ClassVar[tuple[str, ...]] = (
        "temporal_context",
        "territorial_context",
        "actors",
        "ai_systems",
        "recruitment_workflows",
        "processing_operations",
        "recruitment_processes",
        "compliance_artefacts",
        "retired_entity_ids",
        "source_schema_version",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_schema_version_at_construction", self.schema_version
        )
        self.validate_schema_consistency()

    def validate_schema_consistency(self) -> None:
        """Enforce the declared schema at every persistence boundary."""

        if self.schema_version != self._schema_version_at_construction:
            raise ValueError(
                "schema_version cannot be changed in place; use an explicit "
                "compatibility adapter"
            )
        if self.schema_version not in (
            self.V2_SCHEMA_VERSION,
            self.V3_SCHEMA_VERSION,
        ):
            raise ValueError(
                f"unsupported AssessmentFacts schema {self.schema_version!r}"
            )
        if self.schema_version == self.V2_SCHEMA_VERSION:
            populated = [
                name
                for name in self._V3_FIELDS
                if (
                    getattr(self, name) not in (None, [])
                    if name != "retired_entity_ids"
                    else bool(getattr(self, name))
                )
            ]
            if populated:
                raise ValueError(
                    "schema 2.0.0 cannot contain v3 fields: "
                    + ", ".join(populated)
                )
        else:
            self.validate_v3()
        self._validate_source_shape()

    @classmethod
    def new_v3(cls, **changes: object) -> AssessmentFacts:
        """Create an explicitly versioned v3 draft without changing v0.5 defaults."""

        return cls(schema_version=cls.V3_SCHEMA_VERSION, **changes)

    def to_dict(self) -> dict[str, object]:
        self.validate_schema_consistency()
        payload = SerializableModel.to_dict(self)
        if self.schema_version == self.V2_SCHEMA_VERSION:
            for field_name in self._V3_FIELDS:
                payload.pop(field_name, None)
        if self._serialized_fields is not None:
            payload = {
                key: value
                for key, value in payload.items()
                if key in self._serialized_fields
            }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AssessmentFacts:
        """Read v2 or v3 facts without silently adding absent source fields."""

        restored = model_from_dict(cls, payload)
        restored._serialized_fields = frozenset(payload)
        return restored

    def make_editable(self) -> AssessmentFacts:
        """Return an isolated draft that may serialize newly populated fields."""

        clone = deepcopy(self)
        clone._serialized_fields = None
        return clone

    def _validate_source_shape(self) -> None:
        if self._serialized_fields is None:
            return
        model_fields = {item.name: item for item in fields(self)}
        for name, model_field in model_fields.items():
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
                    f"field {name!r} was absent from the source payload; use "
                    "make_editable() or an explicit compatibility adapter "
                    "before populating it"
                )

    def active_entity_ids(self) -> set[str]:
        """Return stable v3 entity identities, never list positions."""

        identifiers = {
            str(identifier)
            for collection, attribute in (
                (self.actors, "actor_id"),
                (self.ai_systems, "system_id"),
                (self.recruitment_workflows, "workflow_id"),
                (self.processing_operations, "processing_operation_id"),
                (self.recruitment_processes, "process_id"),
                (self.compliance_artefacts, "artefact_id"),
            )
            for record in collection or ()
            for identifier in (getattr(record, attribute),)
        }
        identifiers.update(
            str(criterion.criterion_id)
            for process in self.recruitment_processes or ()
            for criterion in process.screening_criteria or ()
        )
        return identifiers

    def validate_v3(self) -> None:
        """Validate stable identities and references for a v3 fact snapshot."""

        if self.schema_version != self.V3_SCHEMA_VERSION:
            raise ValueError("validate_v3 requires AssessmentFacts schema 3.0.0")
        for record in self.actors or ():
            record.normalize_references()
        for record in self.ai_systems or ():
            record.normalize_references()
        for record in self.recruitment_workflows or ():
            record.normalize_references()
        for record in self.processing_operations or ():
            record.normalize_references()
        for record in self.recruitment_processes or ():
            record.normalize_references()
        if self.temporal_context is not None:
            self.temporal_context.normalize_references()
        if self.territorial_context is not None:
            self.territorial_context.normalize_references()
        self.retired_entity_ids = normalized_identifier_list(
            self.retired_entity_ids,
            StableIdentifier,
            field_name="retired_entity_ids",
        ) or []

        all_ids: list[str] = []
        for collection, attribute in (
            (self.actors, "actor_id"),
            (self.ai_systems, "system_id"),
            (self.recruitment_workflows, "workflow_id"),
            (self.processing_operations, "processing_operation_id"),
        ):
            identifiers = [
                str(getattr(record, attribute)) for record in collection or ()
            ]
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"duplicate stable identifiers in {attribute}")
            all_ids.extend(identifiers)
        process_ids = [
            str(item.process_id) for item in self.recruitment_processes or ()
        ]
        artefact_ids = [
            str(item.artefact_id) for item in self.compliance_artefacts or ()
        ]
        criterion_ids = [
            str(criterion.criterion_id)
            for process in self.recruitment_processes or ()
            for criterion in process.screening_criteria or ()
        ]
        for label, identifiers in (
            ("process_id", process_ids),
            ("artefact_id", artefact_ids),
            ("criterion_id", criterion_ids),
        ):
            if len(set(identifiers)) != len(identifiers):
                raise ValueError(f"duplicate stable identifiers in {label}")
        all_ids.extend(process_ids)
        all_ids.extend(artefact_ids)
        all_ids.extend(criterion_ids)
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("stable entity identifiers must be case-wide unique")
        retired = set(self.retired_entity_ids)
        if len(retired) != len(self.retired_entity_ids):
            raise ValueError("retired_entity_ids must not contain duplicates")
        if retired.intersection(all_ids):
            raise ValueError("retired identifiers cannot be reused")

        actor_ids = {str(item.actor_id) for item in self.actors or ()}
        system_ids = {str(item.system_id) for item in self.ai_systems or ()}
        workflow_ids = {
            str(item.workflow_id) for item in self.recruitment_workflows or ()
        }
        operation_ids = {
            str(item.processing_operation_id)
            for item in self.processing_operations or ()
        }
        for actor in self.actors or ():
            for references in (
                actor.operates_system_ids,
                actor.develops_or_commissions_system_ids,
                actor.markets_system_ids_under_own_name,
                actor.uses_system_ids_in_own_organisation,
            ):
                self._reject_unknown_refs(references, system_ids, "actor system")
            for system_id, actor_references in (
                actor.uses_system_ids_on_behalf_of_actor_ids or {}
            ).items():
                self._reject_unknown_refs(
                    [system_id], system_ids, "actor behalf system"
                )
                self._reject_unknown_refs(
                    actor_references, actor_ids, "actor behalf actor"
                )
        for system in self.ai_systems or ():
            for references in (
                system.vendor_actor_ids,
                system.commissioning_actor_ids,
                system.branding_actor_ids,
                system.selected_by_actor_ids,
                system.configured_by_actor_ids,
            ):
                self._reject_unknown_refs(references, actor_ids, "system actor")
        for workflow in self.recruitment_workflows or ():
            self._reject_unknown_refs(
                workflow.system_ids, system_ids, "workflow system"
            )
            self._reject_unknown_refs(
                workflow.processing_operation_ids,
                operation_ids,
                "workflow processing operation",
            )
            for references in (
                workflow.employer_actor_ids,
                workflow.recruiter_actor_ids,
                workflow.output_recipient_actor_ids,
                workflow.final_decision_actor_ids,
            ):
                self._reject_unknown_refs(references, actor_ids, "workflow actor")
        for operation in self.processing_operations or ():
            self._reject_unknown_refs(
                operation.system_ids, system_ids, "operation system"
            )
            self._reject_unknown_refs(
                operation.participating_actor_ids,
                actor_ids,
                "operation actor",
            )
            self._reject_unknown_refs(
                operation.recipients,
                actor_ids,
                "operation recipient",
            )
            if (
                operation.workflow_id is not None
                and str(operation.workflow_id) not in workflow_ids
            ):
                raise ValueError(
                    f"unknown operation workflow reference {operation.workflow_id!r}"
                )
        for process in self.recruitment_processes or ():
            self._reject_optional_ref(
                process.workflow_id, workflow_ids, "process workflow"
            )
            self._reject_optional_ref(
                process.processing_operation_id,
                operation_ids,
                "process processing operation",
            )
            self._reject_optional_ref(
                process.system_id, system_ids, "process system"
            )
            for criterion in process.screening_criteria or ():
                self._reject_unknown_refs(
                    criterion.selecting_actor_ids,
                    actor_ids,
                    "criterion selecting actor",
                )
                self._reject_unknown_refs(
                    criterion.configuring_actor_ids,
                    actor_ids,
                    "criterion configuring actor",
                )
        if self.temporal_context is not None:
            self._reject_unknown_refs(
                list((self.temporal_context.intended_use_dates or {}).keys()),
                workflow_ids,
                "temporal workflow",
            )
            self._reject_unknown_refs(
                list((self.temporal_context.put_into_service_dates or {}).keys()),
                system_ids,
                "temporal system",
            )
            self._reject_unknown_refs(
                list((self.temporal_context.operation_dates or {}).keys()),
                operation_ids,
                "temporal operation",
            )
        if self.territorial_context is not None:
            for mapping, known, label in (
                (
                    self.territorial_context.actor_establishment_locations,
                    actor_ids,
                    "territorial actor",
                ),
                (
                    self.territorial_context.system_use_locations,
                    system_ids,
                    "territorial system",
                ),
                (
                    self.territorial_context.output_use_locations,
                    workflow_ids,
                    "territorial output workflow",
                ),
                (
                    self.territorial_context.affected_person_locations,
                    workflow_ids,
                    "territorial affected-person workflow",
                ),
                (
                    self.territorial_context.processing_operation_context,
                    operation_ids,
                    "territorial operation",
                ),
            ):
                self._reject_unknown_refs(
                    list((mapping or {}).keys()), known, label
                )
        for artefact in self.compliance_artefacts or ():
            self._validate_scope(
                artefact.scope,
                actor_ids=actor_ids,
                system_ids=system_ids,
                workflow_ids=workflow_ids,
                operation_ids=operation_ids,
                label="compliance artefact",
            )

    @classmethod
    def _validate_scope(
        cls,
        scope: object,
        *,
        actor_ids: set[str],
        system_ids: set[str],
        workflow_ids: set[str],
        operation_ids: set[str],
        label: str,
    ) -> None:
        cls._reject_optional_ref(scope.actor_id, actor_ids, f"{label} actor")
        cls._reject_optional_ref(scope.system_id, system_ids, f"{label} system")
        cls._reject_optional_ref(
            scope.workflow_id, workflow_ids, f"{label} workflow"
        )
        cls._reject_optional_ref(
            scope.processing_operation_id,
            operation_ids,
            f"{label} processing operation",
        )

    @staticmethod
    def _reject_optional_ref(
        reference: object | None,
        known: set[str],
        label: str,
    ) -> None:
        if reference is not None and str(reference) not in known:
            raise ValueError(f"unknown {label} reference {reference!r}")

    @staticmethod
    def _reject_unknown_refs(
        references: list[object] | None,
        known: set[str],
        label: str,
    ) -> None:
        if references is None:
            return
        unknown = sorted(
            str(reference)
            for reference in references
            if str(reference) not in known
        )
        if unknown:
            raise ValueError(f"unknown {label} references: {unknown!r}")
