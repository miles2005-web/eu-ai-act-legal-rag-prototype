"""Typed fact schema for an EU AI Act assessment case."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum

from src.assessment.models import SerializableModel, TriState


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
    data_protection: DataProtectionFacts = field(
        default_factory=DataProtectionFacts
    )
    data_act: DataActFacts = field(default_factory=DataActFacts)
    fact_metadata: dict[str, FactMetadata] = field(default_factory=dict)
