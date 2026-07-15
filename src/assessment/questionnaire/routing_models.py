"""Domain models for deterministic questionnaire-module routing."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.assessment.facts import UseDomain
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import SerializableModel
from src.assessment.questionnaire.models import AnswerType, Question, QuestionOption


class FactConditionOperator(str, Enum):
    """Supported deterministic comparisons for eligibility facts."""

    EQUALS = "equals"
    IN = "in"


class QuestionResponseState(str, Enum):
    """Inspectable state for a persisted questionnaire response."""

    ANSWERED = "answered"
    EXPLICIT_UNKNOWN = "explicit_unknown"


@dataclass(frozen=True, slots=True)
class LocalizedTextKeys(SerializableModel):
    """Translation catalogue keys for one bilingual question or message."""

    en_label_key: str
    en_help_key: str
    zh_cn_label_key: str
    zh_cn_help_key: str

    def __post_init__(self) -> None:
        for field_name in (
            "en_label_key",
            "en_help_key",
            "zh_cn_label_key",
            "zh_cn_help_key",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class QuestionDependency(SerializableModel):
    """One upstream fact that gives a question its current context."""

    fact_path: str
    accepted_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.fact_path, str) or not self.fact_path.strip():
            raise ValueError("fact_path must be a non-empty string")
        if any(not isinstance(value, str) or not value for value in self.accepted_values):
            raise ValueError("accepted_values must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class QuestionInvalidation(SerializableModel):
    """Facts that may become stale when the owning answer changes."""

    fact_paths: tuple[str, ...]
    preserve_explicitly_confirmed: bool = False

    def __post_init__(self) -> None:
        if not self.fact_paths:
            raise ValueError("fact_paths must not be empty")
        if any(not isinstance(path, str) or not path.strip() for path in self.fact_paths):
            raise ValueError("fact_paths must contain non-empty strings")
        if not isinstance(self.preserve_explicitly_confirmed, bool):
            raise TypeError("preserve_explicitly_confirmed must be a bool")


@dataclass(frozen=True, slots=True)
class RoutingQuestionDefinition(SerializableModel):
    """Module-neutral metadata compiled into the existing Question model."""

    question_id: str
    fact_path: str
    answer_type: AnswerType
    text_keys: LocalizedTextKeys
    options: tuple[QuestionOption, ...] = ()
    dependencies: tuple[QuestionDependency, ...] = ()
    any_dependencies: tuple[QuestionDependency, ...] = ()
    invalidations: tuple[QuestionInvalidation, ...] = ()
    universal: bool = False

    def __post_init__(self) -> None:
        for field_name in ("question_id", "fact_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.answer_type, AnswerType):
            raise TypeError("answer_type must be an AnswerType")
        if not isinstance(self.text_keys, LocalizedTextKeys):
            raise TypeError("text_keys must be LocalizedTextKeys")
        if any(not isinstance(item, QuestionOption) for item in self.options):
            raise TypeError("options must contain QuestionOption values")
        if any(not isinstance(item, QuestionDependency) for item in self.dependencies):
            raise TypeError("dependencies must contain QuestionDependency values")
        if any(
            not isinstance(item, QuestionDependency)
            for item in self.any_dependencies
        ):
            raise TypeError(
                "any_dependencies must contain QuestionDependency values"
            )
        if any(not isinstance(item, QuestionInvalidation) for item in self.invalidations):
            raise TypeError("invalidations must contain QuestionInvalidation values")
        if not isinstance(self.universal, bool):
            raise TypeError("universal must be a bool")

    def as_question(self) -> Question:
        """Compile into the existing questionnaire engine contract."""

        return Question(
            question_id=self.question_id,
            text=self.text_keys.en_label_key,
            fact_path=self.fact_path,
            answer_type=self.answer_type,
            options=self.options,
            required=True,
            legal_relevance=(),
        )


@dataclass(frozen=True, slots=True)
class FactCondition(SerializableModel):
    """One canonical fact comparison used by an eligibility group."""

    fact_path: str
    operator: FactConditionOperator
    expected_values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fact_path, str) or not self.fact_path.strip():
            raise ValueError("fact_path must be a non-empty string")
        if not isinstance(self.operator, FactConditionOperator):
            raise TypeError("operator must be a FactConditionOperator")
        if not self.expected_values or any(
            not isinstance(value, str) or not value
            for value in self.expected_values
        ):
            raise ValueError("expected_values must contain non-empty strings")
        if self.operator is FactConditionOperator.EQUALS and len(self.expected_values) != 1:
            raise ValueError("equals conditions require exactly one expected value")


@dataclass(frozen=True, slots=True)
class EligibilityHintGroup(SerializableModel):
    """One sufficient, declarative route to a module suggestion."""

    reason_code: str
    all_conditions: tuple[FactCondition, ...] = ()
    any_conditions: tuple[FactCondition, ...] = ()
    any_routing_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("reason_code must be a non-empty string")
        if not (self.all_conditions or self.any_conditions or self.any_routing_hints):
            raise ValueError("an eligibility group must declare at least one hint")
        if any(not isinstance(item, FactCondition) for item in self.all_conditions):
            raise TypeError("all_conditions must contain FactCondition values")
        if any(not isinstance(item, FactCondition) for item in self.any_conditions):
            raise TypeError("any_conditions must contain FactCondition values")
        if any(not isinstance(item, str) or not item for item in self.any_routing_hints):
            raise ValueError("any_routing_hints must contain non-empty strings")


@dataclass(frozen=True, slots=True)
class RuleQuestionnaireDefinition(SerializableModel):
    """Companion questionnaire metadata for one implemented legal rule."""

    rule_id: str
    framework: RegulatoryFramework
    display_module_key: str
    confirmation_question_id: str
    confirmation_text_keys: LocalizedTextKeys
    confirmation_answer_type: AnswerType
    confirmation_options: tuple[QuestionOption, ...]
    eligibility_fact_paths: tuple[str, ...]
    required_fact_paths: tuple[str, ...]
    question_ids: tuple[str, ...]
    supported_domains: tuple[UseDomain, ...] = ()
    routing_hints: tuple[str, ...] = ()
    eligibility_groups: tuple[EligibilityHintGroup, ...] = ()
    dependency_metadata: tuple[QuestionDependency, ...] = ()
    unsupported_path_ids: tuple[str, ...] = ()
    supplemental_question_ids: tuple[str, ...] = ()
    boundary_note_key: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "rule_id",
            "display_module_key",
            "confirmation_question_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.framework, RegulatoryFramework):
            raise TypeError("framework must be a RegulatoryFramework")
        if not isinstance(self.confirmation_text_keys, LocalizedTextKeys):
            raise TypeError("confirmation_text_keys must be LocalizedTextKeys")
        if not isinstance(self.confirmation_answer_type, AnswerType):
            raise TypeError("confirmation_answer_type must be an AnswerType")
        if not self.confirmation_options or any(
            not isinstance(option, QuestionOption)
            for option in self.confirmation_options
        ):
            raise TypeError(
                "confirmation_options must contain QuestionOption values"
            )
        for field_name in (
            "eligibility_fact_paths",
            "required_fact_paths",
            "question_ids",
        ):
            values = getattr(self, field_name)
            if not values or any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
        if any(not isinstance(domain, UseDomain) for domain in self.supported_domains):
            raise TypeError("supported_domains must contain UseDomain values")
        if any(not isinstance(value, str) or not value for value in self.routing_hints):
            raise ValueError("routing_hints must contain non-empty strings")
        if any(not isinstance(group, EligibilityHintGroup) for group in self.eligibility_groups):
            raise TypeError("eligibility_groups must contain EligibilityHintGroup values")
        if any(
            not isinstance(item, QuestionDependency)
            for item in self.dependency_metadata
        ):
            raise TypeError(
                "dependency_metadata must contain QuestionDependency values"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.supplemental_question_ids
        ):
            raise ValueError(
                "supplemental_question_ids must contain non-empty strings"
            )
        if self.boundary_note_key is not None and (
            not isinstance(self.boundary_note_key, str)
            or not self.boundary_note_key.strip()
        ):
            raise ValueError("boundary_note_key must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class UnsupportedPathDefinition(SerializableModel):
    """Recognized legal route for which no assessment rule exists."""

    path_id: str
    framework: RegulatoryFramework
    display_module_key: str
    message_keys: LocalizedTextKeys
    eligibility_groups: tuple[EligibilityHintGroup, ...]

    def __post_init__(self) -> None:
        for field_name in ("path_id", "display_module_key"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.framework, RegulatoryFramework):
            raise TypeError("framework must be a RegulatoryFramework")
        if not isinstance(self.message_keys, LocalizedTextKeys):
            raise TypeError("message_keys must be LocalizedTextKeys")
        if not self.eligibility_groups:
            raise ValueError("eligibility_groups must not be empty")


@dataclass(frozen=True, slots=True)
class UnsupportedPathRoute(SerializableModel):
    """One unsupported route reached by current deterministic facts."""

    path_id: str
    framework: RegulatoryFramework
    display_module_key: str
    message_keys: LocalizedTextKeys
    routing_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FactProvenance(SerializableModel):
    """Questionnaire provenance used for dependency invalidation."""

    fact_path: str
    question_id: str
    module_id: str | None = None
    explicitly_confirmed: bool = False
    depends_on: tuple[str, ...] = ()
    response_state: QuestionResponseState = QuestionResponseState.ANSWERED

    def __post_init__(self) -> None:
        for field_name in ("fact_path", "question_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.module_id is not None and not self.module_id.strip():
            raise ValueError("module_id must be non-empty when provided")
        if not isinstance(self.explicitly_confirmed, bool):
            raise TypeError("explicitly_confirmed must be a bool")
        if not isinstance(self.response_state, QuestionResponseState):
            raise TypeError("response_state must be a QuestionResponseState")
        if any(not isinstance(path, str) or not path for path in self.depends_on):
            raise ValueError("depends_on must contain non-empty strings")


@dataclass(slots=True)
class QuestionnaireRoute(SerializableModel):
    """Deterministic questionnaire routing output; never a legal finding."""

    suggested_modules: list[str] = field(default_factory=list)
    confirmed_modules: list[str] = field(default_factory=list)
    unsupported_modules: list[UnsupportedPathRoute] = field(default_factory=list)
    screened_out_modules: list[str] = field(default_factory=list)
    missing_fact_paths: dict[str, list[str]] = field(default_factory=dict)
    next_questions: list[Question] = field(default_factory=list)
    module_confirmation_question_ids: list[str] = field(default_factory=list)
    ordered_step_ids: list[str] = field(default_factory=list)
    routing_reasons: dict[str, list[str]] = field(default_factory=dict)
    recorded_unknown_question_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InvalidationResult(SerializableModel):
    """Pure description of stale questionnaire state for a caller to apply."""

    changed_upstream_fact_paths: list[str] = field(default_factory=list)
    stale_fact_paths: list[str] = field(default_factory=list)
    invalidated_question_ids: list[str] = field(default_factory=list)
    invalidated_module_ids: list[str] = field(default_factory=list)
    removed_provenance_fact_paths: list[str] = field(default_factory=list)
    reasons: dict[str, list[str]] = field(default_factory=dict)
