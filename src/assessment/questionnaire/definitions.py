"""Declarative questionnaire definitions for implemented assessment rules."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from src.assessment.facts import AffectedPerson, AssessmentFacts, UseDomain
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import TriState
from src.assessment.questionnaire.models import AnswerType, QuestionOption
from src.assessment.questionnaire.registry import QuestionRegistry
from src.assessment.questionnaire.routing_models import (
    EligibilityHintGroup,
    FactCondition,
    FactConditionOperator,
    LocalizedTextKeys,
    QuestionDependency,
    QuestionInvalidation,
    RoutingQuestionDefinition,
    RuleQuestionnaireDefinition,
    UnsupportedPathDefinition,
)
from src.assessment.rules import AssessmentRule, RuleRegistry


AI_ACT_EMPLOYMENT_RULE_ID = "AI_ACT_HIGH_RISK_EMPLOYMENT"
GDPR_ARTICLE22_RULE_ID = "GDPR_ARTICLE22_RELEVANCE"
EU_DATA_ACT_RULE_ID = "EU_DATA_ACT_RELEVANCE"

HINT_RECRUITMENT = "employment.recruitment"
HINT_SELECTION = "employment.selection"
HINT_CANDIDATE_RANKING = "employment.candidate_ranking"
HINT_WORKER_MANAGEMENT = "employment.worker_management"
HINT_INDIVIDUAL_SIGNIFICANT_DECISION = "decision.individual_significant"
HINT_CREDIT_DECISION = "decision.credit"
HINT_INDUSTRIAL_CONNECTED_EQUIPMENT = "data_act.industrial_connected_equipment"
HINT_PRODUCT_SAFETY_COMPONENT = "ai_act.product_safety_component"


def _text_keys(key: str) -> LocalizedTextKeys:
    return LocalizedTextKeys(
        en_label_key=f"question.{key}.label.en",
        en_help_key=f"question.{key}.help.en",
        zh_cn_label_key=f"question.{key}.label.zh_cn",
        zh_cn_help_key=f"question.{key}.help.zh_cn",
    )


def _tri_state_options(key: str) -> tuple[QuestionOption, ...]:
    return tuple(
        QuestionOption(
            value=value.value,
            label=f"question.{key}.option.{value.value}",
        )
        for value in TriState
    )


def _confirmation_options(key: str) -> tuple[QuestionOption, ...]:
    return (
        QuestionOption(
            value="confirm",
            label=f"question.{key}.option.confirm",
        ),
        QuestionOption(
            value="deselect",
            label=f"question.{key}.option.deselect",
        ),
    )


QUESTION_DEFINITIONS: tuple[RoutingQuestionDefinition, ...] = (
    RoutingQuestionDefinition(
        question_id="INTAKE-SYSTEM-NAME",
        fact_path="system.name",
        answer_type=AnswerType.TEXT,
        text_keys=_text_keys("system_name"),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="INTAKE-SYSTEM-PURPOSE",
        fact_path="system.intended_purpose",
        answer_type=AnswerType.TEXT,
        text_keys=_text_keys("system_purpose"),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="INTAKE-USE-DOMAIN",
        fact_path="use_context.domain",
        answer_type=AnswerType.SINGLE_CHOICE,
        text_keys=_text_keys("use_domain"),
        options=tuple(
            QuestionOption(
                value=domain.value,
                label=f"question.use_domain.option.{domain.value}",
            )
            for domain in UseDomain
        ),
        invalidations=(
            QuestionInvalidation(
                fact_paths=(
                    "use_context.task",
                    "use_context.materially_influences_decision",
                )
            ),
        ),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="INTAKE-USE-TASK",
        fact_path="use_context.task",
        answer_type=AnswerType.TEXT,
        text_keys=_text_keys("use_task"),
        dependencies=(QuestionDependency("use_context.domain"),),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="INTAKE-AFFECTED-PERSONS",
        fact_path="use_context.affected_persons",
        answer_type=AnswerType.MULTIPLE_CHOICE,
        text_keys=_text_keys("affected_persons"),
        options=tuple(
            QuestionOption(
                value=person.value,
                label=f"question.affected_persons.option.{person.value}",
            )
            for person in AffectedPerson
        ),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="INTAKE-DECISION-IMPACT",
        fact_path="use_context.materially_influences_decision",
        answer_type=AnswerType.TRI_STATE,
        text_keys=_text_keys("decision_impact"),
        options=_tri_state_options("decision_impact"),
        dependencies=(QuestionDependency("use_context.domain"),),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="INTAKE-HUMAN-REVIEW",
        fact_path="use_context.human_review_before_effect",
        answer_type=AnswerType.TRI_STATE,
        text_keys=_text_keys("human_review_before_effect"),
        options=_tri_state_options("human_review_before_effect"),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="INTAKE-PERSONAL-DATA",
        fact_path="data_protection.personal_data_processed",
        answer_type=AnswerType.TRI_STATE,
        text_keys=_text_keys("personal_data_processed"),
        options=_tri_state_options("personal_data_processed"),
        invalidations=(
            QuestionInvalidation(
                fact_paths=("data_protection.automated_individual_decision",)
            ),
        ),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="GDPR-AUTOMATED-DECISION",
        fact_path="data_protection.automated_individual_decision",
        answer_type=AnswerType.TRI_STATE,
        text_keys=_text_keys("automated_individual_decision"),
        options=_tri_state_options("automated_individual_decision"),
        dependencies=(
            QuestionDependency(
                "data_protection.personal_data_processed",
                accepted_values=(TriState.YES.value, TriState.UNKNOWN.value),
            ),
        ),
    ),
    RoutingQuestionDefinition(
        question_id="INTAKE-CONNECTED-PRODUCT",
        fact_path="data_act.connected_product",
        answer_type=AnswerType.TRI_STATE,
        text_keys=_text_keys("connected_product"),
        options=_tri_state_options("connected_product"),
        invalidations=(
            QuestionInvalidation(
                fact_paths=("data_act.related_service", "data_act.data_generated")
            ),
        ),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="INTAKE-RELATED-SERVICE",
        fact_path="data_act.related_service",
        answer_type=AnswerType.TRI_STATE,
        text_keys=_text_keys("related_service"),
        options=_tri_state_options("related_service"),
        dependencies=(QuestionDependency("data_act.connected_product"),),
        universal=True,
    ),
    RoutingQuestionDefinition(
        question_id="DATA-ACT-DATA-GENERATED",
        fact_path="data_act.data_generated",
        answer_type=AnswerType.TRI_STATE,
        text_keys=_text_keys("data_generated"),
        options=_tri_state_options("data_generated"),
        dependencies=(
            QuestionDependency("data_act.connected_product"),
            QuestionDependency("data_act.related_service"),
        ),
    ),
)


def _equals(path: str, value: str) -> FactCondition:
    return FactCondition(
        fact_path=path,
        operator=FactConditionOperator.EQUALS,
        expected_values=(value,),
    )


RULE_QUESTIONNAIRE_DEFINITIONS: tuple[RuleQuestionnaireDefinition, ...] = (
    RuleQuestionnaireDefinition(
        rule_id=AI_ACT_EMPLOYMENT_RULE_ID,
        framework=RegulatoryFramework.EU_AI_ACT,
        display_module_key="module.ai_act.employment",
        confirmation_question_id=(
            "CONFIRM-MODULE::AI_ACT_HIGH_RISK_EMPLOYMENT"
        ),
        confirmation_text_keys=_text_keys("confirm_ai_act_employment"),
        confirmation_answer_type=AnswerType.SINGLE_CHOICE,
        confirmation_options=_confirmation_options("confirm_ai_act_employment"),
        eligibility_fact_paths=("use_context.domain", "use_context.task"),
        required_fact_paths=(
            "use_context.domain",
            "use_context.task",
            "use_context.materially_influences_decision",
        ),
        question_ids=(
            "INTAKE-USE-DOMAIN",
            "INTAKE-USE-TASK",
            "INTAKE-DECISION-IMPACT",
        ),
        supported_domains=(UseDomain.EMPLOYMENT,),
        routing_hints=(
            HINT_RECRUITMENT,
            HINT_SELECTION,
            HINT_CANDIDATE_RANKING,
            HINT_WORKER_MANAGEMENT,
        ),
        eligibility_groups=(
            EligibilityHintGroup(
                reason_code="EMPLOYMENT_CONTEXT_AND_CONFIRMED_FUNCTION",
                all_conditions=(
                    _equals("use_context.domain", UseDomain.EMPLOYMENT.value),
                ),
                any_routing_hints=(
                    HINT_RECRUITMENT,
                    HINT_SELECTION,
                    HINT_CANDIDATE_RANKING,
                    HINT_WORKER_MANAGEMENT,
                ),
            ),
        ),
        dependency_metadata=(
            QuestionDependency("use_context.domain"),
            QuestionDependency("use_context.task"),
        ),
    ),
    RuleQuestionnaireDefinition(
        rule_id=GDPR_ARTICLE22_RULE_ID,
        framework=RegulatoryFramework.GDPR,
        display_module_key="module.gdpr.article22",
        confirmation_question_id="CONFIRM-MODULE::GDPR_ARTICLE22_RELEVANCE",
        confirmation_text_keys=_text_keys("confirm_gdpr_article22"),
        confirmation_answer_type=AnswerType.SINGLE_CHOICE,
        confirmation_options=_confirmation_options("confirm_gdpr_article22"),
        eligibility_fact_paths=(
            "data_protection.personal_data_processed",
            "data_protection.automated_individual_decision",
            "use_context.materially_influences_decision",
        ),
        required_fact_paths=(
            "data_protection.personal_data_processed",
            "data_protection.automated_individual_decision",
            "use_context.materially_influences_decision",
        ),
        question_ids=(
            "INTAKE-PERSONAL-DATA",
            "GDPR-AUTOMATED-DECISION",
            "INTAKE-DECISION-IMPACT",
            "INTAKE-HUMAN-REVIEW",
        ),
        supported_domains=(
            UseDomain.EMPLOYMENT,
            UseDomain.ESSENTIAL_SERVICES,
            UseDomain.EDUCATION,
            UseDomain.JUSTICE_DEMOCRATIC_PROCESSES,
            UseDomain.OTHER,
        ),
        routing_hints=(
            HINT_INDIVIDUAL_SIGNIFICANT_DECISION,
            HINT_CREDIT_DECISION,
            HINT_RECRUITMENT,
        ),
        eligibility_groups=(
            EligibilityHintGroup(
                reason_code="PERSONAL_DATA_AND_AUTOMATED_SIGNIFICANT_DECISION",
                all_conditions=(
                    _equals(
                        "data_protection.personal_data_processed",
                        TriState.YES.value,
                    ),
                    _equals(
                        "data_protection.automated_individual_decision",
                        TriState.YES.value,
                    ),
                    _equals(
                        "use_context.materially_influences_decision",
                        TriState.YES.value,
                    ),
                ),
            ),
            EligibilityHintGroup(
                reason_code="PERSONAL_DATA_AND_CONFIRMED_INDIVIDUAL_DECISION_CONTEXT",
                all_conditions=(
                    _equals(
                        "data_protection.personal_data_processed",
                        TriState.YES.value,
                    ),
                ),
                any_routing_hints=(
                    HINT_INDIVIDUAL_SIGNIFICANT_DECISION,
                    HINT_CREDIT_DECISION,
                    HINT_RECRUITMENT,
                ),
            ),
        ),
        dependency_metadata=(
            QuestionDependency("data_protection.personal_data_processed"),
            QuestionDependency("use_context.materially_influences_decision"),
        ),
    ),
    RuleQuestionnaireDefinition(
        rule_id=EU_DATA_ACT_RULE_ID,
        framework=RegulatoryFramework.EU_DATA_ACT,
        display_module_key="module.data_act.relevance",
        confirmation_question_id="CONFIRM-MODULE::EU_DATA_ACT_RELEVANCE",
        confirmation_text_keys=_text_keys("confirm_data_act_relevance"),
        confirmation_answer_type=AnswerType.SINGLE_CHOICE,
        confirmation_options=_confirmation_options("confirm_data_act_relevance"),
        eligibility_fact_paths=(
            "data_act.connected_product",
            "data_act.related_service",
            "data_act.data_generated",
        ),
        required_fact_paths=(
            "data_act.connected_product",
            "data_act.related_service",
            "data_act.data_generated",
        ),
        question_ids=(
            "INTAKE-CONNECTED-PRODUCT",
            "INTAKE-RELATED-SERVICE",
            "DATA-ACT-DATA-GENERATED",
        ),
        supported_domains=(UseDomain.PRODUCT_SAFETY, UseDomain.OTHER),
        routing_hints=(HINT_INDUSTRIAL_CONNECTED_EQUIPMENT,),
        eligibility_groups=(
            EligibilityHintGroup(
                reason_code="CONNECTED_PRODUCT_OR_SERVICE_GENERATES_DATA",
                all_conditions=(
                    _equals("data_act.data_generated", TriState.YES.value),
                ),
                any_conditions=(
                    _equals("data_act.connected_product", TriState.YES.value),
                    _equals("data_act.related_service", TriState.YES.value),
                ),
            ),
            EligibilityHintGroup(
                reason_code="CONFIRMED_CONNECTED_EQUIPMENT_CONTEXT",
                any_conditions=(
                    _equals("data_act.connected_product", TriState.YES.value),
                    _equals("data_act.related_service", TriState.YES.value),
                ),
                any_routing_hints=(HINT_INDUSTRIAL_CONNECTED_EQUIPMENT,),
            ),
        ),
        dependency_metadata=(
            QuestionDependency("data_act.connected_product"),
            QuestionDependency("data_act.related_service"),
        ),
    ),
)


UNSUPPORTED_PATH_DEFINITIONS: tuple[UnsupportedPathDefinition, ...] = (
    UnsupportedPathDefinition(
        path_id="AI_ACT_ESSENTIAL_SERVICES_CREDIT_UNSUPPORTED",
        framework=RegulatoryFramework.EU_AI_ACT,
        display_module_key="module.ai_act.credit_essential_services",
        message_keys=_text_keys("unsupported_ai_act_credit"),
        eligibility_groups=(
            EligibilityHintGroup(
                reason_code="ESSENTIAL_SERVICES_CREDIT_CONTEXT",
                all_conditions=(
                    _equals(
                        "use_context.domain",
                        UseDomain.ESSENTIAL_SERVICES.value,
                    ),
                ),
                any_routing_hints=(HINT_CREDIT_DECISION,),
            ),
        ),
    ),
    UnsupportedPathDefinition(
        path_id="AI_ACT_JUDICIAL_ROUTE_UNSUPPORTED",
        framework=RegulatoryFramework.EU_AI_ACT,
        display_module_key="module.ai_act.judicial",
        message_keys=_text_keys("unsupported_ai_act_judicial"),
        eligibility_groups=(
            EligibilityHintGroup(
                reason_code="JUSTICE_OR_DEMOCRATIC_PROCESS_CONTEXT",
                all_conditions=(
                    _equals(
                        "use_context.domain",
                        UseDomain.JUSTICE_DEMOCRATIC_PROCESSES.value,
                    ),
                ),
            ),
        ),
    ),
    UnsupportedPathDefinition(
        path_id="AI_ACT_PRODUCT_SAFETY_ROUTE_UNSUPPORTED",
        framework=RegulatoryFramework.EU_AI_ACT,
        display_module_key="module.ai_act.product_safety",
        message_keys=_text_keys("unsupported_ai_act_product_safety"),
        eligibility_groups=(
            EligibilityHintGroup(
                reason_code="CONFIRMED_PRODUCT_SAFETY_COMPONENT_CONTEXT",
                all_conditions=(
                    _equals(
                        "use_context.domain",
                        UseDomain.PRODUCT_SAFETY.value,
                    ),
                ),
                any_routing_hints=(HINT_PRODUCT_SAFETY_COMPONENT,),
            ),
        ),
    ),
)


class RuleQuestionnaireRegistry:
    """Validated companion registry preserving authored rule order."""

    def __init__(
        self,
        rule_registry: RuleRegistry,
        question_registry: QuestionRegistry,
        definitions: Iterable[RuleQuestionnaireDefinition],
    ) -> None:
        if not isinstance(rule_registry, RuleRegistry):
            raise TypeError("rule_registry must be a RuleRegistry")
        if not isinstance(question_registry, QuestionRegistry):
            raise TypeError("question_registry must be a QuestionRegistry")
        self._rule_registry = rule_registry
        self._question_registry = question_registry
        self._definitions: dict[str, RuleQuestionnaireDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(
        self,
        definition: RuleQuestionnaireDefinition,
    ) -> RuleQuestionnaireDefinition:
        if not isinstance(definition, RuleQuestionnaireDefinition):
            raise TypeError("definition must be a RuleQuestionnaireDefinition")
        if definition.rule_id in self._definitions:
            raise ValueError(
                f"questionnaire definition {definition.rule_id!r} already exists"
            )
        rule = self._rule_registry.get(definition.rule_id)
        self._validate_against_rule(definition, rule)
        for question_id in definition.question_ids:
            self._question_registry.get(question_id)
        self._definitions[definition.rule_id] = definition
        return definition

    def get(self, rule_id: str) -> RuleQuestionnaireDefinition:
        try:
            return self._definitions[rule_id]
        except KeyError as exc:
            raise KeyError(
                f"questionnaire definition for rule {rule_id!r} was not found"
            ) from exc

    def all(self) -> tuple[RuleQuestionnaireDefinition, ...]:
        return tuple(self._definitions.values())

    def __iter__(self) -> Iterator[RuleQuestionnaireDefinition]:
        return iter(self._definitions.values())

    def __len__(self) -> int:
        return len(self._definitions)

    @staticmethod
    def _validate_against_rule(
        definition: RuleQuestionnaireDefinition,
        rule: AssessmentRule,
    ) -> None:
        if definition.framework is not rule.framework:
            raise ValueError(
                f"framework mismatch for rule {definition.rule_id!r}"
            )
        if definition.required_fact_paths != rule.required_fact_paths:
            raise ValueError(
                f"required facts for {definition.rule_id!r} do not match the rule"
            )


def build_question_registry() -> QuestionRegistry:
    """Compile authored definitions into the existing question registry."""

    return QuestionRegistry(
        definition.as_question() for definition in QUESTION_DEFINITIONS
    )


def build_rule_questionnaire_registry(
    rule_registry: RuleRegistry,
    question_registry: QuestionRegistry | None = None,
) -> RuleQuestionnaireRegistry:
    """Build the validated default companion registry."""

    questions = question_registry or build_question_registry()
    return RuleQuestionnaireRegistry(
        rule_registry,
        questions,
        RULE_QUESTIONNAIRE_DEFINITIONS,
    )


def question_definitions_by_id() -> dict[str, RoutingQuestionDefinition]:
    """Return an insertion-ordered lookup for invalidation and presentation."""

    return {definition.question_id: definition for definition in QUESTION_DEFINITIONS}


def universal_question_ids() -> tuple[str, ...]:
    return tuple(
        definition.question_id
        for definition in QUESTION_DEFINITIONS
        if definition.universal
    )


def validate_definition_fact_paths() -> None:
    """Fail fast if declarative metadata references an unknown fact path."""

    facts = AssessmentFacts()
    paths = {
        definition.fact_path for definition in QUESTION_DEFINITIONS
    }
    paths.update(
        path
        for definition in RULE_QUESTIONNAIRE_DEFINITIONS
        for path in (
            *definition.eligibility_fact_paths,
            *definition.required_fact_paths,
        )
    )
    for fact_path in paths:
        current: object = facts
        for segment in fact_path.split("."):
            if not segment or not hasattr(current, segment):
                raise ValueError(f"unknown questionnaire fact path {fact_path!r}")
            current = getattr(current, segment)


validate_definition_fact_paths()
