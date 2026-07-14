"""Deterministic routing over canonical facts and confirmed routing hints."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from src.assessment.facts import AssessmentFacts
from src.assessment.models import TriState
from src.assessment.questionnaire.definitions import (
    QUESTION_DEFINITIONS,
    UNSUPPORTED_PATH_DEFINITIONS,
    RuleQuestionnaireRegistry,
    build_question_registry,
    build_rule_questionnaire_registry,
    question_definitions_by_id,
    universal_question_ids,
)
from src.assessment.questionnaire.engine import QuestionnaireEngine
from src.assessment.questionnaire.models import Question
from src.assessment.questionnaire.registry import QuestionRegistry
from src.assessment.questionnaire.routing_models import (
    EligibilityHintGroup,
    FactCondition,
    FactConditionOperator,
    FactProvenance,
    QuestionnaireRoute,
    RoutingQuestionDefinition,
    UnsupportedPathDefinition,
    UnsupportedPathRoute,
)
from src.assessment.requirements import FactRequirementValidator
from src.assessment.rules import (
    AIActHighRiskEmploymentRule,
    EUDataActRelevanceRule,
    GDPRArticle22RelevanceRule,
    RuleRegistry,
)


class QuestionnaireRouter:
    """Route questions without evaluating legal rules or producing findings."""

    def __init__(
        self,
        *,
        rule_registry: RuleRegistry,
        definition_registry: RuleQuestionnaireRegistry,
        question_registry: QuestionRegistry,
        unsupported_definitions: Iterable[UnsupportedPathDefinition] = (),
        requirement_validator: FactRequirementValidator | None = None,
    ) -> None:
        if not isinstance(rule_registry, RuleRegistry):
            raise TypeError("rule_registry must be a RuleRegistry")
        if not isinstance(definition_registry, RuleQuestionnaireRegistry):
            raise TypeError(
                "definition_registry must be a RuleQuestionnaireRegistry"
            )
        if not isinstance(question_registry, QuestionRegistry):
            raise TypeError("question_registry must be a QuestionRegistry")
        self._rule_registry = rule_registry
        self._definitions = definition_registry
        self._questions = question_registry
        self._questionnaire_engine = QuestionnaireEngine(question_registry)
        self._unsupported = tuple(unsupported_definitions)
        if any(
            not isinstance(item, UnsupportedPathDefinition)
            for item in self._unsupported
        ):
            raise TypeError(
                "unsupported_definitions must contain UnsupportedPathDefinition values"
            )
        self._validator = requirement_validator or FactRequirementValidator()
        if not isinstance(self._validator, FactRequirementValidator):
            raise TypeError(
                "requirement_validator must be a FactRequirementValidator"
            )
        self._question_metadata = question_definitions_by_id()

    def route(
        self,
        facts: AssessmentFacts,
        *,
        confirmed_modules: Iterable[str] = (),
        confirmed_routing_hints: Iterable[str] = (),
        fact_provenance: Iterable[FactProvenance] = (),
    ) -> QuestionnaireRoute:
        """Return a deterministic route for canonical facts and provenance.

        Provenance is validated and carried by the caller for invalidation; it
        does not independently activate a module or alter legal screening.
        """

        if not isinstance(facts, AssessmentFacts):
            raise TypeError("facts must be an AssessmentFacts instance")
        provenance_records = tuple(fact_provenance)
        if any(
            not isinstance(item, FactProvenance)
            for item in provenance_records
        ):
            raise TypeError("fact_provenance must contain FactProvenance values")
        confirmed_input = self._stable_unique_strings(
            confirmed_modules,
            field_name="confirmed_modules",
        )
        hints = frozenset(
            self._stable_unique_strings(
                confirmed_routing_hints,
                field_name="confirmed_routing_hints",
            )
        )
        known_rule_ids = {definition.rule_id for definition in self._definitions}
        unknown_confirmed = [
            rule_id for rule_id in confirmed_input if rule_id not in known_rule_ids
        ]
        if unknown_confirmed:
            raise ValueError(
                "confirmed modules are not registered: "
                + ", ".join(unknown_confirmed)
            )

        suggested: list[str] = []
        confirmed: list[str] = []
        screened_out: list[str] = []
        reasons: dict[str, list[str]] = {}
        missing_by_rule: dict[str, list[str]] = {}
        matched_reasons_by_rule: dict[str, list[str]] = {}

        for definition in self._definitions:
            matched_reasons = self._matched_group_reasons(
                definition.eligibility_groups,
                facts,
                hints,
            )
            matched_reasons_by_rule[definition.rule_id] = matched_reasons
            if definition.rule_id in confirmed_input:
                confirmed.append(definition.rule_id)
                reasons[definition.rule_id] = [
                    "USER_CONFIRMED_MODULE",
                    *matched_reasons,
                ]
            elif matched_reasons:
                suggested.append(definition.rule_id)
                reasons[definition.rule_id] = matched_reasons
            else:
                screened_out.append(definition.rule_id)
                reasons[definition.rule_id] = ["NO_DETERMINISTIC_ROUTE_MATCH"]

            if definition.rule_id in (*suggested, *confirmed):
                requirement = self._validator.validate(
                    self._rule_registry.get(definition.rule_id),
                    facts,
                )
                missing_by_rule[definition.rule_id] = [
                    item.fact_path for item in requirement.missing_facts
                ]

        unsupported_routes = self._unsupported_routes(facts, hints)
        for unsupported in unsupported_routes:
            reasons[unsupported.path_id] = list(unsupported.routing_reasons)

        universal_questions = self._unanswered_universal_questions(facts)
        confirmation_ids = [
            self._definitions.get(rule_id).confirmation_question_id
            for rule_id in suggested
        ]
        follow_up_questions = self._confirmed_follow_up_questions(
            facts,
            confirmed,
        )
        next_questions = self._deduplicate_questions(
            [*universal_questions, *follow_up_questions]
        )
        universal_ids = [question.question_id for question in universal_questions]
        follow_up_ids = [
            question.question_id
            for question in follow_up_questions
            if question.question_id not in universal_ids
        ]

        return QuestionnaireRoute(
            suggested_modules=suggested,
            confirmed_modules=confirmed,
            unsupported_modules=unsupported_routes,
            screened_out_modules=screened_out,
            missing_fact_paths=missing_by_rule,
            next_questions=next_questions,
            module_confirmation_question_ids=confirmation_ids,
            ordered_step_ids=[
                *universal_ids,
                *confirmation_ids,
                *follow_up_ids,
            ],
            routing_reasons=reasons,
        )

    def _confirmed_follow_up_questions(
        self,
        facts: AssessmentFacts,
        confirmed_rule_ids: list[str],
    ) -> list[Question]:
        """Reuse QuestionnaireEngine once per module to preserve module order."""

        confirmed_set = frozenset(confirmed_rule_ids)
        questions: list[Question] = []
        for definition in self._definitions:
            if definition.rule_id not in confirmed_set:
                continue
            requirement = self._validator.validate(
                self._rule_registry.get(definition.rule_id),
                facts,
            )
            plan = self._questionnaire_engine.build([requirement])
            allowed_question_ids = frozenset(definition.question_ids)
            questions.extend(
                question
                for question in plan.questions
                if question.question_id in allowed_question_ids
                and self._dependencies_satisfied(question.question_id, facts)
            )
        return questions

    def _unanswered_universal_questions(
        self,
        facts: AssessmentFacts,
    ) -> list[Question]:
        questions: list[Question] = []
        for question_id in universal_question_ids():
            question = self._questions.get(question_id)
            if self._is_missing(self._resolve_fact(facts, question.fact_path)):
                if self._dependencies_satisfied(question_id, facts):
                    questions.append(question)
        return questions

    def _dependencies_satisfied(
        self,
        question_id: str,
        facts: AssessmentFacts,
    ) -> bool:
        metadata: RoutingQuestionDefinition = self._question_metadata[question_id]
        for dependency in metadata.dependencies:
            value = self._resolve_fact(facts, dependency.fact_path)
            primitive = self._primitive(value)
            if self._is_missing(value):
                if primitive not in dependency.accepted_values:
                    return False
            elif dependency.accepted_values and primitive not in dependency.accepted_values:
                return False
        return True

    def _unsupported_routes(
        self,
        facts: AssessmentFacts,
        hints: frozenset[str],
    ) -> list[UnsupportedPathRoute]:
        routes: list[UnsupportedPathRoute] = []
        for definition in self._unsupported:
            matched_reasons = self._matched_group_reasons(
                definition.eligibility_groups,
                facts,
                hints,
            )
            if not matched_reasons:
                continue
            routes.append(
                UnsupportedPathRoute(
                    path_id=definition.path_id,
                    framework=definition.framework,
                    display_module_key=definition.display_module_key,
                    message_keys=definition.message_keys,
                    routing_reasons=tuple(matched_reasons),
                )
            )
        return routes

    def _matched_group_reasons(
        self,
        groups: tuple[EligibilityHintGroup, ...],
        facts: AssessmentFacts,
        hints: frozenset[str],
    ) -> list[str]:
        return [
            group.reason_code
            for group in groups
            if self._group_matches(group, facts, hints)
        ]

    def _group_matches(
        self,
        group: EligibilityHintGroup,
        facts: AssessmentFacts,
        hints: frozenset[str],
    ) -> bool:
        if not all(
            self._condition_matches(condition, facts)
            for condition in group.all_conditions
        ):
            return False
        if group.any_conditions and not any(
            self._condition_matches(condition, facts)
            for condition in group.any_conditions
        ):
            return False
        if group.any_routing_hints and not hints.intersection(
            group.any_routing_hints
        ):
            return False
        return True

    def _condition_matches(
        self,
        condition: FactCondition,
        facts: AssessmentFacts,
    ) -> bool:
        actual = self._primitive(
            self._resolve_fact(facts, condition.fact_path)
        )
        if condition.operator is FactConditionOperator.EQUALS:
            return actual == condition.expected_values[0]
        if condition.operator is FactConditionOperator.IN:
            return actual in condition.expected_values
        return False

    @staticmethod
    def _resolve_fact(facts: AssessmentFacts, fact_path: str) -> object:
        value: object = facts
        for segment in fact_path.split("."):
            value = getattr(value, segment)
        return value

    @staticmethod
    def _primitive(value: object) -> object:
        return value.value if isinstance(value, Enum) else value

    @classmethod
    def _is_missing(cls, value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, Enum):
            return value.value == TriState.UNKNOWN.value
        if isinstance(value, str):
            return not value.strip()
        return False

    @staticmethod
    def _deduplicate_questions(questions: list[Question]) -> list[Question]:
        seen: set[str] = set()
        ordered: list[Question] = []
        for question in questions:
            if question.question_id in seen:
                continue
            seen.add(question.question_id)
            ordered.append(question)
        return ordered

    @staticmethod
    def _stable_unique_strings(
        values: Iterable[str],
        *,
        field_name: str,
    ) -> list[str]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{field_name} must be an iterable of strings")
        ordered: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must contain non-empty strings")
            normalized = value.strip()
            if normalized not in ordered:
                ordered.append(normalized)
        return ordered


def build_default_questionnaire_router() -> QuestionnaireRouter:
    """Build the Phase 1 router with the three existing legal rules."""

    rule_registry = RuleRegistry(
        [
            AIActHighRiskEmploymentRule(),
            GDPRArticle22RelevanceRule(),
            EUDataActRelevanceRule(),
        ]
    )
    question_registry = build_question_registry()
    definition_registry = build_rule_questionnaire_registry(
        rule_registry,
        question_registry,
    )
    return QuestionnaireRouter(
        rule_registry=rule_registry,
        definition_registry=definition_registry,
        question_registry=question_registry,
        unsupported_definitions=UNSUPPORTED_PATH_DEFINITIONS,
    )
