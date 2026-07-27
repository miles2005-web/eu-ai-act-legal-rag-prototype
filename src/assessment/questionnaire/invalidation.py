"""Pure dependency invalidation for questionnaire answers and module state."""

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

from src.assessment.facts import AssessmentFacts
from src.assessment.questionnaire.definitions import (
    QUESTION_DEFINITIONS,
    RULE_QUESTIONNAIRE_DEFINITIONS,
)
from src.assessment.questionnaire.routing_models import (
    FactProvenance,
    InvalidationResult,
    QuestionInvalidation,
    RoutingQuestionDefinition,
    RuleQuestionnaireDefinition,
)


def calculate_invalidations(
    previous_facts: AssessmentFacts,
    current_facts: AssessmentFacts,
    provenance: Iterable[FactProvenance],
    *,
    question_definitions: Iterable[RoutingQuestionDefinition] = (
        QUESTION_DEFINITIONS
    ),
    rule_definitions: Iterable[RuleQuestionnaireDefinition] = (
        RULE_QUESTIONNAIRE_DEFINITIONS
    ),
) -> InvalidationResult:
    """Return stale state without mutating either fact snapshot."""

    if not isinstance(previous_facts, AssessmentFacts):
        raise TypeError("previous_facts must be an AssessmentFacts instance")
    if not isinstance(current_facts, AssessmentFacts):
        raise TypeError("current_facts must be an AssessmentFacts instance")
    provenance_records = tuple(provenance)
    if any(not isinstance(item, FactProvenance) for item in provenance_records):
        raise TypeError("provenance must contain FactProvenance values")
    questions = tuple(question_definitions)
    rules = tuple(rule_definitions)
    if any(not isinstance(item, RoutingQuestionDefinition) for item in questions):
        raise TypeError(
            "question_definitions must contain RoutingQuestionDefinition values"
        )
    if any(not isinstance(item, RuleQuestionnaireDefinition) for item in rules):
        raise TypeError(
            "rule_definitions must contain RuleQuestionnaireDefinition values"
        )

    potential_upstream_paths = _stable_unique(
        [
            *(
                dependency_path
                for item in provenance_records
                for dependency_path in item.depends_on
            ),
            *(
                question.fact_path
                for question in questions
                if question.invalidations
            ),
            "use_context.domain",
        ]
    )
    changed_paths = [
        path
        for path in potential_upstream_paths
        if _primitive(_resolve(previous_facts, path))
        != _primitive(_resolve(current_facts, path))
    ]

    declared_invalidations_by_source: dict[
        str,
        list[QuestionInvalidation],
    ] = {}
    for question in questions:
        if not question.invalidations:
            continue
        declared_invalidations_by_source.setdefault(
            question.fact_path,
            [],
        ).extend(question.invalidations)

    questions_by_id = {question.question_id: question for question in questions}

    stale_fact_paths: list[str] = []
    invalidated_question_ids: list[str] = []
    invalidated_module_ids: list[str] = []
    reasons: dict[str, list[str]] = {}

    for record in provenance_records:
        triggering_paths = [
            path
            for path in changed_paths
            if path in record.depends_on
            and (
                _is_declared_target(
                    path,
                    record.fact_path,
                    declared_invalidations_by_source,
                )
                or not declared_invalidations_by_source.get(path)
            )
        ]
        if not triggering_paths:
            continue
        question = questions_by_id.get(record.question_id)
        if (
            question is not None
            and question.any_dependencies
            and set(triggering_paths).issubset(
                {
                    dependency.fact_path
                    for dependency in question.any_dependencies
                }
            )
            and _any_dependency_satisfied(question, current_facts)
        ):
            continue
        if record.explicitly_confirmed and all(
            _preserves_explicit_confirmation(
                path,
                record.fact_path,
                declared_invalidations_by_source,
            )
            for path in triggering_paths
        ):
            continue
        _append_once(stale_fact_paths, record.fact_path)
        _append_once(invalidated_question_ids, record.question_id)
        if record.module_id:
            _append_once(invalidated_module_ids, record.module_id)
        reasons[record.fact_path] = [
            f"UPSTREAM_FACT_CHANGED::{path}" for path in triggering_paths
        ]

    previous_domain = previous_facts.use_context.domain
    current_domain = current_facts.use_context.domain
    if previous_domain is not current_domain:
        for definition in rules:
            if (
                previous_domain in definition.supported_domains
                and current_domain not in definition.supported_domains
            ):
                _append_once(invalidated_module_ids, definition.rule_id)
                reasons.setdefault(definition.rule_id, []).append(
                    "SUPPORTED_DOMAIN_CHANGED"
                )

    return InvalidationResult(
        changed_upstream_fact_paths=changed_paths,
        stale_fact_paths=stale_fact_paths,
        invalidated_question_ids=invalidated_question_ids,
        invalidated_module_ids=invalidated_module_ids,
        removed_provenance_fact_paths=list(stale_fact_paths),
        reasons=reasons,
    )


def _resolve(facts: AssessmentFacts, fact_path: str) -> object:
    value: object = facts
    for segment in fact_path.split("."):
        value = getattr(value, segment)
    return value


def _primitive(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _is_declared_target(
    source_path: str,
    target_path: str,
    declarations_by_source: dict[str, list[QuestionInvalidation]],
) -> bool:
    return any(
        target_path in declaration.fact_paths
        for declaration in declarations_by_source.get(source_path, ())
    )


def _preserves_explicit_confirmation(
    source_path: str,
    target_path: str,
    declarations_by_source: dict[str, list[QuestionInvalidation]],
) -> bool:
    matching = [
        declaration
        for declaration in declarations_by_source.get(source_path, ())
        if target_path in declaration.fact_paths
    ]
    return bool(matching) and all(
        declaration.preserve_explicitly_confirmed
        for declaration in matching
    )


def _any_dependency_satisfied(
    question: RoutingQuestionDefinition,
    facts: AssessmentFacts,
) -> bool:
    for dependency in question.any_dependencies:
        value = _primitive(_resolve(facts, dependency.fact_path))
        if dependency.accepted_values:
            if value in dependency.accepted_values:
                return True
        elif value is not None and value != "unknown":
            return True
    return False


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _stable_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered
