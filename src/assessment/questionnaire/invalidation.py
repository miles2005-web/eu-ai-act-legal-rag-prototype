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

    declared_targets_by_source: dict[str, set[str]] = {}
    for question in questions:
        if not question.invalidations:
            continue
        targets = declared_targets_by_source.setdefault(question.fact_path, set())
        for declaration in question.invalidations:
            targets.update(declaration.fact_paths)

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
                record.fact_path in declared_targets_by_source.get(path, set())
                or not declared_targets_by_source.get(path)
            )
        ]
        if not triggering_paths:
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


def _append_once(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _stable_unique(values: Iterable[str]) -> list[str]:
    ordered: list[str] = []
    for value in values:
        if value not in ordered:
            ordered.append(value)
    return ordered
