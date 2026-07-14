"""UI-neutral helpers for the deterministic assessment questionnaire."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from src.assessment.facts import (
    AffectedPerson,
    AssessmentFacts,
    FactMetadata,
    FactSource,
    UseDomain,
)
from src.assessment.models import TriState
from src.assessment.questionnaire.definitions import (
    AI_ACT_EMPLOYMENT_RULE_ID,
    EU_DATA_ACT_RULE_ID,
    GDPR_ARTICLE22_RULE_ID,
    HINT_CANDIDATE_RANKING,
    HINT_CREDIT_DECISION,
    HINT_INDIVIDUAL_SIGNIFICANT_DECISION,
    HINT_INDUSTRIAL_CONNECTED_EQUIPMENT,
    HINT_PRODUCT_SAFETY_COMPONENT,
    HINT_RECRUITMENT,
    HINT_SELECTION,
    HINT_WORKER_MANAGEMENT,
    QUESTION_DEFINITIONS,
    RULE_QUESTIONNAIRE_DEFINITIONS,
    question_definitions_by_id,
)
from src.assessment.questionnaire.models import AnswerType, Question
from src.assessment.questionnaire.routing_models import FactProvenance
from src.assessment.questionnaire.routing_models import QuestionnaireRoute


UNIVERSAL_INTAKE_QUESTION_IDS = (
    "INTAKE-SYSTEM-NAME",
    "INTAKE-SYSTEM-PURPOSE",
    "INTAKE-USE-DOMAIN",
    "INTAKE-USE-TASK",
    "INTAKE-AFFECTED-PERSONS",
    "INTAKE-DECISION-IMPACT",
    "INTAKE-PERSONAL-DATA",
    "INTAKE-CONNECTED-PRODUCT",
    "INTAKE-RELATED-SERVICE",
)

ROUTING_HINT_IDS = (
    HINT_RECRUITMENT,
    HINT_SELECTION,
    HINT_CANDIDATE_RANKING,
    HINT_WORKER_MANAGEMENT,
    HINT_INDIVIDUAL_SIGNIFICANT_DECISION,
    HINT_CREDIT_DECISION,
    HINT_INDUSTRIAL_CONNECTED_EQUIPMENT,
    HINT_PRODUCT_SAFETY_COMPONENT,
)

IMPLEMENTED_MODULE_IDS = tuple(
    definition.rule_id for definition in RULE_QUESTIONNAIRE_DEFINITIONS
)

_QUESTION_DEFINITIONS = question_definitions_by_id()
_MODULE_DEFINITIONS = {
    definition.rule_id: definition
    for definition in RULE_QUESTIONNAIRE_DEFINITIONS
}
_QUESTION_MODULES: dict[str, tuple[str, ...]] = {}
for _question in QUESTION_DEFINITIONS:
    _QUESTION_MODULES[_question.question_id] = tuple(
        definition.rule_id
        for definition in RULE_QUESTIONNAIRE_DEFINITIONS
        if _question.question_id in definition.question_ids
    )

_NORMALIZATION_HINTS = {
    "employment.recruitment_screening.v1": HINT_RECRUITMENT,
    "employment.candidate_ranking.v1": HINT_CANDIDATE_RANKING,
    "employment.material_influence.v1": HINT_INDIVIDUAL_SIGNIFICANT_DECISION,
    "gdpr.significant_effect.v1": HINT_INDIVIDUAL_SIGNIFICANT_DECISION,
    "decision.credit.v1": HINT_CREDIT_DECISION,
    "data_act.connected_product.v1": HINT_INDUSTRIAL_CONNECTED_EQUIPMENT,
}


@dataclass(frozen=True, slots=True)
class QuestionnaireAnswer:
    """One canonical answer plus the localized text that produced it."""

    question_id: str
    value: object
    original_input: str | None = None


def universal_questions() -> tuple[Question, ...]:
    """Return the stable, regulation-neutral intake questions."""

    return tuple(
        _QUESTION_DEFINITIONS[question_id].as_question()
        for question_id in UNIVERSAL_INTAKE_QUESTION_IDS
    )


def question_definition(question_id: str):
    """Resolve authored metadata for a stable question ID."""

    try:
        return _QUESTION_DEFINITIONS[question_id]
    except KeyError as exc:
        raise KeyError(f"unknown questionnaire question {question_id!r}") from exc


def module_definition(module_id: str):
    """Resolve authored module metadata for a stable rule ID."""

    try:
        return _MODULE_DEFINITIONS[module_id]
    except KeyError as exc:
        raise KeyError(f"unknown questionnaire module {module_id!r}") from exc


def modules_for_question(question_id: str) -> tuple[str, ...]:
    return _QUESTION_MODULES.get(question_id, ())


def question_id_for_fact_path(fact_path: str) -> str:
    """Return the first authored question for a canonical leaf path."""

    for definition in QUESTION_DEFINITIONS:
        if definition.fact_path == fact_path:
            return definition.question_id
    raise KeyError(f"no questionnaire question maps to {fact_path!r}")


def localized_text_key(question_id: str, language: str, *, help_text: bool = False) -> str:
    """Return the catalogue key declared by the routing definition."""

    keys = question_definition(question_id).text_keys
    if language == "zh-CN":
        return keys.zh_cn_help_key if help_text else keys.zh_cn_label_key
    return keys.en_help_key if help_text else keys.en_label_key


def current_answer(facts: AssessmentFacts, question: Question) -> object:
    """Return a canonical widget value for an authored question."""

    value = resolve_fact(facts, question.fact_path)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [item.value if isinstance(item, Enum) else item for item in value]
    return value


def apply_question_answers(
    facts: AssessmentFacts,
    answers: Iterable[QuestionnaireAnswer],
    *,
    module_id: str | None = None,
    explicitly_confirmed: bool = True,
) -> list[FactProvenance]:
    """Write canonical questionnaire values and return invalidation provenance."""

    if not isinstance(facts, AssessmentFacts):
        raise TypeError("facts must be an AssessmentFacts instance")
    provenance: list[FactProvenance] = []
    for answer in answers:
        definition = question_definition(answer.question_id)
        canonical_value = _coerce_answer(definition.fact_path, answer.value)
        set_fact(facts, definition.fact_path, canonical_value)
        facts.fact_metadata[definition.fact_path] = FactMetadata(
            source=FactSource.QUESTIONNAIRE,
            question_id=definition.question_id,
        )
        owner = module_id
        if owner is None:
            candidate_modules = modules_for_question(definition.question_id)
            owner = candidate_modules[0] if len(candidate_modules) == 1 else None
        provenance.append(
            FactProvenance(
                fact_path=definition.fact_path,
                question_id=definition.question_id,
                module_id=owner,
                explicitly_confirmed=explicitly_confirmed,
                depends_on=tuple(
                    dependency.fact_path for dependency in definition.dependencies
                ),
            )
        )
    return provenance


def clear_fact_paths(facts: AssessmentFacts, fact_paths: Iterable[str]) -> None:
    """Clear only the requested fact leaves and their metadata."""

    for fact_path in fact_paths:
        current = resolve_fact(facts, fact_path)
        if isinstance(current, TriState):
            replacement: object = TriState.UNKNOWN
        elif isinstance(current, UseDomain):
            replacement = UseDomain.UNKNOWN
        elif isinstance(current, list):
            replacement = None
        else:
            replacement = None
        set_fact(facts, fact_path, replacement)
        facts.fact_metadata.pop(fact_path, None)


def merge_provenance(
    existing: Iterable[FactProvenance],
    updates: Iterable[FactProvenance],
) -> list[FactProvenance]:
    """Replace provenance by fact path while preserving deterministic order."""

    merged = {record.fact_path: record for record in existing}
    for record in updates:
        merged[record.fact_path] = record
    return list(merged.values())


def remove_provenance(
    provenance: Iterable[FactProvenance],
    fact_paths: Iterable[str],
) -> list[FactProvenance]:
    removed = frozenset(fact_paths)
    return [record for record in provenance if record.fact_path not in removed]


def hints_from_normalization(mapping_ids: Iterable[str]) -> tuple[str, ...]:
    """Map only audited normalization IDs to controlled routing hints."""

    hints: list[str] = []
    for mapping_id in mapping_ids:
        hint = _NORMALIZATION_HINTS.get(mapping_id)
        if hint and hint not in hints:
            hints.append(hint)
    return tuple(hints)


def execution_facts_for_modules(
    facts: AssessmentFacts,
    confirmed_modules: Iterable[str],
) -> AssessmentFacts:
    """Mask unconfirmed rule-only inputs in an immutable execution snapshot.

    This keeps module confirmation meaningful without changing a legal rule or
    the case's current canonical fact record. Shared facts required by a
    confirmed module remain available.
    """

    selected = frozenset(confirmed_modules)
    unknown = selected.difference(IMPLEMENTED_MODULE_IDS)
    if unknown:
        raise ValueError(f"unknown confirmed modules: {', '.join(sorted(unknown))}")
    execution_facts = deepcopy(facts)
    selected_paths = {
        path
        for module_id in selected
        for path in module_definition(module_id).required_fact_paths
    }
    paths_to_clear = {
        path
        for module_id in IMPLEMENTED_MODULE_IDS
        if module_id not in selected
        for path in module_definition(module_id).required_fact_paths
        if path not in selected_paths
    }
    clear_fact_paths(execution_facts, paths_to_clear)
    return execution_facts


def confirmed_missing_fact_paths(route: QuestionnaireRoute) -> tuple[str, ...]:
    """Return only missing facts owned by explicitly confirmed modules."""

    if not isinstance(route, QuestionnaireRoute):
        raise TypeError("route must be a QuestionnaireRoute")
    missing: list[str] = []
    for module_id in route.confirmed_modules:
        for fact_path in route.missing_fact_paths.get(module_id, []):
            if fact_path not in missing:
                missing.append(fact_path)
    return tuple(missing)


def required_facts_complete(route: QuestionnaireRoute) -> bool:
    """Derive completion solely from confirmed implemented modules."""

    if not isinstance(route, QuestionnaireRoute):
        raise TypeError("route must be a QuestionnaireRoute")
    return bool(route.confirmed_modules) and not confirmed_missing_fact_paths(route)


def resolve_fact(facts: AssessmentFacts, fact_path: str) -> object:
    value: object = facts
    for segment in fact_path.split("."):
        value = getattr(value, segment)
    return value


def set_fact(facts: AssessmentFacts, fact_path: str, value: object) -> None:
    target: object = facts
    segments = fact_path.split(".")
    for segment in segments[:-1]:
        target = getattr(target, segment)
    setattr(target, segments[-1], value)


def _coerce_answer(fact_path: str, value: object) -> object:
    if fact_path == "use_context.domain":
        return value if isinstance(value, UseDomain) else UseDomain(str(value))
    if fact_path == "use_context.affected_persons":
        if value is None:
            return None
        return [
            item if isinstance(item, AffectedPerson) else AffectedPerson(str(item))
            for item in value
        ]
    current = resolve_fact(AssessmentFacts(), fact_path)
    if isinstance(current, TriState):
        return value if isinstance(value, TriState) else TriState(str(value))
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


__all__ = [
    "AI_ACT_EMPLOYMENT_RULE_ID",
    "EU_DATA_ACT_RULE_ID",
    "GDPR_ARTICLE22_RULE_ID",
    "IMPLEMENTED_MODULE_IDS",
    "QuestionnaireAnswer",
    "ROUTING_HINT_IDS",
    "UNIVERSAL_INTAKE_QUESTION_IDS",
    "apply_question_answers",
    "clear_fact_paths",
    "confirmed_missing_fact_paths",
    "current_answer",
    "execution_facts_for_modules",
    "hints_from_normalization",
    "localized_text_key",
    "merge_provenance",
    "module_definition",
    "modules_for_question",
    "question_definition",
    "question_id_for_fact_path",
    "remove_provenance",
    "required_facts_complete",
    "resolve_fact",
    "set_fact",
    "universal_questions",
]
