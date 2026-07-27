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
from src.assessment.product_regulation import load_annex_i_instrument_catalog
from src.assessment.questionnaire.definitions import (
    AI_ACT_EMPLOYMENT_RULE_ID,
    AI_ACT_PRODUCT_SAFETY_RULE_ID,
    ANNEX_I_UNKNOWN_OPTION,
    EU_DATA_ACT_RULE_ID,
    GDPR_ARTICLE22_RULE_ID,
    HINT_CANDIDATE_RANKING,
    HINT_CREDIT_DECISION,
    HINT_INDIVIDUAL_SIGNIFICANT_DECISION,
    HINT_INDUSTRIAL_CONNECTED_EQUIPMENT,
    HINT_CONFORMITY_ASSESSMENT,
    HINT_MEDICAL_DEVICE_CONTEXT,
    HINT_PRODUCT_SAFETY_COMPONENT,
    HINT_PRODUCT_SAFETY_CONTEXT,
    HINT_REGULATED_AI_PRODUCT,
    HINT_REGULATED_EQUIPMENT_CONTEXT,
    HINT_RECRUITMENT,
    HINT_SELECTION,
    HINT_WORKER_MANAGEMENT,
    QUESTION_DEFINITIONS,
    RULE_QUESTIONNAIRE_DEFINITIONS,
    question_definitions_by_id,
)
from src.assessment.questionnaire.models import AnswerType, Question
from src.assessment.questionnaire.routing_models import (
    FactProvenance,
    QuestionnaireRoute,
    QuestionResponseState,
)


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
    HINT_REGULATED_AI_PRODUCT,
    HINT_PRODUCT_SAFETY_CONTEXT,
    HINT_MEDICAL_DEVICE_CONTEXT,
    HINT_REGULATED_EQUIPMENT_CONTEXT,
    HINT_CONFORMITY_ASSESSMENT,
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
    "ai_act.product_safety_component.v1": HINT_PRODUCT_SAFETY_COMPONENT,
    "ai_act.regulated_ai_product.v1": HINT_REGULATED_AI_PRODUCT,
    "ai_act.product_safety_context.v1": HINT_PRODUCT_SAFETY_CONTEXT,
    "ai_act.medical_device_context.v1": HINT_MEDICAL_DEVICE_CONTEXT,
    "ai_act.regulated_equipment_context.v1": HINT_REGULATED_EQUIPMENT_CONTEXT,
    "ai_act.conformity_assessment.v1": HINT_CONFORMITY_ASSESSMENT,
}

_ANNEX_I_CATALOG = load_annex_i_instrument_catalog()
UNANSWERED_WIDGET_VALUE = "__unanswered__"


@dataclass(frozen=True, slots=True)
class QuestionnaireAnswer:
    """One canonical answer plus the localized text that produced it."""

    question_id: str
    value: object
    original_input: str | None = None


@dataclass(frozen=True, slots=True)
class QuestionGap:
    """One unresolved or dependency-blocked questionnaire fact."""

    question_id: str
    fact_path: str
    recorded_unknown: bool = False


@dataclass(frozen=True, slots=True)
class ConfirmedModuleGap:
    """Presentation-safe gaps for one explicitly confirmed module."""

    module_id: str
    unresolved: tuple[QuestionGap, ...]
    blocked: tuple[QuestionGap, ...]

    @property
    def unresolved_count(self) -> int:
        return len(self.unresolved)


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


def question_option_label(
    question_id: str,
    option_value: str,
    language: str,
) -> str | None:
    """Return a human label for options with domain-backed identities."""

    definition = question_definition(question_id)
    if definition.fact_path != "product_regulation.annex_i_instrument":
        return None
    if option_value == ANNEX_I_UNKNOWN_OPTION:
        return None
    instrument = _ANNEX_I_CATALOG.get(option_value)
    return (
        f"{instrument.display_label(language)} — "
        f"{instrument.canonical_reference}"
    )


def apply_question_answers(
    facts: AssessmentFacts,
    answers: Iterable[QuestionnaireAnswer],
    *,
    module_id: str | None = None,
    explicitly_confirmed: bool = True,
    record_explicit_unknown: bool = False,
) -> list[FactProvenance]:
    """Write canonical questionnaire values and return invalidation provenance."""

    if not isinstance(facts, AssessmentFacts):
        raise TypeError("facts must be an AssessmentFacts instance")
    if not isinstance(record_explicit_unknown, bool):
        raise TypeError("record_explicit_unknown must be a bool")
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
                    dependency.fact_path
                    for dependency in (
                        *definition.dependencies,
                        *definition.any_dependencies,
                    )
                ),
                response_state=(
                    QuestionResponseState.EXPLICIT_UNKNOWN
                    if record_explicit_unknown
                    and _is_missing_canonical_value(canonical_value)
                    else QuestionResponseState.ANSWERED
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


def authorized_rule_ids_for_modules(
    confirmed_modules: Iterable[str],
) -> tuple[str, ...]:
    """Map confirmed implemented modules to deterministic engine rule IDs."""

    if isinstance(confirmed_modules, (str, bytes)):
        raise TypeError("confirmed_modules must be an iterable of module IDs")
    requested = tuple(confirmed_modules)
    if any(not isinstance(module_id, str) or not module_id for module_id in requested):
        raise ValueError("confirmed_modules must contain non-empty strings")
    if len(set(requested)) != len(requested):
        raise ValueError("confirmed_modules must not contain duplicates")
    unknown = set(requested).difference(IMPLEMENTED_MODULE_IDS)
    if unknown:
        raise ValueError(
            "unknown confirmed modules: " + ", ".join(sorted(unknown))
        )
    selected = frozenset(requested)
    return tuple(
        _MODULE_DEFINITIONS[module_id].rule_id
        for module_id in IMPLEMENTED_MODULE_IDS
        if module_id in selected
    )


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


def confirmed_module_gaps(
    route: QuestionnaireRoute,
    facts: AssessmentFacts,
) -> tuple[ConfirmedModuleGap, ...]:
    """Separate actionable missing facts from dependency-blocked questions.

    The route remains the source of truth for facts currently required by a
    confirmed rule. Authored dependency metadata is then used only to explain
    why later questions are not yet available; blocked questions never inflate
    the unresolved-fact count.
    """

    if not isinstance(route, QuestionnaireRoute):
        raise TypeError("route must be a QuestionnaireRoute")
    if not isinstance(facts, AssessmentFacts):
        raise TypeError("facts must be an AssessmentFacts instance")

    available_ids = {
        question.question_id for question in route.next_questions
    }
    recorded_unknown_ids = frozenset(route.recorded_unknown_question_ids)
    summaries: list[ConfirmedModuleGap] = []

    for module_id in route.confirmed_modules:
        definition = module_definition(module_id)
        missing_paths = route.missing_fact_paths.get(module_id, [])
        unresolved: list[QuestionGap] = []
        blocked: list[QuestionGap] = []
        classified_ids: set[str] = set()

        for fact_path in missing_paths:
            question_id = question_id_for_fact_path(fact_path)
            gap = QuestionGap(
                question_id=question_id,
                fact_path=fact_path,
                recorded_unknown=question_id in recorded_unknown_ids,
            )
            if question_id in available_ids or question_id in recorded_unknown_ids:
                unresolved.append(gap)
            else:
                blocked.append(gap)
            classified_ids.add(question_id)

        dependency_paths = {
            gap.fact_path for gap in (*unresolved, *blocked)
        }
        changed = True
        while changed:
            changed = False
            for question_id in definition.question_ids:
                if question_id in classified_ids:
                    continue
                question_metadata = question_definition(question_id)
                if not _is_missing_canonical_value(
                    resolve_fact(facts, question_metadata.fact_path)
                ):
                    continue
                dependencies = (
                    *question_metadata.dependencies,
                    *question_metadata.any_dependencies,
                )
                if not any(
                    dependency.fact_path in dependency_paths
                    for dependency in dependencies
                ):
                    continue
                gap = QuestionGap(
                    question_id=question_id,
                    fact_path=question_metadata.fact_path,
                    recorded_unknown=question_id in recorded_unknown_ids,
                )
                blocked.append(gap)
                classified_ids.add(question_id)
                dependency_paths.add(question_metadata.fact_path)
                changed = True

        summaries.append(
            ConfirmedModuleGap(
                module_id=module_id,
                unresolved=tuple(unresolved),
                blocked=tuple(blocked),
            )
        )
    return tuple(summaries)


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
    if fact_path == "product_regulation.annex_i_instrument":
        if value is None or value == ANNEX_I_UNKNOWN_OPTION:
            return None
        if not isinstance(value, str):
            raise TypeError("Annex I instrument selection must be a stable ID")
        return _ANNEX_I_CATALOG.get(value).instrument_id
    current = resolve_fact(AssessmentFacts(), fact_path)
    if isinstance(current, TriState):
        return value if isinstance(value, TriState) else TriState(str(value))
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return value


def _is_missing_canonical_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, Enum):
        return value.value == TriState.UNKNOWN.value
    if isinstance(value, str):
        return not value.strip()
    return False


__all__ = [
    "AI_ACT_EMPLOYMENT_RULE_ID",
    "AI_ACT_PRODUCT_SAFETY_RULE_ID",
    "ConfirmedModuleGap",
    "EU_DATA_ACT_RULE_ID",
    "GDPR_ARTICLE22_RULE_ID",
    "IMPLEMENTED_MODULE_IDS",
    "UNANSWERED_WIDGET_VALUE",
    "QuestionnaireAnswer",
    "QuestionGap",
    "ROUTING_HINT_IDS",
    "UNIVERSAL_INTAKE_QUESTION_IDS",
    "apply_question_answers",
    "authorized_rule_ids_for_modules",
    "clear_fact_paths",
    "confirmed_missing_fact_paths",
    "confirmed_module_gaps",
    "current_answer",
    "execution_facts_for_modules",
    "hints_from_normalization",
    "localized_text_key",
    "merge_provenance",
    "module_definition",
    "modules_for_question",
    "question_definition",
    "question_id_for_fact_path",
    "question_option_label",
    "remove_provenance",
    "required_facts_complete",
    "resolve_fact",
    "set_fact",
    "universal_questions",
]
