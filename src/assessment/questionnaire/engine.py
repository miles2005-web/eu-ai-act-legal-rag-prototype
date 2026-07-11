"""Map missing fact requirements to deterministic questionnaire plans."""

from __future__ import annotations

from collections.abc import Iterable

from src.assessment.questionnaire.models import Question, QuestionnairePlan
from src.assessment.questionnaire.registry import QuestionRegistry
from src.assessment.requirements import MissingFactReason, RuleRequirementResult


class QuestionnaireEngine:
    """Select unanswered questions without evaluating facts or legal rules."""

    def __init__(self, registry: QuestionRegistry) -> None:
        if not isinstance(registry, QuestionRegistry):
            raise TypeError("registry must be a QuestionRegistry")
        self._registry = registry

    def build(
        self,
        missing_fact_requirements: Iterable[RuleRequirementResult],
    ) -> QuestionnairePlan:
        """Build a question plan in registry order from missing requirements."""

        requirements = tuple(missing_fact_requirements)
        if any(
            not isinstance(requirement, RuleRequirementResult)
            for requirement in requirements
        ):
            raise TypeError(
                "missing_fact_requirements must contain RuleRequirementResult values"
            )

        missing_paths: list[str] = []
        invalid_paths: set[str] = set()
        for requirement in requirements:
            for missing_fact in requirement.missing_facts:
                if missing_fact.fact_path not in missing_paths:
                    missing_paths.append(missing_fact.fact_path)
                if missing_fact.reason is MissingFactReason.PATH_NOT_FOUND:
                    invalid_paths.add(missing_fact.fact_path)

        answerable_paths = [
            fact_path
            for fact_path in missing_paths
            if fact_path not in invalid_paths
        ]
        questions = list(self._registry.questions_for_fact_paths(answerable_paths))
        mapped_paths = {question.fact_path for question in questions}
        unmapped_paths = [
            fact_path for fact_path in missing_paths if fact_path not in mapped_paths
        ]
        return QuestionnairePlan(
            questions=questions,
            unmapped_fact_paths=unmapped_paths,
        )

    def unanswered_questions(
        self,
        missing_fact_requirements: Iterable[RuleRequirementResult],
    ) -> list[Question]:
        """Return only the ordered question list for convenience."""

        return self.build(missing_fact_requirements).questions

