"""Deterministic registration and lookup for assessment questions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import is_dataclass

from src.assessment.facts import AssessmentFacts
from src.assessment.questionnaire.models import Question


class DuplicateQuestionError(ValueError):
    """Raised when a question ID has already been registered."""


class DuplicateQuestionFactPathError(ValueError):
    """Raised when two questions target the same fact path."""


class InvalidQuestionFactPathError(ValueError):
    """Raised when a question does not target an AssessmentFacts leaf field."""


class QuestionNotFoundError(KeyError):
    """Raised when a question ID is absent from the registry."""


class QuestionRegistry:
    """Question registry preserving authored insertion order."""

    def __init__(self, questions: Iterable[Question] | None = None) -> None:
        self._questions_by_id: dict[str, Question] = {}
        self._question_ids_by_fact_path: dict[str, str] = {}
        if questions is not None:
            self.register_many(questions)

    def register(self, question: Question) -> Question:
        """Validate and register one question."""

        if not isinstance(question, Question):
            raise TypeError("question must be a Question instance")
        if question.question_id in self._questions_by_id:
            raise DuplicateQuestionError(
                f"Question ID {question.question_id!r} is already registered"
            )
        if question.fact_path in self._question_ids_by_fact_path:
            raise DuplicateQuestionFactPathError(
                f"Fact path {question.fact_path!r} already has a question"
            )
        self._validate_fact_path(question.fact_path)

        self._questions_by_id[question.question_id] = question
        self._question_ids_by_fact_path[question.fact_path] = question.question_id
        return question

    def register_many(self, questions: Iterable[Question]) -> None:
        """Register questions in iterable order."""

        for question in questions:
            self.register(question)

    def get(self, question_id: str) -> Question:
        """Return a question by stable ID."""

        try:
            return self._questions_by_id[question_id]
        except KeyError as exc:
            raise QuestionNotFoundError(
                f"Question ID {question_id!r} is not registered"
            ) from exc

    def get_by_fact_path(self, fact_path: str) -> Question | None:
        """Return the question mapped to a fact path, if one exists."""

        question_id = self._question_ids_by_fact_path.get(fact_path)
        return self._questions_by_id.get(question_id) if question_id else None

    def all(self) -> tuple[Question, ...]:
        """Return all questions in authored order."""

        return tuple(self._questions_by_id.values())

    def questions_for_fact_paths(
        self,
        fact_paths: Iterable[str],
    ) -> tuple[Question, ...]:
        """Select mapped questions while preserving registry order."""

        requested_paths = frozenset(fact_paths)
        return tuple(
            question
            for question in self._questions_by_id.values()
            if question.fact_path in requested_paths
        )

    def __contains__(self, question_id: object) -> bool:
        return question_id in self._questions_by_id

    def __iter__(self) -> Iterator[Question]:
        return iter(self._questions_by_id.values())

    def __len__(self) -> int:
        return len(self._questions_by_id)

    @staticmethod
    def _validate_fact_path(fact_path: str) -> None:
        current: object = AssessmentFacts()
        for segment in fact_path.split("."):
            if not segment or not hasattr(current, segment):
                raise InvalidQuestionFactPathError(
                    f"Fact path {fact_path!r} does not exist in AssessmentFacts"
                )
            current = getattr(current, segment)
        if is_dataclass(current):
            raise InvalidQuestionFactPathError(
                f"Fact path {fact_path!r} must target a leaf field"
            )

