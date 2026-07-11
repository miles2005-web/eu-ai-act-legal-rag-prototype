"""Serializable models for fact-oriented assessment questions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.assessment.models import SerializableModel


class AnswerType(str, Enum):
    """Input shapes supported by the questionnaire domain layer."""

    TRI_STATE = "tri_state"
    TEXT = "text"
    SINGLE_CHOICE = "single_choice"
    MULTIPLE_CHOICE = "multiple_choice"
    DATE = "date"


@dataclass(frozen=True, slots=True)
class QuestionOption(SerializableModel):
    """Stable machine value and user-facing label for a choice."""

    value: str
    label: str

    def __post_init__(self) -> None:
        for field_name in ("value", "label"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Question(SerializableModel):
    """A version-neutral question mapped to one AssessmentFacts leaf path."""

    question_id: str
    text: str
    fact_path: str
    answer_type: AnswerType
    options: tuple[QuestionOption, ...] = field(default_factory=tuple)
    required: bool = True
    legal_relevance: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for field_name in ("question_id", "text", "fact_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.answer_type, AnswerType):
            raise TypeError("answer_type must be an AnswerType")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")
        if not isinstance(self.options, tuple):
            raise TypeError("options must be a tuple")
        if any(not isinstance(option, QuestionOption) for option in self.options):
            raise TypeError("options must contain QuestionOption values")
        option_values = [option.value for option in self.options]
        if len(set(option_values)) != len(option_values):
            raise ValueError("question options must have unique values")
        if self.answer_type in {
            AnswerType.SINGLE_CHOICE,
            AnswerType.MULTIPLE_CHOICE,
        } and not self.options:
            raise ValueError("choice questions must define at least one option")
        if not isinstance(self.legal_relevance, tuple):
            raise TypeError("legal_relevance must be a tuple")
        if any(
            not isinstance(reference, str) or not reference.strip()
            for reference in self.legal_relevance
        ):
            raise ValueError("legal_relevance must contain non-empty strings")


@dataclass(slots=True)
class QuestionnairePlan(SerializableModel):
    """Ordered unanswered questions plus fact paths lacking a mapping."""

    questions: list[Question] = field(default_factory=list)
    unmapped_fact_paths: list[str] = field(default_factory=list)
