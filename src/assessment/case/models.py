"""Domain model for a mutable assessment case."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime

from src.assessment.facts import AssessmentFacts
from src.assessment.models import SerializableModel, new_identifier, utc_now


@dataclass(slots=True)
class AssessmentCase(SerializableModel):
    """Current case facts, separate from immutable runs and reports."""

    name: str
    current_facts: AssessmentFacts
    description: str | None = None
    case_id: str = field(default_factory=new_identifier)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    schema_version: str = "2.0.0"

    def __post_init__(self) -> None:
        for field_name in ("name", "case_id", "schema_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string or None")
        if not isinstance(self.current_facts, AssessmentFacts):
            raise TypeError("current_facts must be an AssessmentFacts instance")
        if self.schema_version != self.current_facts.schema_version:
            raise ValueError(
                "schema_version must match current_facts.schema_version"
            )
        for field_name in ("created_at", "updated_at"):
            value = getattr(self, field_name)
            if not isinstance(value, datetime) or value.utcoffset() is None:
                raise TypeError(f"{field_name} must be a timezone-aware datetime")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")

        self.current_facts = deepcopy(self.current_facts)

