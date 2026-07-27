"""Shared models and utilities for structured assessment runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from src.assessment.facts import AssessmentFacts
    from src.assessment.findings import Finding


class TriState(str, Enum):
    """A legal fact that may be confirmed, rejected, or not yet known."""

    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class AssessmentRunStatus(str, Enum):
    """Lifecycle state of an in-memory assessment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def new_identifier() -> str:
    """Return a stable string representation for a new domain identifier."""

    return str(uuid4())


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def to_primitive(value: Any) -> Any:
    """Convert nested domain values into JSON-compatible primitives."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            model_field.name: to_primitive(getattr(value, model_field.name))
            for model_field in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value


class SerializableModel:
    """Mixin providing a persistence-neutral dictionary representation."""

    def to_dict(self) -> dict[str, Any]:
        primitive = to_primitive(self)
        if not isinstance(primitive, dict):
            raise TypeError("Domain model did not serialize to a dictionary")
        return primitive


@dataclass(slots=True)
class AssessmentRun(SerializableModel):
    """A versioned, reproducible execution over an immutable fact snapshot."""

    case_id: str
    facts_snapshot: AssessmentFacts
    id: str = field(default_factory=new_identifier)
    ruleset_version: str = "2.0.0"
    questionnaire_version: str = "2.0.0"
    corpus_version: str | None = None
    authorized_rule_ids: list[str] = field(default_factory=list)
    input_fingerprint: str | None = None
    status: AssessmentRunStatus = AssessmentRunStatus.PENDING
    findings: list[Finding] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        # A run must retain the facts it evaluated even if the draft case changes.
        self.facts_snapshot = deepcopy(self.facts_snapshot)
        self.authorized_rule_ids = list(self.authorized_rule_ids)
        if any(
            not isinstance(rule_id, str) or not rule_id.strip()
            for rule_id in self.authorized_rule_ids
        ):
            raise ValueError("authorized_rule_ids must contain non-empty strings")
        if len(set(self.authorized_rule_ids)) != len(self.authorized_rule_ids):
            raise ValueError("authorized_rule_ids must not contain duplicates")
        if self.input_fingerprint is not None and (
            not isinstance(self.input_fingerprint, str)
            or not self.input_fingerprint.strip()
        ):
            raise ValueError("input_fingerprint must be non-empty when provided")
