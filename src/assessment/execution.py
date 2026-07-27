"""Operational execution records kept separate from legal Findings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import ClassVar

from src.assessment.invocations import RuleInvocation
from src.assessment.models import SerializableModel, validate_stable_identifier


class RuleExecutionStatus(str, Enum):
    COMPLETED = "completed"
    NOT_AUTHORIZED = "not_authorized"
    BLOCKED_BY_DEPENDENCY = "blocked_by_dependency"
    BLOCKED_BY_EVIDENCE = "blocked_by_evidence"
    MISSING_FACTS = "missing_facts"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RuleExecutionRecord(SerializableModel):
    invocation: RuleInvocation
    status: RuleExecutionStatus
    missing_fact_paths: tuple[str, ...] = ()
    dependency_reason: str | None = None
    evidence_block_reason: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    finding_id: str | None = None
    ruleset_baseline_id: str | None = None
    evidence_baseline_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    schema_version: str = "1.0.0"

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"invocation", "status"}
    )

    def __post_init__(self) -> None:
        if isinstance(self.invocation, dict):
            object.__setattr__(
                self,
                "invocation",
                RuleInvocation.from_dict(self.invocation),
            )
        if not isinstance(self.invocation, RuleInvocation):
            raise TypeError("invocation must be a RuleInvocation")
        if isinstance(self.status, str):
            try:
                object.__setattr__(
                    self,
                    "status",
                    RuleExecutionStatus(self.status),
                )
            except ValueError as exc:
                raise ValueError(
                    f"invalid RuleExecutionStatus: {self.status!r}"
                ) from exc
        if not isinstance(self.status, RuleExecutionStatus):
            raise TypeError("status must be a RuleExecutionStatus")
        if self.schema_version != "1.0.0":
            raise ValueError("unsupported RuleExecutionRecord schema_version")
        if self.status is RuleExecutionStatus.COMPLETED:
            if self.finding_id is not None:
                validate_stable_identifier(
                    self.finding_id, field_name="finding_id"
                )
        elif self.finding_id is not None:
            raise ValueError(
                "only completed execution records may reference a Finding"
            )
        if not isinstance(self.missing_fact_paths, (list, tuple)):
            raise TypeError("missing_fact_paths must be a list or tuple")
        missing_paths: list[str] = []
        for path in self.missing_fact_paths:
            if not isinstance(path, str) or not path.strip():
                raise ValueError(
                    "missing_fact_paths must contain non-empty strings"
                )
            missing_paths.append(path)
        if len(set(missing_paths)) != len(missing_paths):
            raise ValueError("missing_fact_paths must not contain duplicates")
        if self.status is RuleExecutionStatus.MISSING_FACTS:
            if not missing_paths:
                raise ValueError("missing_facts requires missing_fact_paths")
        object.__setattr__(
            self, "missing_fact_paths", tuple(sorted(missing_paths))
        )
        for field_name in ("ruleset_baseline_id", "evidence_baseline_id"):
            value = getattr(self, field_name)
            if value is not None:
                validate_stable_identifier(value, field_name=field_name)
        for field_name in (
            "dependency_reason",
            "evidence_block_reason",
            "failure_type",
            "failure_message",
        ):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{field_name} must be None or a non-empty string"
                    )
                object.__setattr__(self, field_name, value.strip())
        populated_fields = {
            "missing_fact_paths": bool(self.missing_fact_paths),
            "dependency_reason": self.dependency_reason is not None,
            "evidence_block_reason": self.evidence_block_reason is not None,
            "failure_type": self.failure_type is not None,
            "failure_message": self.failure_message is not None,
        }
        allowed_fields = {
            RuleExecutionStatus.COMPLETED: frozenset(),
            RuleExecutionStatus.NOT_AUTHORIZED: frozenset(),
            RuleExecutionStatus.BLOCKED_BY_DEPENDENCY: frozenset(
                {"dependency_reason"}
            ),
            RuleExecutionStatus.BLOCKED_BY_EVIDENCE: frozenset(
                {"evidence_block_reason"}
            ),
            RuleExecutionStatus.MISSING_FACTS: frozenset(
                {"missing_fact_paths"}
            ),
            RuleExecutionStatus.FAILED: frozenset(
                {"failure_type", "failure_message"}
            ),
        }[self.status]
        contradictory = sorted(
            field_name
            for field_name, populated in populated_fields.items()
            if populated and field_name not in allowed_fields
        )
        if contradictory:
            raise ValueError(
                f"{self.status.value} execution records cannot contain: "
                + ", ".join(contradictory)
            )
        if (
            self.status is RuleExecutionStatus.BLOCKED_BY_DEPENDENCY
            and self.dependency_reason is None
        ):
            raise ValueError(
                "blocked_by_dependency requires dependency_reason"
            )
        if (
            self.status is RuleExecutionStatus.BLOCKED_BY_EVIDENCE
            and self.evidence_block_reason is None
        ):
            raise ValueError(
                "blocked_by_evidence requires evidence_block_reason"
            )
        if (
            self.status is RuleExecutionStatus.FAILED
            and self.failure_type is None
            and self.failure_message is None
        ):
            raise ValueError(
                "failed requires failure_type or failure_message"
            )
        for field_name in ("started_at", "completed_at"):
            value = getattr(self, field_name)
            if value is not None:
                if not isinstance(value, datetime):
                    raise TypeError(f"{field_name} must be a datetime")
                if value.utcoffset() is None:
                    raise ValueError(f"{field_name} must be timezone-aware")
        if (
            self.started_at is not None
            and self.completed_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError("completed_at cannot precede started_at")
