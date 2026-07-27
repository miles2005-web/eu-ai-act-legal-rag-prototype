"""Stable case-local identities and canonical assessment scope."""

from __future__ import annotations

from dataclasses import dataclass

from src.assessment.models import SerializableModel, validate_stable_identifier


class StableIdentifier(str):
    """Validated opaque identity that is independent of display text or order."""

    def __new__(cls, value: str):
        return str.__new__(
            cls,
            validate_stable_identifier(value, field_name=cls.__name__),
        )


class ActorId(StableIdentifier):
    pass


class SystemId(StableIdentifier):
    pass


class WorkflowId(StableIdentifier):
    pass


class ProcessingOperationId(StableIdentifier):
    pass


class InvocationId(StableIdentifier):
    pass


@dataclass(frozen=True, slots=True, order=True)
class AssessmentScope(SerializableModel):
    """Canonical subject and operation tuple for v0.6 contracts."""

    actor_id: ActorId | None = None
    system_id: SystemId | None = None
    workflow_id: WorkflowId | None = None
    processing_operation_id: ProcessingOperationId | None = None

    def __post_init__(self) -> None:
        for field_name, identifier_type in (
            ("actor_id", ActorId),
            ("system_id", SystemId),
            ("workflow_id", WorkflowId),
            ("processing_operation_id", ProcessingOperationId),
        ):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, identifier_type):
                object.__setattr__(self, field_name, identifier_type(value))
    def canonical_tuple(self) -> tuple[str | None, ...]:
        return (
            self.actor_id,
            self.system_id,
            self.workflow_id,
            self.processing_operation_id,
        )
