"""Domain models for binding legal evidence to assessment findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

from src.assessment.models import (
    SerializableModel,
    new_identifier,
    validate_stable_identifier,
)


class AuthorityLevel(str, Enum):
    """General authority classification for a legal evidence source."""

    BINDING_LEGISLATION = "binding_legislation"
    CASE_LAW = "case_law"
    OFFICIAL_GUIDANCE = "official_guidance"
    NON_BINDING_OFFICIAL_MATERIAL = "non_binding_official_material"
    SECONDARY_SOURCE = "secondary_source"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Evidence(SerializableModel):
    """A version-aware excerpt supporting one or more legal findings."""

    legal_source: str
    citation: str
    excerpt: str
    authority_level: AuthorityLevel = AuthorityLevel.UNKNOWN
    document_version: str | None = None
    evidence_id: str = field(default_factory=new_identifier)

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"evidence_id"}
    )

    def __post_init__(self) -> None:
        for field_name in ("legal_source", "citation", "excerpt", "evidence_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.authority_level, AuthorityLevel):
            raise TypeError("authority_level must be an AuthorityLevel")
        if self.document_version is not None and (
            not isinstance(self.document_version, str)
            or not self.document_version.strip()
        ):
            raise ValueError("document_version must be None or a non-empty string")
        validate_stable_identifier(self.evidence_id, field_name="evidence_id")


@dataclass(slots=True)
class FindingEvidenceBinding(SerializableModel):
    """ID-based relationship from one Finding to its Evidence records."""

    finding_id: str
    evidence_refs: list[str]

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"finding_id", "evidence_refs"}
    )

    def __post_init__(self) -> None:
        if not isinstance(self.finding_id, str) or not self.finding_id.strip():
            raise ValueError("finding_id must be a non-empty string")
        if not isinstance(self.evidence_refs, list) or not self.evidence_refs:
            raise ValueError("evidence_refs must be a non-empty list")
        if any(
            not isinstance(evidence_id, str) or not evidence_id.strip()
            for evidence_id in self.evidence_refs
        ):
            raise ValueError("evidence_refs must contain non-empty string IDs")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("evidence_refs must not contain duplicate IDs")
        validate_stable_identifier(self.finding_id, field_name="finding_id")
        for evidence_id in self.evidence_refs:
            validate_stable_identifier(evidence_id, field_name="evidence_refs")
