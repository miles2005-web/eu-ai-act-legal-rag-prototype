"""Scoped rule invocation and authorization contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
import json
from typing import Any, ClassVar

from src.assessment.models import (
    FrozenDict,
    SerializableModel,
    freeze_value,
    validate_stable_identifier,
)
from src.assessment.scope import AssessmentScope, InvocationId


def derive_invocation_id(
    rule_id: str,
    rule_version: str,
    scope: AssessmentScope,
) -> InvocationId:
    """Derive a stable identity without random values or translated text."""

    canonical = json.dumps(
        {
            "rule_id": rule_id,
            "rule_version": rule_version,
            "scope": scope.to_dict(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return InvocationId(f"invocation:{sha256(canonical.encode()).hexdigest()[:32]}")


@dataclass(frozen=True, slots=True)
class RuleInvocation(SerializableModel):
    invocation_id: InvocationId
    rule_id: str
    rule_version: str
    scope: AssessmentScope
    phase: str | None = None
    ordering_key: str | None = None
    prerequisite_invocation_ids: tuple[InvocationId, ...] = ()
    accepted_upstream_statuses: dict[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    authorization_reference: str | None = None
    contract_version: str = "1.0.0"

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"invocation_id"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "invocation_id", InvocationId(self.invocation_id))
        if isinstance(self.scope, dict):
            object.__setattr__(self, "scope", AssessmentScope.from_dict(self.scope))
        if not isinstance(self.scope, AssessmentScope):
            raise TypeError("scope must be an AssessmentScope")
        for field_name in ("rule_id", "rule_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.contract_version != "1.0.0":
            raise ValueError("unsupported RuleInvocation contract_version")
        if all(value is None for value in self.scope.canonical_tuple()):
            raise ValueError("RuleInvocation requires a scoped subject")
        expected = derive_invocation_id(self.rule_id, self.rule_version, self.scope)
        if self.invocation_id != expected:
            raise ValueError("invocation_id does not match rule identity and scope")
        if self.authorization_reference is not None:
            validate_stable_identifier(
                self.authorization_reference,
                field_name="authorization_reference",
            )
        if not isinstance(self.prerequisite_invocation_ids, (list, tuple)):
            raise TypeError(
                "prerequisite_invocation_ids must be a list or tuple"
            )
        prerequisites = tuple(
            InvocationId(item) for item in self.prerequisite_invocation_ids
        )
        if len(set(prerequisites)) != len(prerequisites):
            raise ValueError(
                "prerequisite_invocation_ids must not contain duplicates"
            )
        object.__setattr__(
            self, "prerequisite_invocation_ids", tuple(sorted(prerequisites))
        )
        if not isinstance(self.accepted_upstream_statuses, Mapping):
            raise TypeError("accepted_upstream_statuses must be a mapping")
        canonical_statuses: dict[InvocationId, tuple[str, ...]] = {}
        for dependency, statuses in self.accepted_upstream_statuses.items():
            dependency_id = InvocationId(dependency)
            if not isinstance(statuses, (list, tuple)):
                raise TypeError(
                    "accepted status collections must be a list or tuple"
                )
            normalized_statuses: list[str] = []
            for status in statuses:
                if not isinstance(status, str) or not status.strip():
                    raise ValueError(
                        "accepted statuses must be non-empty strings"
                    )
                normalized_statuses.append(status)
            if not normalized_statuses:
                raise ValueError(
                    "accepted status collections must not be empty"
                )
            if len(set(normalized_statuses)) != len(normalized_statuses):
                raise ValueError("accepted statuses must not contain duplicates")
            canonical_statuses[dependency_id] = tuple(
                sorted(normalized_statuses)
            )
        declared = set(prerequisites)
        configured = set(canonical_statuses)
        undeclared = configured.difference(declared)
        if undeclared:
            raise ValueError(
                "accepted statuses reference undeclared prerequisite "
                f"invocations: {sorted(undeclared)!r}"
            )
        missing = declared.difference(configured)
        if missing:
            raise ValueError(
                "every prerequisite invocation requires accepted statuses: "
                f"{sorted(missing)!r}"
            )
        object.__setattr__(
            self,
            "accepted_upstream_statuses",
            FrozenDict(
                {
                    dependency: canonical_statuses[dependency]
                    for dependency in sorted(canonical_statuses)
                }
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        rule_id: str,
        rule_version: str,
        scope: AssessmentScope,
        **metadata: Any,
    ) -> RuleInvocation:
        return cls(
            invocation_id=derive_invocation_id(rule_id, rule_version, scope),
            rule_id=rule_id,
            rule_version=rule_version,
            scope=scope,
            **metadata,
        )


@dataclass(frozen=True, slots=True)
class AuthorizedRuleInvocation(SerializableModel):
    authorization_id: str
    rule_id: str
    rule_version: str
    scopes: tuple[AssessmentScope, ...] = ()
    subject_selector: str | None = None
    expansion_inputs: dict[str, Any] = field(default_factory=dict)
    authorization_source: str = "unknown"
    authorized_at: datetime | None = None
    contract_version: str = "1.0.0"

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"authorization_id"}
    )

    def __post_init__(self) -> None:
        for field_name in (
            "authorization_id",
            "rule_id",
            "rule_version",
            "authorization_source",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        validate_stable_identifier(
            self.authorization_id, field_name="authorization_id"
        )
        if self.contract_version != "1.0.0":
            raise ValueError(
                "unsupported AuthorizedRuleInvocation contract_version"
            )
        if self.subject_selector is not None:
            if not isinstance(self.subject_selector, str):
                raise TypeError("subject_selector must be a string")
            normalized_selector = self.subject_selector.strip()
            if not normalized_selector:
                raise ValueError(
                    "subject_selector must be a non-empty string"
                )
            object.__setattr__(
                self,
                "subject_selector",
                normalized_selector,
            )
        if not self.scopes and self.subject_selector is None:
            raise ValueError("authorization requires scopes or a subject_selector")
        if self.authorized_at is not None and self.authorized_at.utcoffset() is None:
            raise ValueError("authorized_at must be timezone-aware")
        normalized_scopes = tuple(
            item
            if isinstance(item, AssessmentScope)
            else AssessmentScope.from_dict(item)
            for item in self.scopes
        )
        canonical_scopes = tuple(
            sorted(
                normalized_scopes,
                key=lambda scope: tuple(
                    value or "" for value in scope.canonical_tuple()
                ),
            )
        )
        if len(set(canonical_scopes)) != len(canonical_scopes):
            raise ValueError("scopes must not contain duplicates")
        object.__setattr__(self, "scopes", canonical_scopes)
        canonical_expansion = {
            key: self.expansion_inputs[key] for key in sorted(self.expansion_inputs)
        }
        try:
            json.dumps(
                canonical_expansion,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "expansion_inputs must contain deterministic JSON values"
            ) from exc
        object.__setattr__(
            self, "expansion_inputs", freeze_value(canonical_expansion)
        )
