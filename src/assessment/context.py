"""Read-only assessment context contract for one future v0.6 invocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from src.assessment.baselines import AssessmentBaseline
from src.assessment.facts import AssessmentFacts
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.invocations import (
    AuthorizedRuleInvocation,
    RuleInvocation,
    derive_invocation_id,
)
from src.assessment.models import (
    SerializableModel,
    freeze_value,
    to_primitive,
    validate_stable_identifier,
)
from src.assessment.recruitment_models import (
    normalize_enum,
    normalized_string_collection,
)
from src.assessment.scope import AssessmentScope, InvocationId


@dataclass(frozen=True, slots=True)
class PrerequisiteFindingSummary(SerializableModel):
    finding_id: str
    prerequisite_invocation_id: InvocationId
    scope: AssessmentScope
    framework: RegulatoryFramework
    rule_id: str
    rule_version: str
    status: str
    reason_codes: tuple[str, ...] = ()
    trace_references: tuple[str, ...] = ()

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"finding_id", "prerequisite_invocation_id"}
    )

    def __post_init__(self) -> None:
        validate_stable_identifier(self.finding_id, field_name="finding_id")
        object.__setattr__(
            self,
            "prerequisite_invocation_id",
            InvocationId(self.prerequisite_invocation_id),
        )
        if isinstance(self.scope, dict):
            object.__setattr__(self, "scope", AssessmentScope.from_dict(self.scope))
        if not isinstance(self.scope, AssessmentScope):
            raise TypeError("scope must be an AssessmentScope")
        object.__setattr__(
            self,
            "framework",
            normalize_enum(
                self.framework,
                RegulatoryFramework,
                field_name="framework",
            ),
        )
        for field_name in ("rule_id", "rule_version", "status"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        object.__setattr__(
            self,
            "reason_codes",
            normalized_string_collection(
                self.reason_codes,
                field_name="reason_codes",
            ),
        )
        object.__setattr__(
            self,
            "trace_references",
            normalized_string_collection(
                self.trace_references,
                field_name="trace_references",
            ),
        )


class AssessmentContext(SerializableModel):
    """Defensively copied, service-free input contract treated as immutable."""

    CONTRACT_VERSION = "1.0.0"
    __slots__ = (
        "_facts_snapshot",
        "_invocation",
        "_prerequisites",
        "_authorization",
        "_baseline",
        "_sealed",
    )

    def __init__(
        self,
        *,
        facts_snapshot: AssessmentFacts,
        invocation: RuleInvocation,
        prerequisite_findings: tuple[PrerequisiteFindingSummary, ...] = (),
        authorization: AuthorizedRuleInvocation,
        baseline: AssessmentBaseline,
        contract_version: str = CONTRACT_VERSION,
    ) -> None:
        if contract_version != self.CONTRACT_VERSION:
            raise ValueError("unsupported AssessmentContext contract_version")
        if not isinstance(facts_snapshot, AssessmentFacts):
            raise TypeError("facts_snapshot must be AssessmentFacts")
        if not isinstance(invocation, RuleInvocation):
            raise TypeError("invocation must be a RuleInvocation")
        if not isinstance(authorization, AuthorizedRuleInvocation):
            raise TypeError(
                "authorization must be an AuthorizedRuleInvocation"
            )
        if not isinstance(baseline, AssessmentBaseline):
            raise TypeError("baseline must be an AssessmentBaseline")
        if not isinstance(prerequisite_findings, (list, tuple)):
            raise TypeError("prerequisite_findings must be a list or tuple")
        if any(
            not isinstance(item, PrerequisiteFindingSummary)
            for item in prerequisite_findings
        ):
            raise TypeError(
                "prerequisite_findings must contain "
                "PrerequisiteFindingSummary instances"
            )
        if invocation.rule_id != authorization.rule_id:
            raise ValueError("invocation and authorization rule_id must match")
        if invocation.rule_version != authorization.rule_version:
            raise ValueError("invocation and authorization rule_version must match")
        if facts_snapshot.schema_version != baseline.facts_schema_version:
            raise ValueError(
                "facts_snapshot schema_version does not match baseline"
            )
        if baseline.assessment_context_version != self.CONTRACT_VERSION:
            raise ValueError(
                "baseline assessment_context_version is incompatible"
            )
        matching_rules = [
            item
            for item in baseline.ordered_rules
            if item.rule_id == invocation.rule_id
        ]
        if len(matching_rules) != 1:
            raise ValueError(
                "invocation rule must appear exactly once in baseline.ordered_rules"
            )
        if matching_rules[0].rule_version != invocation.rule_version:
            raise ValueError(
                "invocation rule_version does not match baseline.ordered_rules"
            )
        if not authorization.scopes:
            raise ValueError(
                "AssessmentContext requires an explicitly expanded authorization scope"
            )
        if invocation.scope not in authorization.scopes:
            raise ValueError("invocation scope is not explicitly authorized")
        if (
            invocation.authorization_reference is not None
            and invocation.authorization_reference
            != authorization.authorization_id
        ):
            raise ValueError(
                "invocation authorization_reference does not match authorization"
            )
        self._validate_prerequisites(
            invocation,
            prerequisite_findings,
            baseline,
        )
        object.__setattr__(
            self, "_facts_snapshot", freeze_value(facts_snapshot.to_dict())
        )
        object.__setattr__(self, "_invocation", RuleInvocation.from_dict(invocation.to_dict()))
        object.__setattr__(
            self,
            "_prerequisites",
            tuple(
                PrerequisiteFindingSummary.from_dict(item.to_dict())
                for item in prerequisite_findings
            ),
        )
        object.__setattr__(
            self,
            "_authorization",
            AuthorizedRuleInvocation.from_dict(authorization.to_dict()),
        )
        object.__setattr__(
            self, "_baseline", AssessmentBaseline.from_dict(baseline.to_dict())
        )
        object.__setattr__(self, "_sealed", True)

    @staticmethod
    def _validate_prerequisites(
        invocation: RuleInvocation,
        prerequisite_findings: list[PrerequisiteFindingSummary]
        | tuple[PrerequisiteFindingSummary, ...],
        baseline: AssessmentBaseline,
    ) -> None:
        declared = set(invocation.prerequisite_invocation_ids)
        seen: set[InvocationId] = set()
        for summary in prerequisite_findings:
            dependency_id = summary.prerequisite_invocation_id
            if dependency_id not in declared:
                raise ValueError(
                    "prerequisite Finding summary references an undeclared "
                    "invocation"
                )
            if dependency_id in seen:
                raise ValueError(
                    "duplicate prerequisite Finding summary for invocation"
                )
            seen.add(dependency_id)
            expected = derive_invocation_id(
                summary.rule_id,
                summary.rule_version,
                summary.scope,
            )
            if expected != dependency_id:
                raise ValueError(
                    "prerequisite Finding rule metadata does not match its "
                    "invocation identity"
                )
            baseline_entries = [
                item
                for item in baseline.ordered_rules
                if item.rule_id == summary.rule_id
            ]
            if len(baseline_entries) != 1:
                raise ValueError(
                    "prerequisite Finding rule must appear exactly once in "
                    "baseline.ordered_rules"
                )
            if baseline_entries[0].rule_version != summary.rule_version:
                raise ValueError(
                    "prerequisite Finding rule_version does not match "
                    "baseline.ordered_rules"
                )
            accepted = invocation.accepted_upstream_statuses[str(dependency_id)]
            if summary.status not in accepted:
                raise ValueError(
                    "prerequisite Finding status is not accepted by invocation"
                )
        missing = declared.difference(seen)
        if missing:
            raise ValueError(
                "every declared prerequisite invocation requires exactly one "
                f"Finding summary: {sorted(missing)!r}"
            )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("AssessmentContext is immutable")
        object.__setattr__(self, name, value)

    @property
    def facts_snapshot(self) -> AssessmentFacts:
        return AssessmentFacts.from_dict(to_primitive(self._facts_snapshot))

    @property
    def invocation(self) -> RuleInvocation:
        return RuleInvocation.from_dict(self._invocation.to_dict())

    @property
    def prerequisite_findings(self) -> tuple[PrerequisiteFindingSummary, ...]:
        return tuple(
            PrerequisiteFindingSummary.from_dict(item.to_dict())
            for item in self._prerequisites
        )

    @property
    def authorization(self) -> AuthorizedRuleInvocation:
        return AuthorizedRuleInvocation.from_dict(self._authorization.to_dict())

    @property
    def baseline(self) -> AssessmentBaseline:
        return AssessmentBaseline.from_dict(self._baseline.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.CONTRACT_VERSION,
            "facts_snapshot": to_primitive(self._facts_snapshot),
            "invocation": self._invocation.to_dict(),
            "prerequisite_findings": [
                summary.to_dict() for summary in self._prerequisites
            ],
            "authorization": self._authorization.to_dict(),
            "baseline": self._baseline.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AssessmentContext:
        if not isinstance(payload, dict):
            raise TypeError("AssessmentContext payload must be an object")
        required = {
            "contract_version",
            "facts_snapshot",
            "invocation",
            "prerequisite_findings",
            "authorization",
            "baseline",
        }
        unknown = sorted(set(payload).difference(required))
        missing = sorted(required.difference(payload))
        if unknown:
            raise ValueError(
                "unknown AssessmentContext fields: " + ", ".join(unknown)
            )
        if missing:
            raise ValueError(
                "AssessmentContext payload is missing fields: " + ", ".join(missing)
            )
        if payload["contract_version"] != cls.CONTRACT_VERSION:
            raise ValueError("unsupported AssessmentContext contract_version")
        return cls(
            contract_version=str(payload["contract_version"]),
            facts_snapshot=AssessmentFacts.from_dict(payload["facts_snapshot"]),
            invocation=RuleInvocation.from_dict(payload["invocation"]),
            prerequisite_findings=tuple(
                PrerequisiteFindingSummary.from_dict(item)
                for item in payload["prerequisite_findings"]
            ),
            authorization=AuthorizedRuleInvocation.from_dict(
                payload["authorization"]
            ),
            baseline=AssessmentBaseline.from_dict(payload["baseline"]),
        )
