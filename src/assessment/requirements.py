"""Required-fact validation for deterministic assessment rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.assessment.facts import AssessmentFacts
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import SerializableModel
from src.assessment.rules.base import AssessmentRule


class MissingFactReason(str, Enum):
    """Why a required fact cannot currently be used by a rule."""

    NOT_PROVIDED = "not_provided"
    UNKNOWN = "unknown"
    PATH_NOT_FOUND = "path_not_found"


@dataclass(slots=True)
class MissingFact(SerializableModel):
    """One unresolved fact path needed before a rule can execute."""

    fact_path: str
    reason: MissingFactReason


@dataclass(slots=True)
class RuleRequirementResult(SerializableModel):
    """Required-fact validation result for one registered rule."""

    rule_id: str
    rule_version: str
    required_fact_paths: list[str]
    missing_facts: list[MissingFact] = field(default_factory=list)
    framework: RegulatoryFramework = RegulatoryFramework.UNKNOWN

    @property
    def is_satisfied(self) -> bool:
        return not self.missing_facts


class FactRequirementValidator:
    """Resolve rule fact paths without applying any legal logic."""

    _PATH_NOT_FOUND = object()

    def validate(
        self,
        rule: AssessmentRule,
        facts: AssessmentFacts,
    ) -> RuleRequirementResult:
        """Return missing-fact details for a rule in declaration order."""

        if not isinstance(rule, AssessmentRule):
            raise TypeError("rule must be an AssessmentRule instance")
        if not isinstance(facts, AssessmentFacts):
            raise TypeError("facts must be an AssessmentFacts instance")

        required_fact_paths = rule.required_fact_paths_for(facts)
        if not isinstance(required_fact_paths, tuple):
            raise TypeError("required_fact_paths_for must return a tuple")
        if any(
            not isinstance(path, str) or not path.strip()
            for path in required_fact_paths
        ):
            raise ValueError(
                "required_fact_paths_for returned an invalid fact path"
            )
        if len(set(required_fact_paths)) != len(required_fact_paths):
            raise ValueError(
                "required_fact_paths_for returned duplicate fact paths"
            )

        missing_facts: list[MissingFact] = []
        for fact_path in required_fact_paths:
            value = self._resolve_path(facts, fact_path)
            reason = self._missing_reason(value)
            if reason is not None:
                missing_facts.append(MissingFact(fact_path=fact_path, reason=reason))

        return RuleRequirementResult(
            rule_id=rule.rule_id,
            rule_version=rule.version,
            required_fact_paths=list(required_fact_paths),
            missing_facts=missing_facts,
            framework=rule.framework,
        )

    def _resolve_path(self, facts: AssessmentFacts, fact_path: str) -> Any:
        current: Any = facts
        for segment in fact_path.split("."):
            if not segment or not hasattr(current, segment):
                return self._PATH_NOT_FOUND
            current = getattr(current, segment)
        return current

    def _missing_reason(self, value: Any) -> MissingFactReason | None:
        if value is self._PATH_NOT_FOUND:
            return MissingFactReason.PATH_NOT_FOUND
        if value is None:
            return MissingFactReason.NOT_PROVIDED
        if isinstance(value, str) and not value.strip():
            return MissingFactReason.NOT_PROVIDED
        if isinstance(value, Enum) and value.value == "unknown":
            return MissingFactReason.UNKNOWN
        return None
