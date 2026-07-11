"""In-memory registration and lookup for assessment rules."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from src.assessment.rules.base import AssessmentRule


class DuplicateRuleError(ValueError):
    """Raised when a registry already contains the requested rule ID."""


class RuleNotFoundError(KeyError):
    """Raised when a rule ID is not present in a registry."""


class RuleRegistry:
    """A small deterministic registry keyed by stable rule ID."""

    def __init__(self, rules: Iterable[AssessmentRule] | None = None) -> None:
        self._rules: dict[str, AssessmentRule] = {}
        if rules is not None:
            self.register_many(rules)

    def register(self, rule: AssessmentRule) -> AssessmentRule:
        """Validate and register one rule, rejecting duplicate IDs."""

        if not isinstance(rule, AssessmentRule):
            raise TypeError("rule must be an AssessmentRule instance")

        rule.validate_definition()
        if rule.rule_id in self._rules:
            raise DuplicateRuleError(
                f"Rule ID {rule.rule_id!r} is already registered"
            )

        self._rules[rule.rule_id] = rule
        return rule

    def register_many(self, rules: Iterable[AssessmentRule]) -> None:
        """Register rules in iterable order."""

        for rule in rules:
            self.register(rule)

    def get(self, rule_id: str) -> AssessmentRule:
        """Return a registered rule by ID."""

        try:
            return self._rules[rule_id]
        except KeyError as exc:
            raise RuleNotFoundError(f"Rule ID {rule_id!r} is not registered") from exc

    def all(self) -> tuple[AssessmentRule, ...]:
        """Return all rules in registration order as an immutable tuple."""

        return tuple(self._rules.values())

    def ids(self) -> tuple[str, ...]:
        """Return registered rule IDs in registration order."""

        return tuple(self._rules)

    def __contains__(self, rule_id: object) -> bool:
        return rule_id in self._rules

    def __iter__(self) -> Iterator[AssessmentRule]:
        return iter(self._rules.values())

    def __len__(self) -> int:
        return len(self._rules)

