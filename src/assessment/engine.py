"""Execution layer for registered deterministic assessment rules."""

from __future__ import annotations

from copy import deepcopy

from src.assessment.facts import AssessmentFacts
from src.assessment.findings import Finding
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.requirements import (
    FactRequirementValidator,
    RuleRequirementResult,
)
from src.assessment.results import AssessmentResult, RuleExecutionFailure
from src.assessment.rules import AssessmentRule, RuleRegistry


class RuleOutputError(TypeError):
    """Raised internally when a rule violates the finding output contract."""


class AssessmentEngine:
    """Execute a stable registry snapshot without embedding legal rules."""

    VERSION = "2.0.0"

    def __init__(
        self,
        registry: RuleRegistry,
        *,
        engine_version: str = VERSION,
        requirement_validator: FactRequirementValidator | None = None,
    ) -> None:
        if not isinstance(registry, RuleRegistry):
            raise TypeError("registry must be a RuleRegistry")
        if not isinstance(engine_version, str) or not engine_version.strip():
            raise ValueError("engine_version must be a non-empty string")
        if requirement_validator is not None and not isinstance(
            requirement_validator, FactRequirementValidator
        ):
            raise TypeError(
                "requirement_validator must be a FactRequirementValidator"
            )

        self._registry = registry
        self._engine_version = engine_version
        self._requirement_validator = (
            requirement_validator or FactRequirementValidator()
        )

    @property
    def engine_version(self) -> str:
        return self._engine_version

    def run(self, facts: AssessmentFacts) -> AssessmentResult:
        """Execute all registered rules in registration order.

        Each rule receives its own copy of the input facts. A failure in one
        rule is recorded and does not prevent later rules from executing.
        """

        if not isinstance(facts, AssessmentFacts):
            raise TypeError("facts must be an AssessmentFacts instance")

        # Freeze rule ordering for the duration of this execution even if the
        # registry is changed elsewhere after the run begins.
        rules = self._registry.all()
        requirement_results = tuple(
            self._requirement_validator.validate(rule, facts) for rule in rules
        )
        findings: list[Finding] = []
        executed_rule_ids: list[str] = []
        failures: list[RuleExecutionFailure] = []
        missing_fact_requirements: list[RuleRequirementResult] = []

        for rule, requirement_result in zip(rules, requirement_results, strict=True):
            if not requirement_result.is_satisfied:
                missing_fact_requirements.append(requirement_result)
                continue

            executed_rule_ids.append(rule.rule_id)
            try:
                finding = rule.evaluate(deepcopy(facts))
                findings.append(self._prepare_finding(rule, finding))
            except Exception as exc:
                failures.append(
                    RuleExecutionFailure(
                        rule_id=rule.rule_id,
                        rule_version=rule.version,
                        error_type=type(exc).__name__,
                        message=str(exc) or "Rule execution failed without a message",
                    )
                )

        return AssessmentResult(
            findings=findings,
            executed_rule_ids=executed_rule_ids,
            engine_version=self._engine_version,
            failures=failures,
            missing_fact_requirements=missing_fact_requirements,
        )

    @staticmethod
    def _prepare_finding(rule: AssessmentRule, finding: Finding) -> Finding:
        """Validate and complete framework-owned finding metadata."""

        if not isinstance(finding, Finding):
            raise RuleOutputError(
                f"Rule {rule.rule_id!r} returned {type(finding).__name__}, not Finding"
            )
        if finding.category is not rule.category:
            raise RuleOutputError(
                f"Rule {rule.rule_id!r} returned category "
                f"{finding.category.value!r}, expected {rule.category.value!r}"
            )
        if finding.rule_id is not None and finding.rule_id != rule.rule_id:
            raise RuleOutputError(
                f"Rule {rule.rule_id!r} returned mismatched rule_id "
                f"{finding.rule_id!r}"
            )
        if finding.rule_version is not None and finding.rule_version != rule.version:
            raise RuleOutputError(
                f"Rule {rule.rule_id!r} returned mismatched rule_version "
                f"{finding.rule_version!r}"
            )
        if (
            finding.framework is not RegulatoryFramework.UNKNOWN
            and finding.framework is not rule.framework
        ):
            raise RuleOutputError(
                f"Rule {rule.rule_id!r} returned framework "
                f"{finding.framework.value!r}, expected {rule.framework.value!r}"
            )

        finding.rule_id = rule.rule_id
        finding.rule_version = rule.version
        finding.framework = rule.framework
        if not finding.legal_basis:
            finding.legal_basis = deepcopy(list(rule.legal_basis))
        return finding
