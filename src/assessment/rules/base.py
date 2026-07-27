"""Reusable interface for deterministic assessment rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType

from src.assessment.facts import AssessmentFacts
from src.assessment.findings import Finding, FindingCategory, LegalBasis
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.rules.planning import (
    RulePhase,
    RulePlanningMetadata,
)


class RuleDefinitionError(ValueError):
    """Raised when a rule does not provide valid framework metadata."""


class AssessmentRule(ABC):
    """Contract implemented by every deterministic assessment rule.

    Concrete rules declare their metadata as class attributes and implement
    ``evaluate`` as a pure transformation from an ``AssessmentFacts`` snapshot
    to one ``Finding``. The interface itself performs no legal analysis.
    """

    framework: RegulatoryFramework = RegulatoryFramework.UNKNOWN
    rule_id: str
    version: str
    category: FindingCategory
    required_fact_paths: tuple[str, ...]
    legal_basis: tuple[LegalBasis, ...]
    planning_phase: RulePhase = RulePhase.SCREENING
    planning_ordering_key: str | None = None
    planning_dependencies: tuple[str, ...] = ()
    planning_accepted_upstream_statuses: Mapping[
        str, tuple[str, ...]
    ] = MappingProxyType({})
    planning_subject_selector: str = "legacy_case_scope"
    planning_metadata_version: str = RulePlanningMetadata.CONTRACT_VERSION

    @abstractmethod
    def evaluate(self, facts: AssessmentFacts) -> Finding:
        """Evaluate a fact snapshot and return one structured finding."""

        raise NotImplementedError

    def required_fact_paths_for(
        self,
        facts: AssessmentFacts,
    ) -> tuple[str, ...]:
        """Return fact paths required for the current fact snapshot.

        Most rules have unconditional requirements and inherit this default.
        Rules with transparent conditional predicates may override the method
        while retaining ``required_fact_paths`` as their complete metadata
        superset.
        """

        if not isinstance(facts, AssessmentFacts):
            raise TypeError("facts must be an AssessmentFacts instance")
        return self.required_fact_paths

    def planning_metadata(self) -> RulePlanningMetadata:
        """Return explicit planning metadata without changing rule execution.

        The default is the compatibility contract for v0.5 rules: screening,
        no dependencies, and a stable rule-ID ordering key. Concrete rules may
        declare execution-significant metadata as class attributes. F2A only
        validates and exposes this metadata; ``AssessmentEngine`` continues to
        execute in registry order.
        """

        ordering_key = (
            self.planning_ordering_key
            if self.planning_ordering_key is not None
            else f"legacy:{self.rule_id}"
        )
        return RulePlanningMetadata(
            rule_id=self.rule_id,
            rule_version=self.version,
            phase=self.planning_phase,
            ordering_key=ordering_key,
            dependencies=self.planning_dependencies,
            accepted_upstream_statuses=(
                self.planning_accepted_upstream_statuses
            ),
            subject_selector=self.planning_subject_selector,
            contract_version=self.planning_metadata_version,
        )

    def validate_definition(self) -> None:
        """Validate metadata required for safe registration and lookup."""

        rule_id = getattr(self, "rule_id", None)
        if not isinstance(rule_id, str) or not rule_id.strip():
            raise RuleDefinitionError("rule_id must be a non-empty string")

        version = getattr(self, "version", None)
        if not isinstance(version, str) or not version.strip():
            raise RuleDefinitionError(
                f"Rule {rule_id!r} must define a non-empty version"
            )

        framework = getattr(self, "framework", RegulatoryFramework.UNKNOWN)
        if not isinstance(framework, RegulatoryFramework):
            raise RuleDefinitionError(
                f"Rule {rule_id!r} must define a RegulatoryFramework"
            )

        category = getattr(self, "category", None)
        if not isinstance(category, FindingCategory):
            raise RuleDefinitionError(
                f"Rule {rule_id!r} must define a FindingCategory"
            )

        required_fact_paths = getattr(self, "required_fact_paths", None)
        if not isinstance(required_fact_paths, tuple):
            raise RuleDefinitionError(
                f"Rule {rule_id!r} required_fact_paths must be a tuple"
            )
        if any(not isinstance(path, str) or not path.strip() for path in required_fact_paths):
            raise RuleDefinitionError(
                f"Rule {rule_id!r} contains an invalid required fact path"
            )
        if len(set(required_fact_paths)) != len(required_fact_paths):
            raise RuleDefinitionError(
                f"Rule {rule_id!r} contains duplicate required fact paths"
            )

        legal_basis = getattr(self, "legal_basis", None)
        if not isinstance(legal_basis, tuple) or not legal_basis:
            raise RuleDefinitionError(
                f"Rule {rule_id!r} must define non-empty legal_basis metadata"
            )
        if any(not isinstance(basis, LegalBasis) for basis in legal_basis):
            raise RuleDefinitionError(
                f"Rule {rule_id!r} legal_basis must contain LegalBasis values"
            )
        if any(
            not basis.instrument.strip()
            or not basis.citation.strip()
            or not basis.anchor.strip()
            for basis in legal_basis
        ):
            raise RuleDefinitionError(
                f"Rule {rule_id!r} contains incomplete legal basis metadata"
            )
