"""Tests for regulation-neutral rule and finding framework metadata."""

from __future__ import annotations

import unittest

from src.assessment import (
    AssessmentEngine,
    AssessmentFacts,
    Finding,
    FindingCategory,
    FindingStatus,
    LegalBasis,
    RegulatoryFramework,
)
from src.assessment.rules import AssessmentRule, RuleDefinitionError, RuleRegistry


class _LegacyCompatibleRule(AssessmentRule):
    rule_id = "LEGACY_COMPATIBLE"
    version = "1.0"
    category = FindingCategory.INFORMATION_GAP
    required_fact_paths = ()
    legal_basis = (
        LegalBasis(
            instrument="UNSPECIFIED",
            citation="Reference 1",
            anchor="reference:1",
        ),
    )

    def evaluate(self, facts: AssessmentFacts) -> Finding:
        return Finding(
            category=self.category,
            issue_code="LEGACY_COMPATIBLE",
            status=FindingStatus.NOT_ASSESSED,
            title="Legacy-compatible finding",
            summary="Framework metadata was not declared by this legacy rule.",
        )


class _InvalidFrameworkRule(_LegacyCompatibleRule):
    rule_id = "INVALID_FRAMEWORK"
    framework = "GDPR"


class RegulatoryFrameworkMetadataTests(unittest.TestCase):
    def test_legacy_rule_and_finding_default_to_unknown(self) -> None:
        rule = _LegacyCompatibleRule()
        result = AssessmentEngine(RuleRegistry([rule])).run(AssessmentFacts())

        self.assertEqual(rule.framework, RegulatoryFramework.UNKNOWN)
        self.assertEqual(
            result.findings[0].framework,
            RegulatoryFramework.UNKNOWN,
        )
        self.assertEqual(result.findings[0].to_dict()["framework"], "UNKNOWN")

    def test_registry_rejects_non_enum_framework_metadata(self) -> None:
        with self.assertRaises(RuleDefinitionError):
            RuleRegistry([_InvalidFrameworkRule()])


if __name__ == "__main__":
    unittest.main()
