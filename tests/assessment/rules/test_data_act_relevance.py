"""Tests for the preliminary EU Data Act relevance trigger."""

from __future__ import annotations

import unittest

from src.assessment import (
    AssessmentEngine,
    AssessmentFacts,
    FindingCategory,
    FindingStatus,
    RegulatoryFramework,
    TriState,
)
from src.assessment.requirements import MissingFactReason
from src.assessment.rules import EUDataActRelevanceRule, RuleRegistry


class EUDataActRelevanceRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = EUDataActRelevanceRule()
        self.engine = AssessmentEngine(RuleRegistry([self.rule]))

    @staticmethod
    def _complete_facts() -> AssessmentFacts:
        facts = AssessmentFacts()
        facts.data_act.connected_product = TriState.NO
        facts.data_act.related_service = TriState.NO
        facts.data_act.data_generated = TriState.NO
        return facts

    def test_connected_product_with_generated_data_is_potentially_relevant(
        self,
    ) -> None:
        facts = self._complete_facts()
        facts.data_act.connected_product = TriState.YES
        facts.data_act.data_generated = TriState.YES

        result = self.engine.run(facts)

        self.assertEqual(
            result.executed_rule_ids,
            ["EU_DATA_ACT_RELEVANCE"],
        )
        self.assertEqual(
            result.assessed_frameworks,
            [RegulatoryFramework.EU_DATA_ACT],
        )
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.framework, RegulatoryFramework.EU_DATA_ACT)
        self.assertEqual(finding.category, FindingCategory.DATA_GOVERNANCE)
        self.assertEqual(finding.status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertEqual(
            finding.title,
            "Data Act relevance potentially applies",
        )
        self.assertTrue(finding.requires_legal_review)
        self.assertIn("preliminary trigger", finding.summary.lower())
        self.assertNotIn("non-compliance", finding.summary.lower())
        self.assertEqual(len(finding.trace), 3)

    def test_unrelated_system_does_not_meet_trigger(self) -> None:
        result = self.engine.run(self._complete_facts())

        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.status, FindingStatus.DOES_NOT_APPLY)
        self.assertIn(
            "NO_CONNECTED_PRODUCT_OR_RELATED_SERVICE",
            finding.reason_codes,
        )
        self.assertIn("NO_DATA_GENERATED", finding.reason_codes)
        self.assertFalse(finding.requires_legal_review)

    def test_unknown_required_fact_is_handled_by_requirement_validation(
        self,
    ) -> None:
        facts = self._complete_facts()
        facts.data_act.related_service = TriState.UNKNOWN

        result = self.engine.run(facts)

        self.assertEqual(result.findings, [])
        self.assertEqual(result.executed_rule_ids, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.missing_fact_requirements), 1)
        requirement = result.missing_fact_requirements[0]
        self.assertEqual(
            requirement.framework,
            RegulatoryFramework.EU_DATA_ACT,
        )
        self.assertEqual(len(requirement.missing_facts), 1)
        self.assertEqual(
            requirement.missing_facts[0].fact_path,
            "data_act.related_service",
        )
        self.assertEqual(
            requirement.missing_facts[0].reason,
            MissingFactReason.UNKNOWN,
        )


if __name__ == "__main__":
    unittest.main()
