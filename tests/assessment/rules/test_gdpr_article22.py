"""Tests for the preliminary GDPR Article 22 relevance trigger."""

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
from src.assessment.rules import GDPRArticle22RelevanceRule, RuleRegistry


class GDPRArticle22RelevanceRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = GDPRArticle22RelevanceRule()
        self.engine = AssessmentEngine(RuleRegistry([self.rule]))

    @staticmethod
    def _complete_facts() -> AssessmentFacts:
        facts = AssessmentFacts()
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.YES
        facts.use_context.materially_influences_decision = TriState.YES
        return facts

    def test_positive_recruitment_scenario_is_potentially_relevant(self) -> None:
        result = self.engine.run(self._complete_facts())

        self.assertEqual(
            result.executed_rule_ids,
            ["GDPR_ARTICLE22_RELEVANCE"],
        )
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.framework, RegulatoryFramework.GDPR)
        self.assertEqual(finding.category, FindingCategory.DATA_PROTECTION)
        self.assertEqual(finding.status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertTrue(finding.requires_legal_review)
        self.assertIn("does not determine", finding.summary.lower())
        self.assertEqual(
            [basis.citation for basis in finding.legal_basis],
            ["Article 22(1)"],
        )
        self.assertEqual(len(finding.trace), 3)

    def test_known_negative_scenario_does_not_meet_trigger(self) -> None:
        facts = self._complete_facts()
        facts.data_protection.automated_individual_decision = TriState.NO

        result = self.engine.run(facts)

        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.status, FindingStatus.DOES_NOT_APPLY)
        self.assertIn(
            "NO_AUTOMATED_INDIVIDUAL_DECISION",
            finding.reason_codes,
        )
        self.assertFalse(finding.requires_legal_review)

    def test_unknown_required_fact_is_handled_by_requirement_validation(self) -> None:
        facts = self._complete_facts()
        facts.data_protection.personal_data_processed = TriState.UNKNOWN

        result = self.engine.run(facts)

        self.assertEqual(result.findings, [])
        self.assertEqual(result.executed_rule_ids, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.missing_fact_requirements), 1)
        missing = result.missing_fact_requirements[0].missing_facts
        self.assertEqual(len(missing), 1)
        self.assertEqual(
            missing[0].fact_path,
            "data_protection.personal_data_processed",
        )
        self.assertEqual(missing[0].reason, MissingFactReason.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
