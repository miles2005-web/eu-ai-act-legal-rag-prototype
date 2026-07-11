"""Tests for preliminary employment-related high-risk screening."""

from __future__ import annotations

import unittest

from src.assessment import AssessmentEngine, AssessmentFacts, FindingStatus, TriState
from src.assessment.facts import UseDomain
from src.assessment.requirements import MissingFactReason
from src.assessment.rules import AIActHighRiskEmploymentRule, RuleRegistry


class AIActHighRiskEmploymentRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = AIActHighRiskEmploymentRule()
        self.engine = AssessmentEngine(RuleRegistry([self.rule]))

    def test_recruitment_ai_ranking_candidates_potentially_applies(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "Recruitment system ranking job candidates"
        facts.use_context.materially_influences_decision = TriState.YES

        result = self.engine.run(facts)

        self.assertEqual(
            result.executed_rule_ids,
            ["AI_ACT_HIGH_RISK_EMPLOYMENT"],
        )
        self.assertEqual(result.failures, [])
        self.assertEqual(result.missing_fact_requirements, [])
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertEqual(finding.category.value, "high_risk")
        self.assertTrue(finding.requires_legal_review)
        self.assertIn("not a definitive", finding.summary.lower())
        self.assertEqual(len(finding.trace), 3)
        self.assertEqual(
            [basis.citation for basis in finding.legal_basis],
            ["Article 6", "Annex III point 4(a)"],
        )

    def test_employment_unrelated_system_does_not_apply(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.OTHER
        facts.use_context.task = "Customer support response generation"
        facts.use_context.materially_influences_decision = TriState.YES

        result = self.engine.run(facts)

        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.status, FindingStatus.DOES_NOT_APPLY)
        self.assertIn("NOT_EMPLOYMENT_CONTEXT", finding.reason_codes)
        self.assertFalse(finding.requires_legal_review)

    def test_missing_material_influence_is_skipped(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "Recruitment system ranking candidates"

        result = self.engine.run(facts)

        self.assertEqual(result.findings, [])
        self.assertEqual(result.executed_rule_ids, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(len(result.missing_fact_requirements), 1)
        missing = result.missing_fact_requirements[0].missing_facts
        self.assertEqual(len(missing), 1)
        self.assertEqual(
            missing[0].fact_path,
            "use_context.materially_influences_decision",
        )
        self.assertEqual(missing[0].reason, MissingFactReason.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
