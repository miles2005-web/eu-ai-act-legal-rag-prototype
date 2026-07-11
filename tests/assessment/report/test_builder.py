"""Tests for deterministic assessment report construction."""

from __future__ import annotations

import json
import unittest

from src.assessment import (
    AssessmentEngine,
    AssessmentFacts,
    AuthorityLevel,
    Evidence,
    InMemoryEvidenceService,
    ReportBuilder,
    TriState,
)
from src.assessment.facts import UseDomain
from src.assessment.rules import AIActHighRiskEmploymentRule, RuleRegistry


class ReportBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        rule = AIActHighRiskEmploymentRule()
        self.engine = AssessmentEngine(RuleRegistry([rule]))
        self.builder = ReportBuilder()

    def test_report_is_deterministic_and_preserves_traceability(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "Recruitment system ranking candidates"
        facts.use_context.materially_influences_decision = TriState.YES
        assessment_result = self.engine.run(facts)

        evidence_service = InMemoryEvidenceService(
            [
                Evidence(
                    evidence_id="article-6-evidence",
                    legal_source="EU_AI_ACT",
                    citation="Article 6",
                    excerpt="Article 6 supporting excerpt.",
                    document_version="2024/1689",
                    authority_level=AuthorityLevel.BINDING_LEGISLATION,
                ),
                Evidence(
                    evidence_id="annex-iii-evidence",
                    legal_source="EU_AI_ACT",
                    citation="Annex III point 4(a)",
                    excerpt="Annex III point 4(a) supporting excerpt.",
                    document_version="2024/1689",
                    authority_level=AuthorityLevel.BINDING_LEGISLATION,
                ),
            ]
        )
        evidence_result = evidence_service.resolve(assessment_result.findings)

        first_report = self.builder.build(assessment_result, evidence_result)
        second_report = self.builder.build(assessment_result, evidence_result)

        self.assertEqual(first_report.to_dict(), second_report.to_dict())
        self.assertEqual(first_report.generated_at, assessment_result.timestamp)
        self.assertTrue(first_report.report_id.startswith("report:"))
        self.assertTrue(
            first_report.assessment_run_reference.startswith("assessment:")
        )
        self.assertEqual(len(first_report.findings), 1)
        finding = first_report.findings[0]
        self.assertIn("use_context.task", finding.fact_refs)
        self.assertEqual(first_report.evidence_bindings[0].finding_id, finding.finding_id)
        self.assertEqual(
            first_report.evidence_bindings[0].evidence_refs,
            ["article-6-evidence", "annex-iii-evidence"],
        )
        self.assertEqual(len(first_report.evidence), 2)
        self.assertEqual(
            first_report.rule_versions[0].rule_id,
            "AI_ACT_HIGH_RISK_EMPLOYMENT",
        )
        self.assertTrue(first_report.recommendations)
        json.dumps(first_report.to_dict())

    def test_missing_information_is_reported_without_legal_finding(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "Recruitment system ranking candidates"
        assessment_result = self.engine.run(facts)
        evidence_result = InMemoryEvidenceService().resolve(
            assessment_result.findings
        )

        report = self.builder.build(assessment_result, evidence_result)

        self.assertEqual(report.findings, [])
        self.assertEqual(report.evidence_bindings, [])
        self.assertEqual(len(report.missing_information), 1)
        self.assertEqual(
            report.missing_information[0].fact_path,
            "use_context.materially_influences_decision",
        )
        self.assertIn(
            "use_context.materially_influences_decision",
            report.recommendations[0],
        )
        self.assertEqual(
            report.rule_versions[0].rule_id,
            "AI_ACT_HIGH_RISK_EMPLOYMENT",
        )


if __name__ == "__main__":
    unittest.main()
