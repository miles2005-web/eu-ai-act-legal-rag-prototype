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
    RegulatoryFramework,
    ReportBuilder,
    TriState,
)
from src.assessment.facts import UseDomain
from src.assessment.rules import (
    AIActHighRiskEmploymentRule,
    GDPRArticle22RelevanceRule,
    RuleRegistry,
)


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
        self.assertEqual(len(first_report.findings_by_framework), 1)
        self.assertEqual(
            first_report.findings_by_framework[0].framework,
            RegulatoryFramework.EU_AI_ACT,
        )
        self.assertEqual(
            first_report.findings_by_framework[0].findings[0].finding_id,
            first_report.findings[0].finding_id,
        )
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
        self.assertEqual(
            first_report.rule_versions[0].framework,
            RegulatoryFramework.EU_AI_ACT,
        )
        self.assertEqual(
            first_report.assessed_frameworks,
            [RegulatoryFramework.EU_AI_ACT],
        )
        self.assertEqual(
            first_report.authorized_rule_ids,
            ["AI_ACT_HIGH_RISK_EMPLOYMENT"],
        )
        self.assertTrue(first_report.recommendations)
        json.dumps(first_report.to_dict())

    def test_gdpr_only_report_groups_findings_under_gdpr(self) -> None:
        facts = AssessmentFacts()
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.YES
        facts.use_context.materially_influences_decision = TriState.YES
        engine = AssessmentEngine(
            RuleRegistry([GDPRArticle22RelevanceRule()])
        )
        assessment_result = engine.run(facts)
        evidence_result = InMemoryEvidenceService().resolve(
            assessment_result.findings
        )

        report = self.builder.build(assessment_result, evidence_result)

        self.assertEqual(len(report.findings), 1)
        self.assertEqual(len(report.findings_by_framework), 1)
        group = report.findings_by_framework[0]
        self.assertEqual(group.framework, RegulatoryFramework.GDPR)
        self.assertEqual(
            report.rule_versions[0].framework,
            RegulatoryFramework.GDPR,
        )
        self.assertEqual(
            report.assessed_frameworks,
            [RegulatoryFramework.GDPR],
        )
        self.assertEqual(
            [finding.finding_id for finding in group.findings],
            [report.findings[0].finding_id],
        )

    def test_mixed_report_groups_frameworks_and_preserves_bindings(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "Recruitment system ranking candidates"
        facts.use_context.materially_influences_decision = TriState.YES
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.YES
        engine = AssessmentEngine(
            RuleRegistry(
                [
                    GDPRArticle22RelevanceRule(),
                    AIActHighRiskEmploymentRule(),
                ]
            )
        )
        assessment_result = engine.run(facts)
        evidence_service = InMemoryEvidenceService(
            [
                Evidence(
                    evidence_id="gdpr-article-22",
                    legal_source="GDPR",
                    citation="Article 22(1)",
                    excerpt="GDPR Article 22 supporting excerpt.",
                    authority_level=AuthorityLevel.BINDING_LEGISLATION,
                ),
                Evidence(
                    evidence_id="ai-act-article-6",
                    legal_source="EU_AI_ACT",
                    citation="Article 6",
                    excerpt="EU AI Act Article 6 supporting excerpt.",
                    authority_level=AuthorityLevel.BINDING_LEGISLATION,
                ),
            ]
        )
        evidence_result = evidence_service.resolve(assessment_result.findings)

        first_report = self.builder.build(assessment_result, evidence_result)
        second_report = self.builder.build(assessment_result, evidence_result)

        self.assertEqual(first_report.to_dict(), second_report.to_dict())
        self.assertEqual(
            [group.framework for group in first_report.findings_by_framework],
            [RegulatoryFramework.EU_AI_ACT, RegulatoryFramework.GDPR],
        )
        self.assertEqual(
            first_report.assessed_frameworks,
            [RegulatoryFramework.GDPR, RegulatoryFramework.EU_AI_ACT],
        )
        self.assertEqual(
            [
                finding.finding_id
                for group in first_report.findings_by_framework
                for finding in group.findings
            ],
            [
                finding.finding_id
                for framework in (
                    RegulatoryFramework.EU_AI_ACT,
                    RegulatoryFramework.GDPR,
                )
                for finding in first_report.findings
                if finding.framework is framework
            ],
        )
        self.assertEqual(
            {binding.finding_id for binding in first_report.evidence_bindings},
            {finding.finding_id for finding in first_report.findings},
        )

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
        self.assertEqual(
            report.rule_versions[0].framework,
            RegulatoryFramework.EU_AI_ACT,
        )
        self.assertEqual(
            report.missing_information[0].framework,
            RegulatoryFramework.EU_AI_ACT,
        )


if __name__ == "__main__":
    unittest.main()
