"""End-to-end regressions for the explicit Industrial multi-framework demo."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from scripts.run_demo_assessment import build_assessment_facts, load_fixture
from src.assessment.demo import create_assessment_workflow
from src.assessment.findings import FindingStatus
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import TriState
from src.assessment.questionnaire.definitions import (
    AI_ACT_PRODUCT_SAFETY_RULE_ID,
    EU_DATA_ACT_RULE_ID,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MULTI_FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "industrial_robot_multiframework_case.json"
)
INDUSTRIAL_FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "industrial_ai_case.json"
)
DATA_ACT_EVIDENCE_IDS = (
    "evidence:v2:064b49b6e320dcc92f19af0c81a63537",
    "evidence:v2:2a1c882fe06379842fbf34b243cefec5",
    "evidence:v2:da1a8ad793bf79ed89c5591788e98322",
)
AI_ACT_EVIDENCE_IDS = (
    "evidence:v2:71fd01f91c4dc552ef2f067d839a12ba",
    "evidence:v2:21ccd8df4d5b12fc796eaa4418cb79e4",
    "evidence:v2:c66cf0d29225996849193a5cd53dce91",
    "evidence:v2:ac2cfe89633b4b87506b1980e0f73d39",
)


class IndustrialMultiFrameworkDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = load_fixture(MULTI_FIXTURE_PATH)

    def _run(self, facts=None, *, rule_ids=None):
        bundle = create_assessment_workflow()
        assessment_case = bundle.case_service.create_case(
            self.payload["scenario"]["name"],
            description=self.payload["scenario"]["description"],
            facts=facts or build_assessment_facts(self.payload["facts"]),
        )
        report = bundle.workflow.run(
            assessment_case.case_id,
            rule_ids=(rule_ids or reversed(self.payload["authorized_rule_ids"])),
        )
        return bundle, report

    @staticmethod
    def _evidence_for(report, rule_id: str):
        finding = next(item for item in report.findings if item.rule_id == rule_id)
        binding = next(
            item
            for item in report.evidence_bindings
            if item.finding_id == finding.finding_id
        )
        evidence_by_id = {item.evidence_id: item for item in report.evidence}
        return finding, [evidence_by_id[item] for item in binding.evidence_refs]

    def test_fixture_contains_explicit_independent_confirmations(self) -> None:
        facts = build_assessment_facts(self.payload["facts"])

        self.assertEqual(
            self.payload["authorized_rule_ids"],
            [AI_ACT_PRODUCT_SAFETY_RULE_ID, EU_DATA_ACT_RULE_ID],
        )
        self.assertIs(facts.product_regulation.ai_is_product, TriState.NO)
        self.assertIs(facts.product_regulation.ai_is_safety_component, TriState.YES)
        self.assertEqual(
            facts.product_regulation.annex_i_instrument,
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
        )
        self.assertIs(
            facts.product_regulation.annex_i_instrument_confirmed,
            TriState.YES,
        )
        self.assertIs(
            facts.product_regulation.third_party_conformity_required,
            TriState.YES,
        )
        self.assertIs(facts.data_act.connected_product, TriState.YES)
        self.assertIs(facts.data_act.related_service, TriState.YES)
        self.assertIs(facts.data_act.data_generated, TriState.YES)
        for path in (
            "product_regulation.annex_i_instrument_confirmed",
            "product_regulation.third_party_conformity_required",
            "data_act.connected_product",
            "data_act.related_service",
            "data_act.data_generated",
        ):
            self.assertIn(path, facts.fact_metadata)

    def test_positive_report_has_two_independent_ordered_findings(self) -> None:
        _, report = self._run()
        expected_findings = self.payload["expected_assessment"]["findings"]

        self.assertEqual(
            report.authorized_rule_ids,
            [AI_ACT_PRODUCT_SAFETY_RULE_ID, EU_DATA_ACT_RULE_ID],
        )
        self.assertEqual(
            [item.rule_id for item in report.findings],
            [AI_ACT_PRODUCT_SAFETY_RULE_ID, EU_DATA_ACT_RULE_ID],
        )
        self.assertEqual(
            [item.framework for item in report.findings],
            [RegulatoryFramework.EU_AI_ACT, RegulatoryFramework.EU_DATA_ACT],
        )
        self.assertTrue(
            all(
                item.status is FindingStatus.POTENTIALLY_APPLIES
                for item in report.findings
            )
        )
        self.assertEqual(len(report.findings_by_framework), 2)
        self.assertEqual(len(report.evidence), 7)
        self.assertEqual(report.missing_information, [])
        self.assertEqual(report.execution_failures, [])
        self.assertNotIn(
            "AI_ACT_HIGH_RISK_EMPLOYMENT",
            {item.rule_id for item in report.findings},
        )
        self.assertNotIn(
            "GDPR_ARTICLE22_RELEVANCE",
            {item.rule_id for item in report.findings},
        )
        for finding, expected in zip(
            report.findings, expected_findings, strict=True
        ):
            self.assertEqual(finding.rule_id, expected["rule_id"])
            self.assertEqual(finding.framework.value, expected["framework"])
            self.assertEqual(finding.category.name, expected["category"])
            self.assertEqual(finding.status.value, expected["status"])
            _, evidence = self._evidence_for(report, finding.rule_id)
            self.assertEqual(
                [item.citation for item in evidence],
                expected["evidence_citations"],
            )
        self.assertEqual(report.to_dict(), deepcopy(report).to_dict())
        json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)

    def test_positive_findings_keep_atomic_source_isolated_evidence(self) -> None:
        _, report = self._run()
        ai_finding, ai_evidence = self._evidence_for(
            report, AI_ACT_PRODUCT_SAFETY_RULE_ID
        )
        data_finding, data_evidence = self._evidence_for(
            report, EU_DATA_ACT_RULE_ID
        )

        self.assertEqual(
            [item.citation for item in ai_evidence],
            [
                "Article 3(14)",
                "Article 6(1)(a)",
                "Article 6(1)(b)",
                "Annex I, Section A, point 1",
            ],
        )
        self.assertEqual(
            tuple(item.evidence_id for item in ai_evidence),
            AI_ACT_EVIDENCE_IDS,
        )
        self.assertEqual(
            [item.citation for item in data_evidence],
            ["Article 1(1)(a)", "Article 2(5)", "Article 2(6)"],
        )
        self.assertEqual(
            tuple(item.evidence_id for item in data_evidence),
            DATA_ACT_EVIDENCE_IDS,
        )
        self.assertEqual(
            {item.legal_source for item in ai_evidence}, {"EU_AI_ACT"}
        )
        self.assertEqual(
            {item.legal_source for item in data_evidence}, {"EU_DATA_ACT"}
        )
        self.assertEqual(
            {item.instrument for item in ai_finding.legal_basis}, {"EU_AI_ACT"}
        )
        self.assertEqual(
            {item.instrument for item in data_finding.legal_basis},
            {"EU_DATA_ACT"},
        )
        self.assertNotEqual(ai_finding.reason_codes, data_finding.reason_codes)

    def test_unresolved_article_6_1_does_not_block_data_act(self) -> None:
        facts = build_assessment_facts(self.payload["facts"])
        facts.product_regulation.annex_i_instrument = None
        facts.product_regulation.annex_i_instrument_confirmed = TriState.UNKNOWN

        _, report = self._run(facts)

        self.assertEqual(
            [item.rule_id for item in report.findings], [EU_DATA_ACT_RULE_ID]
        )
        self.assertIs(
            report.findings[0].status, FindingStatus.POTENTIALLY_APPLIES
        )
        self.assertEqual(
            {item.rule_id for item in report.missing_information},
            {AI_ACT_PRODUCT_SAFETY_RULE_ID},
        )
        self.assertEqual(
            [item.fact_path for item in report.missing_information],
            [
                "product_regulation.annex_i_instrument",
                "product_regulation.annex_i_instrument_confirmed",
            ],
        )
        self.assertEqual(
            tuple(item.evidence_id for item in report.evidence),
            DATA_ACT_EVIDENCE_IDS,
        )

    def test_negative_article_6_1_is_scoped_and_keeps_data_act_positive(self) -> None:
        facts = build_assessment_facts(self.payload["facts"])
        facts.product_regulation.ai_is_product = TriState.NO
        facts.product_regulation.ai_is_safety_component = TriState.NO

        _, report = self._run(facts)
        findings = {item.rule_id: item for item in report.findings}

        self.assertIs(
            findings[AI_ACT_PRODUCT_SAFETY_RULE_ID].status,
            FindingStatus.DOES_NOT_APPLY,
        )
        self.assertIn(
            "Article 6(1)", findings[AI_ACT_PRODUCT_SAFETY_RULE_ID].summary
        )
        self.assertIn(
            "does not exclude Article 6(2)",
            findings[AI_ACT_PRODUCT_SAFETY_RULE_ID].summary,
        )
        self.assertIs(
            findings[EU_DATA_ACT_RULE_ID].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        _, data_evidence = self._evidence_for(report, EU_DATA_ACT_RULE_ID)
        self.assertEqual(
            tuple(item.evidence_id for item in data_evidence),
            DATA_ACT_EVIDENCE_IDS,
        )

    def test_existing_industrial_fixture_remains_data_act_only(self) -> None:
        payload = load_fixture(INDUSTRIAL_FIXTURE_PATH)
        facts = build_assessment_facts(payload["facts"])
        bundle = create_assessment_workflow()
        assessment_case = bundle.case_service.create_case(
            payload["scenario"]["name"], facts=facts
        )
        report = bundle.workflow.run(
            assessment_case.case_id,
            rule_ids=(EU_DATA_ACT_RULE_ID,),
        )

        self.assertEqual(report.authorized_rule_ids, [EU_DATA_ACT_RULE_ID])
        self.assertEqual([item.rule_id for item in report.findings], [EU_DATA_ACT_RULE_ID])
        self.assertEqual(len(report.evidence), 3)
        self.assertEqual(
            tuple(item.evidence_id for item in report.evidence),
            DATA_ACT_EVIDENCE_IDS,
        )
        self.assertIs(facts.product_regulation.ai_is_product, TriState.UNKNOWN)
        self.assertIs(
            facts.product_regulation.ai_is_safety_component, TriState.UNKNOWN
        )


if __name__ == "__main__":
    unittest.main()
