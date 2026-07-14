"""Smoke tests for presentation-only labels and component imports."""

from __future__ import annotations

import unittest

from src.assessment.evidence import AuthorityLevel, Evidence
from src.assessment.facts import AssessmentFacts, UseDomain
from src.assessment.findings import FindingStatus
from src.assessment.frameworks import RegulatoryFramework
from src.ui.components import (
    framework_label,
    render_assessment_summary_card,
    render_evidence_trace_card,
    render_finding_card,
    render_framework_card,
    render_section_header,
    status_label,
    status_tone,
)
from src.ui.components.common import (
    fact_label,
    fact_value,
    group_evidence_by_citation,
    reasoning_state,
)
from src.ui.styles import ENTERPRISE_STYLES


class UIComponentSmokeTests(unittest.TestCase):
    def test_public_components_are_importable(self) -> None:
        for component in (
            render_assessment_summary_card,
            render_evidence_trace_card,
            render_finding_card,
            render_framework_card,
            render_section_header,
        ):
            self.assertTrue(callable(component))

    def test_evidence_trace_styles_separate_facts_and_rule_summary(self) -> None:
        self.assertIn(".ui-trace-fact-source", ENTERPRISE_STYLES)
        self.assertIn(".ui-rule-application-summary", ENTERPRISE_STYLES)
        self.assertIn(".ui-condition-map-row", ENTERPRISE_STYLES)

    def test_framework_labels_cover_supported_frameworks(self) -> None:
        self.assertEqual(
            framework_label(RegulatoryFramework.EU_AI_ACT),
            "EU AI Act",
        )
        self.assertEqual(framework_label(RegulatoryFramework.GDPR), "GDPR")
        self.assertEqual(
            framework_label(RegulatoryFramework.EU_DATA_ACT),
            "EU Data Act",
        )
        self.assertEqual(
            framework_label(RegulatoryFramework.EU_AI_ACT, "zh-CN"),
            "《欧盟人工智能法案》",
        )

    def test_potentially_applies_uses_review_tone(self) -> None:
        self.assertEqual(
            status_label(FindingStatus.POTENTIALLY_APPLIES),
            "Potentially Applies",
        )
        self.assertEqual(
            status_tone(FindingStatus.POTENTIALLY_APPLIES),
            "warning",
        )
        self.assertEqual(
            status_label(FindingStatus.POTENTIALLY_APPLIES, "zh-CN"),
            "可能适用",
        )

    def test_fact_paths_have_human_readable_labels_and_values(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT

        self.assertEqual(fact_label("use_context.domain"), "Employment context")
        self.assertEqual(fact_value(facts, "use_context.domain"), "Employment")
        self.assertEqual(
            fact_label("use_context.materially_influences_decision"),
            "Material influence on decisions",
        )
        self.assertEqual(
            fact_label("use_context.materially_influences_decision", "zh-CN"),
            "对决策产生实质性影响",
        )
        self.assertEqual(
            fact_value(facts, "use_context.domain", "zh-CN"),
            "就业",
        )

    def test_all_current_cross_framework_requirements_are_humanized(self) -> None:
        expected = {
            "data_protection.personal_data_processed": (
                "Personal data processing",
                "个人数据处理",
            ),
            "data_protection.automated_individual_decision": (
                "Automated individual decision-making",
                "自动化个人决策",
            ),
            "data_act.connected_product": ("Connected product", "互联产品"),
            "data_act.related_service": ("Related service", "相关服务"),
            "data_act.data_generated": (
                "Product or related-service data generation",
                "产品或相关服务生成数据",
            ),
        }
        for fact_path, (english, chinese) in expected.items():
            with self.subTest(fact_path=fact_path):
                self.assertEqual(fact_label(fact_path, "en"), english)
                self.assertEqual(fact_label(fact_path, "zh-CN"), chinese)

    def test_reasoning_results_include_textual_state(self) -> None:
        self.assertEqual(reasoning_state("material_influence"), ("Matched", "matched"))
        self.assertEqual(
            reasoning_state("no_material_influence"),
            ("Not matched", "not-matched"),
        )
        self.assertEqual(
            reasoning_state("personal_data_not_processed"),
            ("Not matched", "not-matched"),
        )
        self.assertEqual(reasoning_state(None), ("Unknown", "unknown"))
        self.assertEqual(
            reasoning_state("material_influence", "zh-CN"),
            ("已满足", "matched"),
        )

    def test_evidence_grouping_preserves_atomic_records_and_ids(self) -> None:
        records = [
            Evidence(
                legal_source="EU_AI_ACT",
                citation="Article 6",
                excerpt="First excerpt",
                authority_level=AuthorityLevel.BINDING_LEGISLATION,
                evidence_id="evidence-1",
            ),
            Evidence(
                legal_source="EU_AI_ACT",
                citation="Annex III point 4(a)",
                excerpt="Annex excerpt",
                authority_level=AuthorityLevel.BINDING_LEGISLATION,
                evidence_id="evidence-2",
            ),
            Evidence(
                legal_source="EU_AI_ACT",
                citation="Article 6",
                excerpt="Second excerpt",
                authority_level=AuthorityLevel.BINDING_LEGISLATION,
                evidence_id="evidence-3",
            ),
        ]

        grouped = group_evidence_by_citation(records)

        self.assertEqual([citation for citation, _ in grouped], [
            "Article 6",
            "Annex III point 4(a)",
        ])
        flattened_ids = [
            evidence.evidence_id
            for _, group in grouped
            for evidence in group
        ]
        self.assertCountEqual(
            flattened_ids,
            ["evidence-1", "evidence-2", "evidence-3"],
        )


if __name__ == "__main__":
    unittest.main()
