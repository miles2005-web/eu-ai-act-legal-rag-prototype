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

    def test_potentially_applies_uses_review_tone(self) -> None:
        self.assertEqual(
            status_label(FindingStatus.POTENTIALLY_APPLIES),
            "Potentially Applies",
        )
        self.assertEqual(
            status_tone(FindingStatus.POTENTIALLY_APPLIES),
            "warning",
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
