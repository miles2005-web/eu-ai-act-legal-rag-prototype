"""Smoke tests for presentation-only labels and component imports."""

from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
