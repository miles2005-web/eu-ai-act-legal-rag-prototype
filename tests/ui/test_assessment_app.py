"""Streamlit smoke tests for the enterprise assessment workspace."""

from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from src.assessment.findings import FindingStatus
from src.ui.styles import ENTERPRISE_STYLES


APP_PATH = Path(__file__).resolve().parents[2] / "assessment_app.py"


class AssessmentAppSmokeTests(unittest.TestCase):
    def test_sidebar_uses_lightweight_workspace_navigation(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

        labels = [button.label for button in app.button]
        self.assertIn("Assessment", labels)
        self.assertIn("Demo cases", labels)
        self.assertIn("Findings", labels)
        self.assertIn("Evidence trace", labels)
        self.assertNotIn("Evidence engine", labels)
        self.assertIn(
            '[data-testid="stSidebar"] .stButton > button',
            ENTERPRISE_STYLES,
        )
        self.assertIn("background: transparent", ENTERPRISE_STYLES)
        self.assertIn("border: 0", ENTERPRISE_STYLES)

    def test_landing_renders_both_demo_entry_points(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        button_labels = [button.label for button in app.button]
        self.assertIn("Open recruitment demo", button_labels)
        self.assertIn("Open industrial demo", button_labels)

    def test_industrial_demo_opens_assessment_workspace(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        next(
            button
            for button in app.button
            if button.label == "Open industrial demo"
        ).click()

        app.run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["assessment_view"], "workspace")
        self.assertEqual(app.session_state["demo_scenario"], "industrial")
        self.assertIn("Run assessment", [button.label for button in app.button])
        workspace_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("ui-system-summary", workspace_markup)
        self.assertIn("ForgeMonitor AI", workspace_markup)
        self.assertIn("Technical details", [item.label for item in app.expander])

    def test_recruitment_flow_opens_results_and_evidence_trace(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        next(
            button
            for button in app.button
            if button.label == "Open recruitment demo"
        ).click()
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "Run assessment"
        ).click()

        app.run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["assessment_view"], "results")
        report = app.session_state["assessment_report"]
        self.assertEqual(report.findings[0].status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertEqual(report.findings[0].rule_id, "AI_ACT_HIGH_RISK_EMPLOYMENT")
        result_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("ui-finding-title--hero", result_markup)
        self.assertIn("Decision path", result_markup)
        self.assertIn("Employment context", result_markup)
        self.assertIn("Material influence on decisions", result_markup)
        self.assertIn("Evidence summary", result_markup)
        self.assertIn("Technical details", [item.label for item in app.expander])
        next(
            button
            for button in app.button
            if button.label == "Inspect this finding in Evidence trace"
        ).click()

        app.run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["assessment_view"], "evidence")
        trace_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("Facts", trace_markup)
        self.assertIn("Rule evaluation", trace_markup)
        self.assertIn("Legal basis", trace_markup)
        self.assertIn("Source evidence", trace_markup)
        self.assertIn("Employment context", trace_markup)
        self.assertIn("Material influence on decisions", trace_markup)
        self.assertIn("supporting excerpt(s)", trace_markup)
        code_values = [code.value for code in app.code]
        expected_evidence_ids = {
            evidence.evidence_id for evidence in report.evidence
        }
        self.assertTrue(expected_evidence_ids.issubset(set(code_values)))


if __name__ == "__main__":
    unittest.main()
