"""Streamlit smoke tests for the enterprise assessment workspace."""

from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[2] / "assessment_app.py"


class AssessmentAppSmokeTests(unittest.TestCase):
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
        next(
            button
            for button in app.button
            if button.label == "Open evidence trace"
        ).click()

        app.run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["assessment_view"], "evidence")
        self.assertGreater(len(app.expander), 0)


if __name__ == "__main__":
    unittest.main()
