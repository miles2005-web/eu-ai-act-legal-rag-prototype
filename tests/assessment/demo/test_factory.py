"""Tests for the reusable assessment workflow factory."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.assessment import AssessmentFacts, FindingStatus, TriState
from src.assessment.demo import create_assessment_workflow
from src.assessment.facts import UseDomain


class AssessmentWorkflowFactoryTests(unittest.TestCase):
    def test_factory_wires_complete_employment_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "vector_store.json"
            store_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "article-6",
                            "document": "Article 6 classification evidence.",
                            "metadata": {
                                "source": "EU AI Act，Regulation (EU) 2024:1689.txt",
                                "canonical_citation": "Article 6",
                                "article_number": "6",
                            },
                        },
                        {
                            "id": "annex-iii",
                            "document": "Annex III employment evidence.",
                            "metadata": {
                                "source": "AI Act Annexes I-XIII.txt",
                                "canonical_citation": "Annex III point 4(a)",
                                "annex_ref": "III",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            bundle = create_assessment_workflow(vector_store_path=store_path)
            facts = AssessmentFacts()
            facts.use_context.domain = UseDomain.EMPLOYMENT
            facts.use_context.task = "Recruitment system ranking candidates"
            facts.use_context.materially_influences_decision = TriState.YES
            assessment_case = bundle.case_service.create_case(
                "Recruitment case",
                facts=facts,
                case_id="factory-case",
            )

            report = bundle.workflow.run(assessment_case.case_id)

        self.assertEqual(
            bundle.rule_registry.ids(),
            ("AI_ACT_HIGH_RISK_EMPLOYMENT",),
        )
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(
            report.findings[0].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(len(report.evidence), 2)
        self.assertEqual(len(report.evidence_bindings), 1)

    def test_factory_rejects_non_positive_evidence_limit(self) -> None:
        with self.assertRaises(ValueError):
            create_assessment_workflow(evidence_limit=0)


if __name__ == "__main__":
    unittest.main()
