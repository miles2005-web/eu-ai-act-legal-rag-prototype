"""Tests for atomic evidence resolution from authored citation ranges."""

from __future__ import annotations

import unittest

from src.assessment import (
    AssessmentCaseService,
    AssessmentEngine,
    AssessmentFacts,
    AssessmentWorkflowService,
    AuthorityLevel,
    Evidence,
    FindingStatus,
    InMemoryEvidenceService,
    ReportBuilder,
    TriState,
)
from src.assessment.evidence import (
    expand_citation_reference,
    is_strict_atomic_citation,
    normalize_atomic_citation,
)
from src.assessment.rules import EUDataActRelevanceRule, RuleRegistry


class CitationRangeExpansionTests(unittest.TestCase):
    def test_article_paragraph_range_expands_to_atomic_citations(self) -> None:
        self.assertEqual(
            expand_citation_reference("Article 2(5)-(6)"),
            ("Article 2(5)", "Article 2(6)"),
        )

    def test_non_range_ai_act_and_gdpr_citations_are_unchanged(self) -> None:
        self.assertEqual(
            expand_citation_reference("Article 6"),
            ("Article 6",),
        )
        self.assertEqual(
            expand_citation_reference("Article 22(1)"),
            ("Article 22(1)",),
        )

    def test_atomic_ai_act_formatting_variants_normalize_exactly(self) -> None:
        self.assertEqual(
            normalize_atomic_citation("Art. 6 ( 1 ) ( A )"),
            "Article 6(1)(a)",
        )
        self.assertEqual(
            normalize_atomic_citation("annex i section a point 1"),
            "Annex I, Section A, point 1",
        )
        self.assertTrue(is_strict_atomic_citation("Article 3(14)"))
        self.assertTrue(
            is_strict_atomic_citation("Annex I, Section B, point 20")
        )

    def test_malformed_references_are_not_fuzzily_rewritten(self) -> None:
        malformed = "Annex I, Section A, point nearby 1"
        self.assertEqual(normalize_atomic_citation(malformed), malformed)
        self.assertFalse(is_strict_atomic_citation(malformed))


class DataActRangeBindingTests(unittest.TestCase):
    @staticmethod
    def _evidence() -> list[Evidence]:
        return [
            Evidence(
                evidence_id="data-act-article-1-1-a",
                legal_source="EU_DATA_ACT",
                citation="Article 1(1)(a)",
                excerpt="Data Act scope excerpt.",
                authority_level=AuthorityLevel.BINDING_LEGISLATION,
                document_version="Regulation (EU) 2023/2854",
            ),
            Evidence(
                evidence_id="data-act-article-2-5",
                legal_source="EU_DATA_ACT",
                citation="Article 2(5)",
                excerpt="Connected product definition.",
                authority_level=AuthorityLevel.BINDING_LEGISLATION,
                document_version="Regulation (EU) 2023/2854",
            ),
            Evidence(
                evidence_id="data-act-article-2-6",
                legal_source="EU_DATA_ACT",
                citation="Article 2(6)",
                excerpt="Related service definition.",
                authority_level=AuthorityLevel.BINDING_LEGISLATION,
                document_version="Regulation (EU) 2023/2854",
            ),
        ]

    def test_data_act_range_evidence_is_bound_in_report(self) -> None:
        facts = AssessmentFacts()
        facts.data_act.connected_product = TriState.YES
        facts.data_act.related_service = TriState.YES
        facts.data_act.data_generated = TriState.YES
        case_service = AssessmentCaseService()
        case_service.create_case(
            "Industrial AI Data Act case",
            facts=facts,
            case_id="data-act-range-case",
        )
        workflow = AssessmentWorkflowService(
            case_service=case_service,
            assessment_engine=AssessmentEngine(
                RuleRegistry([EUDataActRelevanceRule()])
            ),
            evidence_service=InMemoryEvidenceService(self._evidence()),
            report_builder=ReportBuilder(),
        )

        report = workflow.run("data-act-range-case")

        self.assertEqual(len(report.findings), 1)
        self.assertEqual(
            report.findings[0].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(
            [basis.citation for basis in report.findings[0].legal_basis],
            ["Article 1(1)(a)", "Article 2(5)-(6)"],
        )
        self.assertEqual(
            [evidence.citation for evidence in report.evidence],
            ["Article 1(1)(a)", "Article 2(5)", "Article 2(6)"],
        )
        self.assertEqual(
            report.evidence_bindings[0].evidence_refs,
            [
                "data-act-article-1-1-a",
                "data-act-article-2-5",
                "data-act-article-2-6",
            ],
        )


if __name__ == "__main__":
    unittest.main()
