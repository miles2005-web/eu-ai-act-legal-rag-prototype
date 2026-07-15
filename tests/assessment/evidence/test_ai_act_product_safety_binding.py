"""Rule-level binding tests for atomic AI Act Article 6(1) Evidence."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.assessment import AssessmentFacts, TriState
from src.assessment.demo import create_assessment_workflow
from src.assessment.evidence import AuthorityLevel, CorpusMetadataV2
from src.assessment.product_regulation import load_annex_i_instrument_catalog
from src.assessment.questionnaire.definitions import AI_ACT_PRODUCT_SAFETY_RULE_ID


def _v2_record(
    instrument_id: str,
    citation: str,
    excerpt: str,
) -> dict[str, object]:
    version = {
        "EU_AI_ACT": "Regulation (EU) 2024/1689",
        "GDPR": "Regulation (EU) 2016/679",
        "EU_DATA_ACT": "Regulation (EU) 2023/2854",
    }[instrument_id]
    metadata = CorpusMetadataV2.from_excerpt(
        instrument_id=instrument_id,
        document_version=version,
        canonical_citation=citation,
        authority_level=AuthorityLevel.BINDING_LEGISLATION,
        excerpt=excerpt,
    )
    return {
        "id": metadata.source_record_id,
        "document": excerpt,
        "metadata": metadata.to_dict(),
    }


def _atomic_candidate_records() -> list[dict[str, object]]:
    records = [
        _v2_record("EU_AI_ACT", "Article 3(14)", "Official safety component definition."),
        _v2_record("EU_AI_ACT", "Article 6(1)(a)", "Official Article 6 point (a)."),
        _v2_record("EU_AI_ACT", "Article 6(1)(b)", "Official Article 6 point (b)."),
    ]
    records.extend(
        _v2_record(
            "EU_AI_ACT",
            instrument.canonical_reference,
            f"Official {instrument.canonical_reference}: {instrument.official_title_en}",
        )
        for instrument in load_annex_i_instrument_catalog().all()
    )
    # Same citation text in other frameworks must never satisfy EU_AI_ACT.
    records.extend(
        [
            _v2_record("GDPR", "Article 6(1)(a)", "Unrelated GDPR text."),
            _v2_record("EU_DATA_ACT", "Article 6(1)(a)", "Unrelated Data Act text."),
        ]
    )
    return records


class AIActProductSafetyEvidenceBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        directory = Path(self.temporary_directory.name)
        self.legacy_store = directory / "vector_store.json"
        self.candidate_store = directory / "ai-act-product-safety.json"
        self.legacy_store.write_text(
            json.dumps(
                [
                    {
                        "id": "legacy-broad-article-6",
                        "document": "Broad legacy Article 6 chunk.",
                        "metadata": {
                            "source": "EU AI Act，Regulation (EU) 2024:1689.txt",
                            "canonical_citation": "Article 6",
                            "article_number": "6",
                        },
                    }
                ]
            ),
            encoding="utf-8",
        )
        self.candidate_store.write_text(
            json.dumps(_atomic_candidate_records()),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _run(self, facts: AssessmentFacts):
        bundle = create_assessment_workflow(
            vector_store_path=self.legacy_store,
            candidate_store_paths=[self.candidate_store],
        )
        assessment_case = bundle.case_service.create_case("Product case", facts=facts)
        return bundle.workflow.run(
            assessment_case.case_id,
            rule_ids=(AI_ACT_PRODUCT_SAFETY_RULE_ID,),
        )

    @staticmethod
    def _positive_facts(*, product: TriState, safety: TriState, instrument: str) -> AssessmentFacts:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = product
        facts.product_regulation.ai_is_safety_component = safety
        facts.product_regulation.product_type = "machinery"
        facts.product_regulation.annex_i_instrument = instrument
        facts.product_regulation.annex_i_instrument_confirmed = TriState.YES
        facts.product_regulation.third_party_conformity_required = TriState.YES
        return facts

    def test_product_only_positive_binds_exactly_three_atomic_records(self) -> None:
        report = self._run(
            self._positive_facts(
                product=TriState.YES,
                safety=TriState.NO,
                instrument="ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
            )
        )

        finding = report.findings[0]
        self.assertEqual(
            [basis.citation for basis in finding.legal_basis],
            ["Article 6(1)(a)", "Article 6(1)(b)", "Annex I, Section A, point 1"],
        )
        self.assertEqual(
            [evidence.citation for evidence in report.evidence],
            ["Article 6(1)(a)", "Article 6(1)(b)", "Annex I, Section A, point 1"],
        )
        self.assertEqual(len(report.evidence_bindings[0].evidence_refs), 3)
        self.assertNotIn("Article 3(14)", [item.citation for item in report.evidence])

    def test_safety_component_positive_binds_exactly_four_records(self) -> None:
        report = self._run(
            self._positive_facts(
                product=TriState.NO,
                safety=TriState.YES,
                instrument="ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
            )
        )

        self.assertEqual(
            [evidence.citation for evidence in report.evidence],
            [
                "Article 3(14)",
                "Article 6(1)(a)",
                "Article 6(1)(b)",
                "Annex I, Section A, point 1",
            ],
        )
        self.assertEqual(len(report.evidence_bindings[0].evidence_refs), 4)

    def test_both_relation_branches_mirror_authored_product_first_basis(self) -> None:
        report = self._run(
            self._positive_facts(
                product=TriState.YES,
                safety=TriState.YES,
                instrument="ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
            )
        )
        self.assertEqual(
            [item.citation for item in report.evidence],
            [basis.citation for basis in report.findings[0].legal_basis],
        )
        self.assertNotIn("Article 3(14)", [item.citation for item in report.evidence])

    def test_selected_instrument_changes_only_annex_binding(self) -> None:
        first = self._run(
            self._positive_facts(
                product=TriState.YES,
                safety=TriState.NO,
                instrument="ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
            )
        )
        second = self._run(
            self._positive_facts(
                product=TriState.YES,
                safety=TriState.NO,
                instrument="ANNEX_I_A_02_TOY_SAFETY_DIRECTIVE_2009_48_EC",
            )
        )
        self.assertEqual(
            [item.evidence_id for item in first.evidence[:2]],
            [item.evidence_id for item in second.evidence[:2]],
        )
        self.assertNotEqual(first.evidence[2].evidence_id, second.evidence[2].evidence_id)
        self.assertEqual(second.evidence[2].citation, "Annex I, Section A, point 2")

    def test_negative_and_undetermined_bind_only_authored_references(self) -> None:
        negative = AssessmentFacts()
        negative.product_regulation.ai_is_product = TriState.NO
        negative.product_regulation.ai_is_safety_component = TriState.NO
        negative_report = self._run(negative)

        inconsistent = AssessmentFacts()
        inconsistent.product_regulation.ai_is_product = TriState.YES
        inconsistent.product_regulation.annex_i_instrument_confirmed = TriState.YES
        inconsistent.product_regulation.third_party_conformity_required = TriState.YES
        inconsistent_report = self._run(inconsistent)

        for report in (negative_report, inconsistent_report):
            self.assertEqual(
                [item.citation for item in report.evidence],
                [basis.citation for basis in report.findings[0].legal_basis],
            )

    def test_missing_information_run_has_no_finding_or_binding(self) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.NO
        report = self._run(facts)
        self.assertEqual(report.findings, [])
        self.assertEqual(report.evidence, [])
        self.assertEqual(report.evidence_bindings, [])

    def test_no_cross_framework_or_broad_legacy_evidence_leaks(self) -> None:
        report = self._run(
            self._positive_facts(
                product=TriState.YES,
                safety=TriState.NO,
                instrument="ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
            )
        )
        self.assertEqual({item.legal_source for item in report.evidence}, {"EU_AI_ACT"})
        self.assertTrue(all(item.evidence_id.startswith("evidence:v2:") for item in report.evidence))
        self.assertNotIn("Broad legacy", " ".join(item.excerpt for item in report.evidence))


if __name__ == "__main__":
    unittest.main()
