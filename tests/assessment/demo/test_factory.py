"""Tests for the reusable assessment workflow factory."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.assessment import AssessmentFacts, FindingStatus, TriState
from src.assessment.evidence import AuthorityLevel, CorpusMetadataV2
from src.assessment.demo import create_assessment_workflow
from src.assessment.facts import UseDomain
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.questionnaire.definitions import (
    AI_ACT_EMPLOYMENT_RULE_ID,
    AI_ACT_PRODUCT_SAFETY_RULE_ID,
    EU_DATA_ACT_RULE_ID,
    GDPR_ARTICLE22_RULE_ID,
)
from src.ui.questionnaire import execution_facts_for_modules


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
                        {
                            "id": "gdpr-article-22",
                            "document": "GDPR Article 22 evidence.",
                            "metadata": {
                                "source": "GDPR2016:679.txt",
                                "canonical_citation": "Article 22(1)",
                                "article_number": "22",
                            },
                        },
                    ]
                ),
                encoding="utf-8",
            )

            bundle = create_assessment_workflow(
                vector_store_path=store_path,
                candidate_store_paths=[],
            )
            facts = AssessmentFacts()
            facts.use_context.domain = UseDomain.EMPLOYMENT
            facts.use_context.task = "Recruitment system ranking candidates"
            facts.use_context.materially_influences_decision = TriState.YES
            facts.data_protection.personal_data_processed = TriState.YES
            facts.data_protection.automated_individual_decision = TriState.YES
            assessment_case = bundle.case_service.create_case(
                "Recruitment case",
                facts=facts,
                case_id="factory-case",
            )

            report = bundle.workflow.run(assessment_case.case_id)

        self.assertEqual(
            bundle.rule_registry.ids(),
            (
                "AI_ACT_HIGH_RISK_EMPLOYMENT",
                "AI_ACT_HIGH_RISK_PRODUCT_SAFETY",
                "GDPR_ARTICLE22_RELEVANCE",
                "EU_DATA_ACT_RELEVANCE",
            ),
        )
        self.assertEqual(len(report.findings), 2)
        self.assertEqual(
            report.authorized_rule_ids,
            list(bundle.rule_registry.ids()),
        )
        self.assertEqual(
            report.findings[0].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(
            [finding.rule_id for finding in report.findings],
            [
                "AI_ACT_HIGH_RISK_EMPLOYMENT",
                "GDPR_ARTICLE22_RELEVANCE",
            ],
        )
        self.assertEqual(len(report.evidence), 3)
        self.assertEqual(len(report.evidence_bindings), 2)
        self.assertNotIn(
            "EU_DATA_ACT",
            {evidence.legal_source for evidence in report.evidence},
        )
        evidence_by_id = {
            evidence.evidence_id: evidence for evidence in report.evidence
        }
        findings_by_id = {
            finding.finding_id: finding for finding in report.findings
        }
        for binding in report.evidence_bindings:
            expected_sources = {
                basis.instrument
                for basis in findings_by_id[binding.finding_id].legal_basis
            }
            actual_sources = {
                evidence_by_id[evidence_id].legal_source
                for evidence_id in binding.evidence_refs
            }
            self.assertTrue(actual_sources.issubset(expected_sources))

    def test_factory_wires_data_act_candidate_corpus(self) -> None:
        data_act_metadata = CorpusMetadataV2.from_excerpt(
            instrument_id="EU_DATA_ACT",
            document_version="Regulation (EU) 2023/2854",
            canonical_citation="Article 2(5)",
            authority_level=AuthorityLevel.BINDING_LEGISLATION,
            excerpt="Connected product definition.",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            legacy_store = directory / "legacy.json"
            candidate_store = directory / "data-act.json"
            legacy_store.write_text("[]", encoding="utf-8")
            candidate_store.write_text(
                json.dumps(
                    [
                        {
                            "id": data_act_metadata.source_record_id,
                            "document": "Connected product definition.",
                            "metadata": {
                                **data_act_metadata.to_dict(),
                                "article_number": "2",
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )

            bundle = create_assessment_workflow(
                vector_store_path=legacy_store,
                candidate_store_paths=[candidate_store],
            )
            facts = AssessmentFacts()
            facts.data_act.connected_product = TriState.YES
            facts.data_act.related_service = TriState.YES
            facts.data_act.data_generated = TriState.YES
            assessment_case = bundle.case_service.create_case(
                "Industrial case",
                facts=facts,
            )
            report = bundle.workflow.run(
                assessment_case.case_id,
                rule_ids=(EU_DATA_ACT_RULE_ID,),
            )

        data_act_findings = [
            finding
            for finding in report.findings
            if finding.rule_id == "EU_DATA_ACT_RELEVANCE"
        ]
        self.assertEqual(len(data_act_findings), 1)
        self.assertEqual(
            data_act_findings[0].framework,
            RegulatoryFramework.EU_DATA_ACT,
        )
        self.assertIn("CONNECTED_PRODUCT", data_act_findings[0].reason_codes)
        self.assertIn("RELATED_SERVICE", data_act_findings[0].reason_codes)
        self.assertIn("DATA_GENERATED", data_act_findings[0].reason_codes)
        binding = next(
            item
            for item in report.evidence_bindings
            if item.finding_id == data_act_findings[0].finding_id
        )
        bound_sources = {
            evidence.legal_source
            for evidence in report.evidence
            if evidence.evidence_id in binding.evidence_refs
        }
        self.assertEqual(bound_sources, {"EU_DATA_ACT"})

    def test_confirmed_product_safety_module_runs_without_evidence_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "vector_store.json"
            store_path.write_text("[]", encoding="utf-8")
            bundle = create_assessment_workflow(
                vector_store_path=store_path,
                candidate_store_paths=[],
            )
            facts = AssessmentFacts()
            facts.product_regulation.ai_is_product = TriState.YES
            facts.product_regulation.product_type = "machinery"
            facts.product_regulation.annex_i_instrument = (
                "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
            )
            facts.product_regulation.annex_i_instrument_confirmed = TriState.YES
            facts.product_regulation.third_party_conformity_required = TriState.YES
            execution = execution_facts_for_modules(
                facts,
                [AI_ACT_PRODUCT_SAFETY_RULE_ID],
            )
            assessment_case = bundle.case_service.create_case(
                "Product-safety case",
                facts=execution,
            )

            report = bundle.workflow.run(
                assessment_case.case_id,
                rule_ids=(AI_ACT_PRODUCT_SAFETY_RULE_ID,),
            )

        product_findings = [
            finding
            for finding in report.findings
            if finding.rule_id == AI_ACT_PRODUCT_SAFETY_RULE_ID
        ]
        self.assertEqual(len(product_findings), 1)
        self.assertEqual(
            product_findings[0].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(report.evidence, [])
        self.assertEqual(report.evidence_bindings, [])
        self.assertEqual(
            report.authorized_rule_ids,
            [AI_ACT_PRODUCT_SAFETY_RULE_ID],
        )
        self.assertEqual(report.missing_information, [])

    def test_incomplete_product_scope_has_only_authorized_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "vector_store.json"
            store_path.write_text("[]", encoding="utf-8")
            bundle = create_assessment_workflow(vector_store_path=store_path)
            facts = AssessmentFacts()
            facts.product_regulation.ai_is_product = TriState.YES
            facts.product_regulation.ai_is_safety_component = TriState.NO
            facts.product_regulation.product_type = "machinery"
            assessment_case = bundle.case_service.create_case(
                "Incomplete product-safety case",
                facts=facts,
            )

            report = bundle.workflow.run(
                assessment_case.case_id,
                rule_ids=(AI_ACT_PRODUCT_SAFETY_RULE_ID,),
            )
            run = bundle.workflow.get_run(report.assessment_run_reference)

        self.assertEqual(report.findings, [])
        self.assertEqual(
            report.authorized_rule_ids,
            [AI_ACT_PRODUCT_SAFETY_RULE_ID],
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
            report.assessed_frameworks,
            [RegulatoryFramework.EU_AI_ACT],
        )
        self.assertEqual(
            [item.rule_id for item in report.rule_versions],
            [AI_ACT_PRODUCT_SAFETY_RULE_ID],
        )
        self.assertEqual(run.authorized_rule_ids, [AI_ACT_PRODUCT_SAFETY_RULE_ID])
        self.assertTrue(run.input_fingerprint)

    def test_unconfirmed_product_module_cannot_emit_finding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "vector_store.json"
            store_path.write_text("[]", encoding="utf-8")
            bundle = create_assessment_workflow(vector_store_path=store_path)
            facts = AssessmentFacts()
            facts.product_regulation.ai_is_product = TriState.YES
            facts.product_regulation.annex_i_instrument = (
                "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
            )
            facts.product_regulation.annex_i_instrument_confirmed = TriState.YES
            facts.product_regulation.third_party_conformity_required = TriState.YES
            execution = execution_facts_for_modules(facts, [])
            assessment_case = bundle.case_service.create_case(
                "Unconfirmed product-safety case",
                facts=execution,
            )

            report = bundle.workflow.run(
                assessment_case.case_id,
                rule_ids=(),
            )

        self.assertNotIn(
            AI_ACT_PRODUCT_SAFETY_RULE_ID,
            [finding.rule_id for finding in report.findings],
        )
        self.assertEqual(report.authorized_rule_ids, [])
        self.assertEqual(report.missing_information, [])

    def test_confirmed_product_safety_negative_and_undetermined_outcomes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "vector_store.json"
            store_path.write_text("[]", encoding="utf-8")
            bundle = create_assessment_workflow(vector_store_path=store_path)

            negative = AssessmentFacts()
            negative.product_regulation.ai_is_product = TriState.NO
            negative.product_regulation.ai_is_safety_component = TriState.NO
            negative_case = bundle.case_service.create_case(
                "Negative product-safety case",
                facts=execution_facts_for_modules(
                    negative,
                    [AI_ACT_PRODUCT_SAFETY_RULE_ID],
                ),
            )
            negative_report = bundle.workflow.run(
                negative_case.case_id,
                rule_ids=(AI_ACT_PRODUCT_SAFETY_RULE_ID,),
            )

            inconsistent = AssessmentFacts()
            inconsistent.product_regulation.ai_is_product = TriState.YES
            inconsistent.product_regulation.annex_i_instrument_confirmed = (
                TriState.YES
            )
            inconsistent.product_regulation.third_party_conformity_required = (
                TriState.YES
            )
            inconsistent_case = bundle.case_service.create_case(
                "Inconsistent product-safety case",
                facts=execution_facts_for_modules(
                    inconsistent,
                    [AI_ACT_PRODUCT_SAFETY_RULE_ID],
                ),
            )
            inconsistent_report = bundle.workflow.run(
                inconsistent_case.case_id,
                rule_ids=(AI_ACT_PRODUCT_SAFETY_RULE_ID,),
            )

        negative_finding = next(
            finding
            for finding in negative_report.findings
            if finding.rule_id == AI_ACT_PRODUCT_SAFETY_RULE_ID
        )
        self.assertEqual(negative_finding.status, FindingStatus.DOES_NOT_APPLY)
        self.assertIn("Article 6(1)", negative_finding.title)
        inconsistent_finding = next(
            finding
            for finding in inconsistent_report.findings
            if finding.rule_id == AI_ACT_PRODUCT_SAFETY_RULE_ID
        )
        self.assertEqual(
            inconsistent_finding.status,
            FindingStatus.UNDETERMINED,
        )
        self.assertEqual(
            negative_report.authorized_rule_ids,
            [AI_ACT_PRODUCT_SAFETY_RULE_ID],
        )
        self.assertEqual(negative_report.missing_information, [])

    def test_two_authorized_modules_execute_in_registry_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "vector_store.json"
            store_path.write_text("[]", encoding="utf-8")
            bundle = create_assessment_workflow(vector_store_path=store_path)
            facts = AssessmentFacts()
            facts.use_context.domain = UseDomain.EMPLOYMENT
            facts.use_context.task = "Recruitment system ranking candidates"
            facts.use_context.materially_influences_decision = TriState.YES
            facts.data_protection.personal_data_processed = TriState.YES
            facts.data_protection.automated_individual_decision = TriState.YES
            assessment_case = bundle.case_service.create_case(
                "Two-module recruitment case",
                facts=facts,
            )

            report = bundle.workflow.run(
                assessment_case.case_id,
                rule_ids=(GDPR_ARTICLE22_RULE_ID, AI_ACT_EMPLOYMENT_RULE_ID),
            )

        self.assertEqual(
            report.authorized_rule_ids,
            [AI_ACT_EMPLOYMENT_RULE_ID, GDPR_ARTICLE22_RULE_ID],
        )
        self.assertEqual(
            [finding.rule_id for finding in report.findings],
            [AI_ACT_EMPLOYMENT_RULE_ID, GDPR_ARTICLE22_RULE_ID],
        )
        self.assertEqual(report.missing_information, [])
        self.assertEqual(
            report.assessed_frameworks,
            [RegulatoryFramework.EU_AI_ACT, RegulatoryFramework.GDPR],
        )

    def test_factory_rejects_non_positive_evidence_limit(self) -> None:
        with self.assertRaises(ValueError):
            create_assessment_workflow(evidence_limit=0)

    def test_factory_rejects_single_candidate_path_string(self) -> None:
        with self.assertRaises(TypeError):
            create_assessment_workflow(candidate_store_paths="candidate.json")


if __name__ == "__main__":
    unittest.main()
