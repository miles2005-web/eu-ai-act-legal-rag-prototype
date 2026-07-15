"""Clean-checkout tests for the committed AI Act product-safety pack."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.ai_act_product_safety_corpus import (
    AI_ACT_PRODUCT_SAFETY_GENERATOR,
    AI_ACT_PRODUCT_SAFETY_GENERATOR_VERSION,
    AI_ACT_PRODUCT_SAFETY_RECORD_COUNT,
    AIActProductSafetyCorpusError,
    DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
    compare_candidate_to_runtime_pack,
    load_ai_act_product_safety_runtime_pack,
)
from src.assessment import AssessmentFacts, TriState
from src.assessment.demo import create_assessment_workflow
from src.assessment.questionnaire.definitions import AI_ACT_PRODUCT_SAFETY_RULE_ID


class AIActRuntimeEvidencePackTests(unittest.TestCase):
    def test_committed_pack_is_complete_and_self_describing(self) -> None:
        records = load_ai_act_product_safety_runtime_pack()

        self.assertEqual(len(records), AI_ACT_PRODUCT_SAFETY_RECORD_COUNT)
        self.assertEqual(
            len(
                {
                    record["metadata"]["stable_evidence_id"]
                    for record in records
                }
            ),
            AI_ACT_PRODUCT_SAFETY_RECORD_COUNT,
        )
        for record in records:
            metadata = record["metadata"]
            self.assertEqual(metadata["celex"], "32024R1689")
            self.assertEqual(metadata["language"], "en")
            self.assertEqual(
                metadata["generation_tool"],
                AI_ACT_PRODUCT_SAFETY_GENERATOR,
            )
            self.assertEqual(
                metadata["generation_tool_version"],
                AI_ACT_PRODUCT_SAFETY_GENERATOR_VERSION,
            )
            self.assertEqual(
                metadata["record_provenance"]["publisher"],
                "EUR-Lex",
            )
            self.assertNotIn("embedding", record)
            self.assertNotIn("embedding", metadata)

    def test_pack_matches_manifest_without_official_source_or_build(self) -> None:
        records = load_ai_act_product_safety_runtime_pack()
        compare_candidate_to_runtime_pack(records)

    def test_pack_drift_is_rejected_explicitly(self) -> None:
        records = load_ai_act_product_safety_runtime_pack()
        drifted = deepcopy(records)
        drifted[0]["document"] += " changed"

        with self.assertRaises(AIActProductSafetyCorpusError):
            compare_candidate_to_runtime_pack(drifted)

    def test_runtime_binding_uses_committed_pack_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vector_store = Path(directory) / "vector_store.json"
            vector_store.write_text("[]", encoding="utf-8")
            bundle = create_assessment_workflow(
                vector_store_path=vector_store,
            )
            reports = []
            for name, product, safety in (
                ("Product-only", TriState.YES, TriState.NO),
                ("Safety component", TriState.NO, TriState.YES),
            ):
                facts = AssessmentFacts()
                facts.product_regulation.ai_is_product = product
                facts.product_regulation.ai_is_safety_component = safety
                facts.product_regulation.product_type = "machinery"
                facts.product_regulation.annex_i_instrument = (
                    "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
                )
                facts.product_regulation.annex_i_instrument_confirmed = (
                    TriState.YES
                )
                facts.product_regulation.third_party_conformity_required = (
                    TriState.YES
                )
                case = bundle.case_service.create_case(name, facts=facts)
                reports.append(
                    bundle.workflow.run(
                        case.case_id,
                        rule_ids=(AI_ACT_PRODUCT_SAFETY_RULE_ID,),
                    )
                )

        self.assertTrue(DEFAULT_RUNTIME_EVIDENCE_PACK_PATH.is_file())
        self.assertEqual(
            [item.citation for item in reports[0].evidence],
            [
                "Article 6(1)(a)",
                "Article 6(1)(b)",
                "Annex I, Section A, point 1",
            ],
        )
        self.assertEqual(
            [item.citation for item in reports[1].evidence],
            [
                "Article 3(14)",
                "Article 6(1)(a)",
                "Article 6(1)(b)",
                "Annex I, Section A, point 1",
            ],
        )

    def test_malformed_runtime_pack_fails_before_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            malformed = Path(directory) / "runtime-pack.json"
            malformed.write_text(json.dumps([]), encoding="utf-8")
            with self.assertRaises(AIActProductSafetyCorpusError):
                load_ai_act_product_safety_runtime_pack(malformed)


if __name__ == "__main__":
    unittest.main()
