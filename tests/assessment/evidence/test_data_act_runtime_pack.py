"""Clean-checkout tests for the committed EU Data Act relevance pack."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_demo_assessment import build_assessment_facts, load_fixture
from src.assessment.demo import create_assessment_workflow
from src.assessment.evidence import MultiCorpusLegalEvidenceRetriever
from src.assessment.findings import FindingStatus
from src.assessment.questionnaire.definitions import EU_DATA_ACT_RULE_ID
from src.data_act_corpus import (
    DATA_ACT_RELEVANCE_CITATIONS,
    DATA_ACT_RELEVANCE_RECORD_COUNT,
    DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
    DataActCandidateBuildError,
    compare_data_act_candidate_to_runtime_pack,
    load_data_act_relevance_runtime_pack,
    write_data_act_relevance_runtime_pack,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INDUSTRIAL_FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "industrial_ai_case.json"
)
EXPECTED_EVIDENCE_IDS = (
    "evidence:v2:064b49b6e320dcc92f19af0c81a63537",
    "evidence:v2:2a1c882fe06379842fbf34b243cefec5",
    "evidence:v2:da1a8ad793bf79ed89c5591788e98322",
)
EXPECTED_EXCERPT_HASHES = (
    "7434c9339d04cef2a6f393a363c9a79052ce8987c20d206588b6bcd2d7fd36a6",
    "d83556429fb786b6d9da737cc2948759995cf329845b2843eb60347b99f97a8e",
    "9af2108db9d7324e57573bd85184d54a9456b14c2ac2cf9c3648a376fab93de8",
)


class DataActRuntimeEvidencePackTests(unittest.TestCase):
    def test_pack_is_exact_manifest_bound_and_embedding_free(self) -> None:
        records = load_data_act_relevance_runtime_pack()

        self.assertEqual(len(records), DATA_ACT_RELEVANCE_RECORD_COUNT)
        self.assertEqual(
            tuple(
                record["metadata"]["canonical_citation"]
                for record in records
            ),
            DATA_ACT_RELEVANCE_CITATIONS,
        )
        self.assertEqual(
            tuple(
                record["metadata"]["stable_evidence_id"]
                for record in records
            ),
            EXPECTED_EVIDENCE_IDS,
        )
        self.assertEqual(
            tuple(
                record["metadata"]["authoritative_excerpt_hash"]
                for record in records
            ),
            EXPECTED_EXCERPT_HASHES,
        )
        for record in records:
            metadata = record["metadata"]
            self.assertEqual(metadata["instrument_id"], "EU_DATA_ACT")
            self.assertEqual(metadata["framework"], "EU_DATA_ACT")
            self.assertEqual(metadata["celex"], "32023R2854")
            self.assertEqual(metadata["language"], "en")
            self.assertEqual(
                metadata["official_source_uri"],
                "https://eur-lex.europa.eu/eli/reg/2023/2854/oj/eng",
            )
            self.assertNotIn("embedding", record)
            self.assertNotIn("embedding", metadata)
            serialized = json.dumps(record, ensure_ascii=False)
            self.assertNotIn("file://", serialized)
            self.assertNotIn("/Users/", serialized)
            self.assertNotIn("/home/", serialized)

    def test_pack_matches_manifest_without_source_or_candidate_build(self) -> None:
        records = load_data_act_relevance_runtime_pack()
        compare_data_act_candidate_to_runtime_pack(records)

        with tempfile.TemporaryDirectory() as directory:
            regenerated = Path(directory) / "runtime-pack.json"
            write_data_act_relevance_runtime_pack(records, regenerated)
            self.assertEqual(
                regenerated.read_bytes(),
                DEFAULT_RUNTIME_EVIDENCE_PACK_PATH.read_bytes(),
            )

    def test_candidate_drift_is_rejected_explicitly(self) -> None:
        drifted = deepcopy(load_data_act_relevance_runtime_pack())
        drifted[0]["document"] += " changed"

        with self.assertRaisesRegex(
            DataActCandidateBuildError,
            "excerpt hash|excerpt drift|stable identity",
        ):
            compare_data_act_candidate_to_runtime_pack(drifted)

    def test_runtime_retrieval_is_exact_and_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy_store = Path(directory) / "vector_store.json"
            legacy_store.write_text("[]", encoding="utf-8")
            retriever = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                legacy_store,
                [DEFAULT_RUNTIME_EVIDENCE_PACK_PATH],
            )

            resolved = [
                retriever.retrieve("EU_DATA_ACT", citation)
                for citation in DATA_ACT_RELEVANCE_CITATIONS
            ]
            unknown = retriever.retrieve("EU_DATA_ACT", "Article 2(999)")
            malformed = retriever.retrieve("EU_DATA_ACT", "Article 2(5)-(x)")

        self.assertEqual(
            [items[0].citation for items in resolved],
            list(DATA_ACT_RELEVANCE_CITATIONS),
        )
        self.assertEqual(
            [items[0].evidence_id for items in resolved],
            list(EXPECTED_EVIDENCE_IDS),
        )
        self.assertTrue(all(len(items) == 1 for items in resolved))
        self.assertEqual(unknown, [])
        self.assertEqual(malformed, [])

    def test_exact_runtime_record_supersedes_broad_legacy_chunk(self) -> None:
        broad_record = {
            "id": "legacy-data-act-article-2",
            "document": "Broad legacy Article 2 definitions.",
            "metadata": {
                "source": "Regulation_2023_2854.html",
                "canonical_citation": "Article 2",
                "article_number": "2",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            legacy_store = Path(directory) / "vector_store.json"
            legacy_store.write_text(
                json.dumps([broad_record]),
                encoding="utf-8",
            )
            retriever = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                legacy_store,
                [DEFAULT_RUNTIME_EVIDENCE_PACK_PATH],
            )

            evidence = retriever.retrieve("EU_DATA_ACT", "Article 2(5)")

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].citation, "Article 2(5)")
        self.assertEqual(evidence[0].evidence_id, EXPECTED_EVIDENCE_IDS[1])

    def test_default_factory_uses_committed_pack_without_local_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            clean_checkout_root = Path(directory)
            self.assertFalse((clean_checkout_root / "corpus_sources").exists())
            self.assertFalse((clean_checkout_root / "corpus_builds").exists())
            vector_store = clean_checkout_root / "vector_store.json"
            vector_store.write_text("[]", encoding="utf-8")

            bundle = create_assessment_workflow(
                vector_store_path=vector_store,
            )
            evidence = bundle.evidence_retriever.retrieve(
                "EU_DATA_ACT",
                "Article 2(5)-(6)",
            )

        self.assertEqual(
            [item.evidence_id for item in evidence],
            list(EXPECTED_EVIDENCE_IDS[1:]),
        )

    def test_existing_industrial_report_binds_only_three_data_act_records(
        self,
    ) -> None:
        payload = load_fixture(INDUSTRIAL_FIXTURE_PATH)
        facts = build_assessment_facts(payload["facts"])
        bundle = create_assessment_workflow()
        assessment_case = bundle.case_service.create_case(
            payload["scenario"]["name"],
            facts=facts,
        )

        report = bundle.workflow.run(
            assessment_case.case_id,
            rule_ids=(EU_DATA_ACT_RULE_ID,),
        )

        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.findings[0].rule_id, EU_DATA_ACT_RULE_ID)
        self.assertIs(
            report.findings[0].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(len(report.evidence), 3)
        self.assertEqual(
            tuple(item.evidence_id for item in report.evidence),
            EXPECTED_EVIDENCE_IDS,
        )
        self.assertEqual(
            {item.legal_source for item in report.evidence},
            {"EU_DATA_ACT"},
        )
        self.assertEqual(report.to_dict(), deepcopy(report).to_dict())
        json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
