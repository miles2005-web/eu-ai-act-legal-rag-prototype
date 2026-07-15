"""Tests for separate metadata-v2 candidate corpus retrieval."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.assessment.evidence import (
    AuthorityLevel,
    CorpusMetadataV2,
    MultiCorpusLegalEvidenceRetriever,
    VectorStoreJSONEvidenceRetriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _v2_record(
    *,
    instrument_id: str,
    version: str,
    citation: str,
    excerpt: str,
    article_number: str,
) -> dict[str, object]:
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
        "metadata": {
            **metadata.to_dict(),
            "article_number": article_number,
        },
    }


class MultiCorpusEvidenceRetrieverTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, records: list[dict[str, object]]) -> None:
        path.write_text(json.dumps(records), encoding="utf-8")

    def test_data_act_citation_returns_metadata_v2_evidence(self) -> None:
        data_act_record = _v2_record(
            instrument_id="EU_DATA_ACT",
            version="Regulation (EU) 2023/2854",
            citation="Article 2(5)",
            excerpt="‘connected product’ means an item that generates data.",
            article_number="2",
        )
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            candidate = Path(directory) / "data-act-candidate.json"
            self._write(existing, [])
            self._write(candidate, [data_act_record])
            retriever = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                existing,
                [candidate],
            )

            evidence = retriever.retrieve(
                "EU_DATA_ACT",
                "Article 2(5)",
            )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].legal_source, "EU_DATA_ACT")
        self.assertEqual(evidence[0].citation, "Article 2(5)")
        self.assertEqual(
            evidence[0].document_version,
            "Regulation (EU) 2023/2854",
        )
        self.assertEqual(
            evidence[0].authority_level,
            AuthorityLevel.BINDING_LEGISLATION,
        )
        self.assertEqual(
            evidence[0].evidence_id,
            data_act_record["metadata"]["stable_evidence_id"],
        )

    def test_data_act_range_retrieves_atomic_citations_in_order(self) -> None:
        records = [
            _v2_record(
                instrument_id="EU_DATA_ACT",
                version="Regulation (EU) 2023/2854",
                citation=citation,
                excerpt=excerpt,
                article_number="2",
            )
            for citation, excerpt in (
                ("Article 2(5)", "Connected product definition."),
                ("Article 2(6)", "Related service definition."),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            candidate = Path(directory) / "data-act-candidate.json"
            self._write(existing, [])
            self._write(candidate, records)
            retriever = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                existing,
                [candidate],
            )

            evidence = retriever.retrieve(
                "EU_DATA_ACT",
                "Article 2(5)-(6)",
            )

        self.assertEqual(
            [item.citation for item in evidence],
            ["Article 2(5)", "Article 2(6)"],
        )
        self.assertEqual(
            [item.evidence_id for item in evidence],
            [
                records[0]["metadata"]["stable_evidence_id"],
                records[1]["metadata"]["stable_evidence_id"],
            ],
        )

    def test_existing_ai_act_retrieval_is_unchanged(self) -> None:
        direct = VectorStoreJSONEvidenceRetriever(
            PROJECT_ROOT / "vector_store.json"
        ).retrieve("EU_AI_ACT", "Article 6", limit=2)

        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "data-act-candidate.json"
            self._write(candidate, [])
            combined = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                PROJECT_ROOT / "vector_store.json",
                [candidate],
            ).retrieve("EU_AI_ACT", "Article 6", limit=2)

        self.assertEqual(
            [item.to_dict() for item in combined],
            [item.to_dict() for item in direct],
        )

    def test_legacy_gdpr_fallback_remains_available(self) -> None:
        gdpr_record = {
            "id": "legacy-gdpr-22",
            "document": "Legacy GDPR Article 22 text.",
            "metadata": {
                "source": "GDPR2016:679.txt",
                "canonical_citation": "Article 22(1)",
                "article_number": "22",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            candidate = Path(directory) / "data-act-candidate.json"
            self._write(existing, [gdpr_record])
            self._write(candidate, [])
            retriever = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                existing,
                [candidate],
            )

            evidence = retriever.retrieve("GDPR", "Article 22(1)")

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].legal_source, "GDPR")
        self.assertTrue(evidence[0].evidence_id.startswith("vector-store:"))

    def test_mixed_framework_retrieval_follows_reference_order(self) -> None:
        ai_act = _v2_record(
            instrument_id="EU_AI_ACT",
            version="Regulation (EU) 2024/1689",
            citation="Article 6",
            excerpt="AI Act classification provision.",
            article_number="6",
        )
        gdpr = _v2_record(
            instrument_id="GDPR",
            version="Regulation (EU) 2016/679",
            citation="Article 22(1)",
            excerpt="GDPR automated decision provision.",
            article_number="22",
        )
        data_act = _v2_record(
            instrument_id="EU_DATA_ACT",
            version="Regulation (EU) 2023/2854",
            citation="Article 2(5)",
            excerpt="Data Act connected product definition.",
            article_number="2",
        )
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            candidate = Path(directory) / "data-act-candidate.json"
            self._write(existing, [ai_act, gdpr])
            self._write(candidate, [data_act])
            retriever = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                existing,
                [candidate],
            )

            evidence = retriever.retrieve_many(
                [
                    ("EU_DATA_ACT", "Article 2(5)"),
                    ("EU_AI_ACT", "Article 6"),
                    ("GDPR", "Article 22(1)"),
                ]
            )

        self.assertEqual(
            [item.legal_source for item in evidence],
            ["EU_DATA_ACT", "EU_AI_ACT", "GDPR"],
        )

    def test_citations_never_resolve_from_another_instrument(self) -> None:
        ai_act = _v2_record(
            instrument_id="EU_AI_ACT",
            version="Regulation (EU) 2024/1689",
            citation="Article 6",
            excerpt="AI Act Article 6.",
            article_number="6",
        )
        data_act = _v2_record(
            instrument_id="EU_DATA_ACT",
            version="Regulation (EU) 2023/2854",
            citation="Article 2(5)",
            excerpt="Data Act Article 2 definition.",
            article_number="2",
        )
        gdpr = _v2_record(
            instrument_id="GDPR",
            version="Regulation (EU) 2016/679",
            citation="Article 22(1)",
            excerpt="GDPR Article 22.",
            article_number="22",
        )
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            candidate = Path(directory) / "data-act-candidate.json"
            self._write(existing, [ai_act, gdpr])
            self._write(candidate, [data_act])
            retriever = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                existing,
                [candidate],
            )

            self.assertEqual(
                retriever.retrieve("EU_AI_ACT", "Article 2(5)"),
                [],
            )
            self.assertEqual(
                retriever.retrieve("EU_DATA_ACT", "Article 6"),
                [],
            )
            self.assertEqual(
                retriever.retrieve("GDPR", "Article 6"),
                [],
            )

    def test_atomic_ai_act_record_precedes_broad_legacy_and_fails_closed(self) -> None:
        atomic = _v2_record(
            instrument_id="EU_AI_ACT",
            version="Regulation (EU) 2024/1689",
            citation="Annex I, Section A, point 1",
            excerpt="Official atomic machinery entry.",
            article_number="",
        )
        atomic["metadata"].update(
            {"annex_ref": "I", "annex_section": "A", "annex_point": 1}
        )
        legacy = {
            "id": "legacy-annex-i",
            "document": "Broad legacy Annex I text.",
            "metadata": {
                "source": "AI Act Annexes I-XIII.txt",
                "canonical_citation": "Annex I",
                "annex_ref": "I",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            candidate = Path(directory) / "ai-act-candidate.json"
            self._write(existing, [legacy])
            self._write(candidate, [atomic])
            retriever = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                existing,
                [candidate],
            )

            exact = retriever.retrieve(
                "EU_AI_ACT",
                "annex i section a point 1",
            )
            unknown = retriever.retrieve(
                "EU_AI_ACT",
                "Annex I, Section A, point 21",
            )
            malformed = retriever.retrieve(
                "EU_AI_ACT",
                "Annex I, Section A, point nearby 1",
            )

        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].evidence_id, atomic["metadata"]["stable_evidence_id"])
        self.assertEqual(unknown, [])
        self.assertEqual(malformed, [])


if __name__ == "__main__":
    unittest.main()
