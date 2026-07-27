"""Tests for versioned corpus metadata and stable evidence identity."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.assessment.evidence import (
    AuthorityLevel,
    CorpusMetadataV2,
    VectorStoreJSONEvidenceRetriever,
    stable_evidence_id,
)


class CorpusMetadataV2Tests(unittest.TestCase):
    @staticmethod
    def _metadata(
        *,
        citation: str = "Article 22(1)",
        excerpt: str = "A decision based solely on automated processing.",
    ) -> CorpusMetadataV2:
        return CorpusMetadataV2.from_excerpt(
            instrument_id="GDPR",
            document_version="Regulation (EU) 2016/679",
            canonical_citation=citation,
            authority_level=AuthorityLevel.BINDING_LEGISLATION,
            excerpt=excerpt,
        )

    def test_stable_id_generation_and_serialization(self) -> None:
        metadata = self._metadata()
        payload = metadata.to_dict()

        self.assertTrue(
            metadata.source_record_id.startswith("legal-chunk:v2:")
        )
        self.assertTrue(
            metadata.stable_evidence_id.startswith("evidence:v2:")
        )
        self.assertEqual(payload["metadata_schema_version"], "2.0.0")
        self.assertEqual(payload["instrument_id"], "GDPR")
        self.assertEqual(payload["authority_level"], "binding_legislation")
        json.dumps(payload)

    def test_same_normalized_input_produces_same_id(self) -> None:
        first = self._metadata(
            excerpt="A decision based solely\r\n on automated processing."
        )
        second = self._metadata(
            excerpt="A decision based solely on   automated processing."
        )

        self.assertEqual(first.stable_evidence_id, second.stable_evidence_id)
        self.assertEqual(first.source_record_id, second.source_record_id)
        self.assertEqual(
            first.stable_evidence_id,
            stable_evidence_id(
                instrument_id="GDPR",
                document_version="Regulation (EU) 2016/679",
                canonical_citation="Article 22(1)",
                excerpt="A decision based solely on automated processing.",
            ),
        )

    def test_different_citation_produces_different_id(self) -> None:
        article_22 = self._metadata(citation="Article 22(1)")
        article_21 = self._metadata(citation="Article 21(1)")

        self.assertNotEqual(
            article_22.stable_evidence_id,
            article_21.stable_evidence_id,
        )
        self.assertNotEqual(
            article_22.source_record_id,
            article_21.source_record_id,
        )

    def test_legacy_record_remains_readable(self) -> None:
        legacy_records = [
            {
                "id": "chunk_legacy_22",
                "document": "Legacy GDPR Article 22 excerpt.",
                "metadata": {
                    "source": "GDPR2016:679.txt",
                    "canonical_citation": "Article 22(1)",
                    "article_number": "22",
                    "annex_ref": "None",
                    "recital_ref": "None",
                },
                "embedding": [0.1, 0.2],
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "vector_store.json"
            store_path.write_text(
                json.dumps(legacy_records),
                encoding="utf-8",
            )

            evidence = VectorStoreJSONEvidenceRetriever(store_path).retrieve(
                "GDPR",
                "Article 22(1)",
            )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].excerpt, "Legacy GDPR Article 22 excerpt.")
        self.assertTrue(
            evidence[0].evidence_id.startswith("vector-store:")
        )


if __name__ == "__main__":
    unittest.main()
