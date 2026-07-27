"""Tests for metadata-aware conversion from corpus records to Evidence."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.assessment.evidence import (
    AuthorityLevel,
    CorpusMetadataV2,
    VectorStoreJSONEvidenceRetriever,
)


class MetadataAwareEvidenceConversionTests(unittest.TestCase):
    @staticmethod
    def _v2_record(
        *,
        excerpt: str = "GDPR Article 22 v2 excerpt.",
        citation: str = "Article 22(1)",
    ) -> dict[str, object]:
        metadata = CorpusMetadataV2.from_excerpt(
            instrument_id="GDPR",
            document_version="Regulation (EU) 2016/679",
            canonical_citation=citation,
            authority_level=AuthorityLevel.BINDING_LEGISLATION,
            excerpt=excerpt,
        )
        return {
            "id": metadata.source_record_id,
            "document": excerpt,
            "metadata": {
                **metadata.to_dict(),
                "source": "renamed-gdpr-source.txt",
                "article_number": "22",
            },
        }

    @staticmethod
    def _legacy_record() -> dict[str, object]:
        return {
            "id": "legacy-gdpr-22",
            "document": "GDPR Article 22 legacy excerpt.",
            "metadata": {
                "source": "GDPR2016:679.txt",
                "canonical_citation": "Article 22(1)",
                "article_number": "22",
            },
        }

    @staticmethod
    def _retrieve(records: list[dict[str, object]]) -> list:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "vector_store.json"
            store_path.write_text(json.dumps(records), encoding="utf-8")
            return VectorStoreJSONEvidenceRetriever(store_path).retrieve(
                "GDPR",
                "Article 22(1)",
                limit=10,
            )

    def test_v2_metadata_controls_evidence_conversion(self) -> None:
        record = self._v2_record()
        evidence = self._retrieve([record])
        metadata = record["metadata"]

        self.assertEqual(len(evidence), 1)
        self.assertEqual(
            evidence[0].evidence_id,
            metadata["stable_evidence_id"],
        )
        self.assertEqual(evidence[0].legal_source, metadata["instrument_id"])
        self.assertEqual(
            evidence[0].document_version,
            metadata["document_version"],
        )
        self.assertEqual(
            evidence[0].citation,
            metadata["canonical_citation"],
        )
        self.assertEqual(
            evidence[0].authority_level,
            AuthorityLevel.BINDING_LEGISLATION,
        )

    def test_legacy_record_conversion_preserves_legacy_id(self) -> None:
        first = self._retrieve([self._legacy_record()])
        second = self._retrieve([self._legacy_record()])

        self.assertEqual(len(first), 1)
        self.assertEqual(
            first[0].evidence_id,
            "vector-store:7c234dfc2b5f53e0758a066d",
        )
        self.assertEqual(first[0].evidence_id, second[0].evidence_id)
        self.assertEqual(first[0].legal_source, "GDPR")
        self.assertEqual(first[0].citation, "Article 22(1)")

    def test_mixed_v2_and_legacy_records_convert_in_store_order(self) -> None:
        v2_record = self._v2_record()
        evidence = self._retrieve([v2_record, self._legacy_record()])

        self.assertEqual(len(evidence), 2)
        self.assertEqual(
            evidence[0].evidence_id,
            v2_record["metadata"]["stable_evidence_id"],
        )
        self.assertTrue(evidence[1].evidence_id.startswith("vector-store:"))
        self.assertEqual(
            [item.excerpt for item in evidence],
            [
                "GDPR Article 22 v2 excerpt.",
                "GDPR Article 22 legacy excerpt.",
            ],
        )

    def test_v2_evidence_id_is_deterministic(self) -> None:
        record = self._v2_record()

        first = self._retrieve([record])
        second = self._retrieve([record])

        self.assertEqual(first[0].evidence_id, second[0].evidence_id)
        self.assertEqual(
            first[0].evidence_id,
            record["metadata"]["stable_evidence_id"],
        )


if __name__ == "__main__":
    unittest.main()
