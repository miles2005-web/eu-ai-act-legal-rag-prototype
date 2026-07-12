"""Tests for opt-in legal corpus metadata-v2 enrichment."""

from __future__ import annotations

import unittest

from src.assessment.evidence import (
    AuthorityLevel,
    load_legal_source_catalog,
)
from src.corpus_enrichment import (
    CitationKind,
    enrich_chunk_metadata_v2,
    enrich_chunks_metadata_v2,
    parse_structured_citation,
)
from src.legal_chunks import build_structured_chunks


class StructuredCitationTests(unittest.TestCase):
    def test_article_citation_levels(self) -> None:
        article = parse_structured_citation("Article 2")
        paragraph = parse_structured_citation("Article 2(5)")
        point = parse_structured_citation("Article 4(1)(a)")

        self.assertEqual(article.kind, CitationKind.ARTICLE)
        self.assertEqual(article.article_number, "2")
        self.assertIsNone(article.paragraph_number)
        self.assertEqual(paragraph.paragraph_number, "5")
        self.assertEqual(paragraph.canonical_citation, "Article 2(5)")
        self.assertEqual(point.article_number, "4")
        self.assertEqual(point.paragraph_number, "1")
        self.assertEqual(point.point_label, "a")
        self.assertEqual(point.canonical_citation, "Article 4(1)(a)")

    def test_recital_and_annex_citations(self) -> None:
        recital = parse_structured_citation("recital 12")
        annex = parse_structured_citation("annex iii")

        self.assertEqual(recital.kind, CitationKind.RECITAL)
        self.assertEqual(recital.recital_ref, "12")
        self.assertEqual(recital.canonical_citation, "Recital 12")
        self.assertEqual(annex.kind, CitationKind.ANNEX)
        self.assertEqual(annex.annex_ref, "III")
        self.assertEqual(annex.canonical_citation, "Annex III")

    def test_non_atomic_or_unsupported_citation_is_rejected(self) -> None:
        for citation in (
            "Article 2(5)-(6)",
            "Article 4 point (a)",
            "Chapter II",
            "Article 4(1)(a) extra text",
        ):
            with self.subTest(citation=citation):
                with self.assertRaises(ValueError):
                    parse_structured_citation(citation)


class CorpusMetadataEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data_act = load_legal_source_catalog().get("EU_DATA_ACT")
        self.chunk = {
            "source": "EU Data Act Regulation 2023-2854.txt",
            "chunk_id": 1,
            "text": "5. ‘connected product’ means an item that obtains data.",
            "canonical_citation": "Article 2(5)",
            "article_number": "2",
            "paragraph_number": "5",
        }

    def test_enrichment_generates_complete_metadata_v2(self) -> None:
        enriched = enrich_chunk_metadata_v2(
            self.chunk,
            legal_source=self.data_act,
        )

        self.assertEqual(enriched["metadata_schema_version"], "2.0.0")
        self.assertEqual(enriched["instrument_id"], "EU_DATA_ACT")
        self.assertEqual(
            enriched["document_version"],
            "Regulation (EU) 2023/2854",
        )
        self.assertEqual(enriched["canonical_citation"], "Article 2(5)")
        self.assertEqual(
            enriched["authority_level"],
            AuthorityLevel.BINDING_LEGISLATION.value,
        )
        self.assertTrue(
            enriched["source_record_id"].startswith("legal-chunk:v2:")
        )
        self.assertTrue(
            enriched["stable_evidence_id"].startswith("evidence:v2:")
        )
        self.assertNotIn("instrument_id", self.chunk)

    def test_stable_ids_are_deterministic(self) -> None:
        first = enrich_chunk_metadata_v2(
            self.chunk,
            legal_source=self.data_act,
        )
        second = enrich_chunk_metadata_v2(
            dict(self.chunk),
            legal_source=self.data_act,
        )

        self.assertEqual(
            first["stable_evidence_id"], second["stable_evidence_id"]
        )
        self.assertEqual(first["source_record_id"], second["source_record_id"])

    def test_batch_enrichment_preserves_input_order(self) -> None:
        second_chunk = {
            **self.chunk,
            "chunk_id": 2,
            "text": "1. This Regulation lays down harmonised rules.",
            "canonical_citation": "Article 1(1)",
            "article_number": "1",
            "paragraph_number": "1",
        }
        enriched = enrich_chunks_metadata_v2(
            [self.chunk, second_chunk],
            legal_source=self.data_act,
        )

        self.assertEqual(
            [item["canonical_citation"] for item in enriched],
            ["Article 2(5)", "Article 1(1)"],
        )

    def test_legacy_chunk_builder_output_is_unchanged(self) -> None:
        text = """Article 6
Classification rules for high-risk AI systems

1. An AI system shall be considered high-risk where conditions apply.
"""

        chunks = build_structured_chunks(text, "legacy-ai-act.txt")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["canonical_citation"], "Article 6(1)")
        self.assertEqual(chunks[0]["article_number"], "6")
        self.assertEqual(chunks[0]["paragraph_number"], "1")
        self.assertNotIn("metadata_schema_version", chunks[0])
        self.assertNotIn("stable_evidence_id", chunks[0])


if __name__ == "__main__":
    unittest.main()
