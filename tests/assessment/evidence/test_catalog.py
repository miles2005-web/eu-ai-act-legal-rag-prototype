"""Tests for legal source catalog and legacy corpus compatibility."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.assessment.evidence import (
    AuthorityLevel,
    VectorStoreJSONEvidenceRetriever,
    load_legal_source_catalog,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class LegalSourceCatalogTests(unittest.TestCase):
    def test_default_catalog_loads_initial_instruments(self) -> None:
        catalog = load_legal_source_catalog()

        self.assertEqual(catalog.schema_version, "1.0.0")
        self.assertEqual(
            [source.instrument_id for source in catalog.all()],
            ["EU_AI_ACT", "GDPR", "EU_DATA_ACT"],
        )
        gdpr = catalog.get("GDPR")
        self.assertEqual(gdpr.title, "General Data Protection Regulation")
        self.assertEqual(gdpr.regulation_number, "Regulation (EU) 2016/679")
        self.assertEqual(
            gdpr.authority_level,
            AuthorityLevel.BINDING_LEGISLATION,
        )

    def test_legacy_alias_resolves_to_canonical_instrument(self) -> None:
        catalog = load_legal_source_catalog()

        gdpr = catalog.resolve_alias("  gdpr2016:679.TXT ")
        ai_act = catalog.resolve_alias(
            "EU AI Act，Regulation (EU) 2024:1689.txt"
        )

        self.assertIsNotNone(gdpr)
        self.assertEqual(gdpr.instrument_id, "GDPR")
        self.assertIsNotNone(ai_act)
        self.assertEqual(ai_act.instrument_id, "EU_AI_ACT")
        self.assertIsNone(catalog.resolve_alias("unknown-source.txt"))

    def test_gdpr_retrieval_uses_legacy_filename_fallback(self) -> None:
        records = [
            {
                "id": "legacy-gdpr-22",
                "document": "Automated individual decision-making provision.",
                "metadata": {
                    "source": "GDPR2016:679.txt",
                    "canonical_citation": "Article 22(1)",
                    "article_number": "22",
                },
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            store_path = Path(temporary_directory) / "vector_store.json"
            store_path.write_text(json.dumps(records), encoding="utf-8")

            evidence = VectorStoreJSONEvidenceRetriever(store_path).retrieve(
                "GDPR",
                "Article 22(1)",
            )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].legal_source, "GDPR")
        self.assertEqual(evidence[0].citation, "Article 22(1)")
        self.assertEqual(
            evidence[0].authority_level,
            AuthorityLevel.BINDING_LEGISLATION,
        )
        self.assertEqual(
            evidence[0].document_version,
            "Regulation (EU) 2016/679",
        )

    def test_existing_ai_act_vector_store_remains_retrievable(self) -> None:
        retriever = VectorStoreJSONEvidenceRetriever(
            PROJECT_ROOT / "vector_store.json"
        )

        evidence = retriever.retrieve("EU_AI_ACT", "Article 6", limit=2)

        self.assertGreaterEqual(len(evidence), 1)
        self.assertTrue(
            all(item.legal_source == "EU_AI_ACT" for item in evidence)
        )
        self.assertTrue(all(item.citation == "Article 6" for item in evidence))
        self.assertTrue(
            all(
                item.authority_level is AuthorityLevel.BINDING_LEGISLATION
                for item in evidence
            )
        )


if __name__ == "__main__":
    unittest.main()
