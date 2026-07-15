"""Tests for the isolated official AI Act Article 6(1) corpus."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.ai_act_product_safety_corpus import (
    AI_ACT_PRODUCT_SAFETY_RECORD_COUNT,
    AIActProductSafetyCorpusError,
    build_ai_act_product_safety_candidate_records,
    compare_candidate_to_runtime_pack,
    validate_ai_act_product_safety_candidate,
    write_ai_act_product_safety_candidate_corpus,
)
from src.assessment.evidence import MultiCorpusLegalEvidenceRetriever
from src.assessment.product_regulation import load_annex_i_instrument_catalog
from src.eurlex_ai_act import EurLexAIActFormatError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCE = (
    PROJECT_ROOT
    / "corpus_sources"
    / "EU_AI_ACT"
    / "Regulation_2024_1689.html"
)
MANIFEST_PATH = (
    PROJECT_ROOT / "config" / "ai_act_product_safety_evidence_manifest.json"
)

ARTICLE_3_14 = (
    "(14) ‘safety component’ means a component of a product or of an AI "
    "system which fulfils a safety function for that product or AI system, "
    "or the failure or malfunctioning of which endangers the health and "
    "safety of persons or property;"
)
ARTICLE_6_1_A = (
    "(a) the AI system is intended to be used as a safety component of a "
    "product, or the AI system is itself a product, covered by the Union "
    "harmonisation legislation listed in Annex I;"
)
ARTICLE_6_1_B = (
    "(b) the product whose safety component pursuant to point (a) is the AI "
    "system, or the AI system itself as a product, is required to undergo a "
    "third-party conformity assessment, with a view to the placing on the "
    "market or the putting into service of that product pursuant to the "
    "Union harmonisation legislation listed in Annex I."
)


def _official_excerpt_html() -> str:
    catalog = load_annex_i_instrument_catalog()
    annex_blocks: list[str] = []
    for section in ("A", "B"):
        annex_blocks.append(f"<p>Section {section}. Official list</p>")
        for instrument in catalog.list_by_section(section):
            annex_blocks.append(
                "<table><tr><td><p>"
                f"{instrument.annex_point}."
                "</p></td><td><span>"
                f"{instrument.official_title_en}"
                "</span></td></tr></table>"
            )
    return """<!doctype html>
<html><head>
<link rel="canonical" href="https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng" />
<meta name="WT.z_docID" content="32024R1689" />
<meta name="WT.z_docTitle" content="Regulation (EU) 2024/1689 official test excerpt" />
</head><body>
<div class="eli-subdivision" id="art_3">
  <p>Article 3</p><p>Definitions</p>
  <table><tr><td><p>(14)</p></td><td><p>""" + ARTICLE_3_14.removeprefix("(14) ") + """</p></td></tr></table>
</div>
<div class="eli-subdivision" id="art_6">
  <p>Article 6</p><p>Classification rules</p>
  <p>1. Both conditions apply:</p>
  <table><tr><td><p>(a)</p></td><td><p>""" + ARTICLE_6_1_A.removeprefix("(a) ") + """</p></td></tr></table>
  <table><tr><td><p>(b)</p></td><td><p>""" + ARTICLE_6_1_B.removeprefix("(b) ") + """</p></td></tr></table>
</div>
<div class="eli-container" id="anx_I">
  <p>ANNEX I</p>""" + "".join(annex_blocks) + """
</div></body></html>"""


class AIActProductSafetyCorpusTests(unittest.TestCase):
    @staticmethod
    def _source(directory: str, html: str | None = None) -> Path:
        path = Path(directory) / "Regulation_2024_1689.html"
        path.write_text(html or _official_excerpt_html(), encoding="utf-8")
        return path

    def test_official_parser_builds_required_atomic_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = build_ai_act_product_safety_candidate_records(
                self._source(directory)
            )

        by_citation = {
            record["metadata"]["canonical_citation"]: record
            for record in records
        }
        self.assertEqual(len(records), AI_ACT_PRODUCT_SAFETY_RECORD_COUNT)
        self.assertIn("Article 3(14)", by_citation)
        self.assertIn("Article 6(1)(a)", by_citation)
        self.assertIn("Article 6(1)(b)", by_citation)
        annex = [
            citation
            for citation in by_citation
            if citation.startswith("Annex I,")
        ]
        self.assertEqual(len(annex), 20)

    def test_metadata_is_complete_isolated_and_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = build_ai_act_product_safety_candidate_records(
                self._source(directory)
            )

        for record in records:
            metadata = record["metadata"]
            self.assertEqual(metadata["instrument_id"], "EU_AI_ACT")
            self.assertEqual(metadata["framework"], "EU_AI_ACT")
            self.assertEqual(metadata["celex"], "32024R1689")
            self.assertEqual(metadata["language"], "en")
            self.assertEqual(metadata["authority_level"], "binding_legislation")
            self.assertEqual(
                metadata["official_source_uri"],
                "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng",
            )
            self.assertTrue(record["document"].strip())
            self.assertTrue(metadata["authoritative_excerpt_hash"])
            self.assertTrue(metadata["stable_evidence_id"].startswith("evidence:v2:"))
            self.assertEqual(record["id"], metadata["source_record_id"])

    def test_rebuild_and_record_order_preserve_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            first = build_ai_act_product_safety_candidate_records(source)
            second = build_ai_act_product_safety_candidate_records(source)

        first_ids = {
            item["metadata"]["canonical_citation"]:
            item["metadata"]["stable_evidence_id"]
            for item in first
        }
        second_ids = {
            item["metadata"]["canonical_citation"]:
            item["metadata"]["stable_evidence_id"]
            for item in reversed(second)
        }
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(set(first_ids.values())), 23)

    def test_catalogue_and_annex_evidence_are_one_to_one(self) -> None:
        catalog = load_annex_i_instrument_catalog()
        with tempfile.TemporaryDirectory() as directory:
            records = build_ai_act_product_safety_candidate_records(
                self._source(directory), annex_catalog=catalog
            )

        validate_ai_act_product_safety_candidate(records, catalog)
        annex_metadata = {
            record["metadata"]["catalogue_instrument_id"]: record["metadata"]
            for record in records
            if record["metadata"]["canonical_citation"].startswith("Annex I,")
        }
        self.assertEqual(set(annex_metadata), {item.instrument_id for item in catalog.all()})
        for instrument in catalog.all():
            metadata = annex_metadata[instrument.instrument_id]
            self.assertEqual(metadata["annex_section"], instrument.annex_section.value)
            self.assertEqual(metadata["annex_point"], instrument.annex_point)
            self.assertEqual(metadata["catalogue_instrument_number"], instrument.instrument_number)
            self.assertEqual(metadata["canonical_citation"], instrument.canonical_reference)

    def test_all_atomic_citations_resolve_exactly(self) -> None:
        catalog = load_annex_i_instrument_catalog()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = build_ai_act_product_safety_candidate_records(
                self._source(directory), annex_catalog=catalog
            )
            existing = root / "vector_store.json"
            candidate = root / "candidate.json"
            existing.write_text("[]", encoding="utf-8")
            candidate.write_text(json.dumps(records), encoding="utf-8")
            retriever = MultiCorpusLegalEvidenceRetriever.from_store_paths(
                existing,
                [candidate],
            )

            citations = [
                "Article 3(14)",
                "Article 6(1)(a)",
                "Article 6(1)(b)",
                *[item.canonical_reference for item in catalog.all()],
            ]
            resolved = [
                retriever.retrieve("EU_AI_ACT", citation)
                for citation in citations
            ]

        self.assertTrue(all(len(items) == 1 for items in resolved))
        self.assertEqual(
            [items[0].citation for items in resolved],
            citations,
        )

    def test_candidate_cannot_overwrite_active_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = self._source(directory)
            records = build_ai_act_product_safety_candidate_records(source)
            active = Path(directory) / "vector_store.json"
            active.write_text("legacy", encoding="utf-8")
            with self.assertRaises(AIActProductSafetyCorpusError):
                write_ai_act_product_safety_candidate_corpus(
                    records,
                    active,
                    existing_store_path=active,
                )
            self.assertEqual(active.read_text(encoding="utf-8"), "legacy")

    def test_wrong_official_source_is_rejected(self) -> None:
        wrong = _official_excerpt_html().replace("/2024/1689/", "/2023/2854/")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(EurLexAIActFormatError):
                build_ai_act_product_safety_candidate_records(
                    self._source(directory, wrong)
                )

    @unittest.skipUnless(OFFICIAL_SOURCE.is_file(), "official source is local-only")
    def test_official_source_matches_stable_id_manifest(self) -> None:
        records = build_ai_act_product_safety_candidate_records(OFFICIAL_SOURCE)
        compare_candidate_to_runtime_pack(records)
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            {
                record["canonical_citation"]: record["stable_evidence_id"]
                for record in manifest["records"]
            },
            {
                record["metadata"]["canonical_citation"]:
                record["metadata"]["stable_evidence_id"]
                for record in records
            },
        )


if __name__ == "__main__":
    unittest.main()
