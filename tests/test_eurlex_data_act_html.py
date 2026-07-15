"""Tests for isolated EUR-Lex Data Act XHTML preprocessing."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from src.assessment.evidence import MultiCorpusLegalEvidenceRetriever
from src.data_act_corpus import (
    build_data_act_candidate_records,
    compare_data_act_candidate_to_runtime_pack,
    write_data_act_candidate_corpus,
)
from src.eurlex_html import (
    EurLexHTMLFormatError,
    preprocess_eurlex_data_act_html,
)


EURLEX_SAMPLE = """<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
  <link rel="canonical" href="https://eur-lex.europa.eu/eli/reg/2023/2854/oj/eng" />
</head>
<body>
  <nav><a href="#art_2">article 2 duplicate TOC text</a></nav>
  <div id="docHtml">
    <div class="eli-subdivision" id="rct_1"><p>Whereas:</p></div>
    <div class="eli-subdivision" id="rct_2">
      <table><tr><td><p>(1)</p></td><td><p>First official recital.</p></td></tr></table>
    </div>
    <div class="eli-subdivision" id="art_2">
      <p>Article&nbsp;2</p>
      <div class="eli-title"><p>Definitions</p></div>
      <p>For the purposes of this Regulation:</p>
      <table><tr><td><p>(5)</p></td><td><p>‘connected product’ means an item that generates data.</p></td></tr></table>
      <table><tr><td><p>(6)</p></td><td><p>‘related service’ means a digital service.</p></td></tr></table>
    </div>
    <div class="eli-subdivision" id="art_4">
      <p>Article&nbsp;4</p><div class="eli-title"><p>Access rights</p></div>
      <div id="004.001"><p>1.&nbsp;Data shall be accessible:</p>
        <table><tr><td><p>(a)</p></td><td><p>without undue delay;</p></td></tr></table>
      </div>
    </div>
    <div class="eli-subdivision" id="art_23">
      <p>Article&nbsp;23</p><div class="eli-title"><p>Switching</p></div>
      <p>Providers shall remove obstacles which inhibit customers from:</p>
      <table><tr><td><p>(a)</p></td><td><p>terminating a contract;</p></td></tr></table>
    </div>
    <div class="eli-subdivision" id="anx_III">
      <p>ANNEX III</p><p>Technical requirements</p>
    </div>
  </div>
</body></html>"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SOURCE = (
    PROJECT_ROOT
    / "corpus_sources"
    / "EU_DATA_ACT"
    / "Regulation_2023_2854.html"
)


class EurLexDataActHTMLTests(unittest.TestCase):
    @staticmethod
    def _source(directory: str, html: str = EURLEX_SAMPLE) -> Path:
        path = Path(directory) / "Regulation_2023_2854.html"
        path.write_text(html, encoding="utf-8")
        return path

    def test_preprocessor_preserves_legal_structure_without_toc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prepared = preprocess_eurlex_data_act_html(
                self._source(directory)
            )

        self.assertIn("(1) First official recital.", prepared)
        self.assertIn("Article 2\nDefinitions", prepared)
        self.assertIn("(5)\n‘connected product’ means", prepared)
        self.assertIn("1. Data shall be accessible:", prepared)
        self.assertIn("(a)\nwithout undue delay;", prepared)
        self.assertIn("ANNEX III", prepared)
        self.assertNotIn("duplicate TOC text", prepared)

    def test_html_source_builds_metadata_v2_candidate_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = build_data_act_candidate_records(
                self._source(directory)
            )

        metadata_by_citation = {
            record["metadata"]["canonical_citation"]: record["metadata"]
            for record in records
        }
        self.assertIn("Recital 1", metadata_by_citation)
        self.assertIn("Article 2(5)", metadata_by_citation)
        self.assertIn("Article 4(1)(a)", metadata_by_citation)
        self.assertIn("Article 23(a)", metadata_by_citation)
        self.assertIn("Annex III", metadata_by_citation)
        definition = metadata_by_citation["Article 2(5)"]
        self.assertEqual(definition["definition_term"], "connected product")
        self.assertEqual(definition["instrument_id"], "EU_DATA_ACT")
        self.assertTrue(
            definition["stable_evidence_id"].startswith("evidence:v2:")
        )

    def test_wrong_canonical_source_is_rejected(self) -> None:
        wrong_source = EURLEX_SAMPLE.replace(
            "/reg/2023/2854/oj/eng",
            "/reg/2024/1689/oj/eng",
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(EurLexHTMLFormatError):
                preprocess_eurlex_data_act_html(
                    self._source(directory, wrong_source)
                )

    def test_html_candidate_resolves_through_multi_corpus_retriever(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = build_data_act_candidate_records(
                self._source(directory)
            )
            existing = Path(directory) / "legacy.json"
            existing.write_text("[]", encoding="utf-8")
            candidate = Path(directory) / "data-act-candidate.json"
            write_data_act_candidate_corpus(
                records,
                candidate,
                existing_store_path=existing,
            )
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
        self.assertTrue(evidence[0].evidence_id.startswith("evidence:v2:"))

    @unittest.skipUnless(OFFICIAL_SOURCE.is_file(), "official source is local-only")
    def test_official_source_matches_committed_runtime_pack(self) -> None:
        records = build_data_act_candidate_records(OFFICIAL_SOURCE)
        compare_data_act_candidate_to_runtime_pack(records)


if __name__ == "__main__":
    unittest.main()
