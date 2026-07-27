"""Tests for the versioned EU AI Act Annex I instrument catalogue."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from src.assessment.product_regulation import (
    AmbiguousAnnexIInstrumentAliasError,
    AnnexIInstrumentAliasNotFoundError,
    AnnexIInstrumentCatalog,
    AnnexIInstrumentCatalogError,
    AnnexIInstrumentNotFoundError,
    AnnexISection,
    DEFAULT_ANNEX_I_CATALOG_PATH,
    load_annex_i_instrument_catalog,
    normalize_annex_i_alias,
)


class AnnexIInstrumentCatalogTests(unittest.TestCase):
    @staticmethod
    def _raw_catalogue() -> dict[str, object]:
        return json.loads(
            Path(DEFAULT_ANNEX_I_CATALOG_PATH).read_text(encoding="utf-8")
        )

    def test_default_catalogue_contains_twenty_unique_records(self) -> None:
        catalog = load_annex_i_instrument_catalog()
        instruments = catalog.all()

        self.assertEqual(catalog.schema_version, "1.0.0")
        self.assertEqual(len(instruments), 20)
        self.assertEqual(
            len({item.instrument_id for item in instruments}),
            len(instruments),
        )
        self.assertEqual(
            len(
                {
                    (item.annex_section, item.annex_point)
                    for item in instruments
                }
            ),
            len(instruments),
        )

    def test_section_and_point_ordering_is_deterministic(self) -> None:
        catalog = load_annex_i_instrument_catalog()

        self.assertEqual(
            [item.annex_point for item in catalog.list_by_section("A")],
            list(range(1, 13)),
        )
        self.assertEqual(
            [
                item.annex_point
                for item in catalog.list_by_section(AnnexISection.B)
            ],
            list(range(13, 21)),
        )
        self.assertEqual(
            [item.annex_point for item in catalog.all()],
            list(range(1, 21)),
        )

    def test_bilingual_labels_preserve_canonical_identity(self) -> None:
        catalog = load_annex_i_instrument_catalog()
        instrument_id = "ANNEX_I_A_11_MEDICAL_DEVICES_REGULATION_2017_745"
        instrument = catalog.get(instrument_id)

        english = catalog.display_label(instrument_id, "en")
        chinese = catalog.display_label(instrument_id, "zh-CN")

        self.assertIn("Regulation (EU) 2017/745", english)
        self.assertIn("医疗器械条例", chinese)
        self.assertEqual(instrument.instrument_id, instrument_id)
        self.assertEqual(catalog.get(instrument_id), instrument)

    def test_controlled_aliases_normalize_and_resolve_exactly(self) -> None:
        catalog = load_annex_i_instrument_catalog()

        by_number = catalog.resolve_alias("  DIRECTIVE   2006/42/EC ")
        by_common_name = catalog.resolve_alias("machinery directive")

        self.assertEqual(by_number, by_common_name)
        self.assertEqual(
            by_number.instrument_id,
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
        )
        self.assertEqual(
            normalize_annex_i_alias("  Machinery   Directive "),
            "machinery directive",
        )

    def test_unknown_ids_and_aliases_fail_explicitly(self) -> None:
        catalog = load_annex_i_instrument_catalog()

        with self.assertRaises(AnnexIInstrumentNotFoundError):
            catalog.get("UNKNOWN_INSTRUMENT")
        with self.assertRaises(AnnexIInstrumentAliasNotFoundError):
            catalog.resolve_alias("not a controlled alias")

    def test_ambiguous_alias_fails_at_resolution(self) -> None:
        payload = deepcopy(self._raw_catalogue())
        payload["instruments"][1]["aliases"].append("machinery directive")
        catalog = AnnexIInstrumentCatalog.from_dict(payload)

        with self.assertRaises(AmbiguousAnnexIInstrumentAliasError) as context:
            catalog.resolve_alias("machinery directive")

        self.assertIn(
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
            str(context.exception),
        )
        self.assertIn(
            "ANNEX_I_A_02_TOY_SAFETY_DIRECTIVE_2009_48_EC",
            str(context.exception),
        )

    def test_duplicate_id_is_rejected(self) -> None:
        payload = deepcopy(self._raw_catalogue())
        payload["instruments"][1]["instrument_id"] = payload[
            "instruments"
        ][0]["instrument_id"]

        with self.assertRaises(AnnexIInstrumentCatalogError):
            AnnexIInstrumentCatalog.from_dict(payload)

    def test_duplicate_section_and_point_is_rejected(self) -> None:
        payload = deepcopy(self._raw_catalogue())
        payload["instruments"][1]["annex_point"] = 1
        payload["instruments"][1]["canonical_reference"] = (
            "Annex I, Section A, point 1"
        )

        with self.assertRaises(AnnexIInstrumentCatalogError):
            AnnexIInstrumentCatalog.from_dict(payload)

    def test_stable_id_must_match_section_and_point(self) -> None:
        payload = deepcopy(self._raw_catalogue())
        payload["instruments"][0]["instrument_id"] = (
            "ANNEX_I_B_01_MACHINERY_DIRECTIVE_2006_42_EC"
        )

        with self.assertRaises(AnnexIInstrumentCatalogError):
            AnnexIInstrumentCatalog.from_dict(payload)

    def test_required_fields_and_canonical_reference_are_validated(self) -> None:
        missing_title = deepcopy(self._raw_catalogue())
        missing_title["instruments"][0]["official_title_en"] = ""
        wrong_reference = deepcopy(self._raw_catalogue())
        wrong_reference["instruments"][0]["canonical_reference"] = "Annex I"

        with self.assertRaises(AnnexIInstrumentCatalogError):
            AnnexIInstrumentCatalog.from_dict(missing_title)
        with self.assertRaises(AnnexIInstrumentCatalogError):
            AnnexIInstrumentCatalog.from_dict(wrong_reference)


if __name__ == "__main__":
    unittest.main()
