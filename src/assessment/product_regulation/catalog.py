"""Validated, deterministic access to the EU AI Act Annex I catalogue."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import unicodedata

from src.assessment.facts import ProductRegulationFacts
from src.assessment.models import TriState
from src.assessment.product_regulation.models import (
    AnnexIInstrument,
    AnnexIInstrumentType,
    AnnexISection,
)


DEFAULT_ANNEX_I_CATALOG_PATH = (
    Path(__file__).resolve().parents[3]
    / "config"
    / "ai_act_annex_i_instruments.json"
)


class AnnexIInstrumentCatalogError(ValueError):
    """Raised when authored Annex I catalogue data are invalid."""


class AnnexIInstrumentNotFoundError(KeyError):
    """Raised when a stable instrument ID is not catalogued."""


class AnnexIInstrumentAliasNotFoundError(KeyError):
    """Raised when a controlled alias is not catalogued."""


class AmbiguousAnnexIInstrumentAliasError(ValueError):
    """Raised when one normalized alias identifies multiple instruments."""


class InvalidProductRegulationFactsError(ValueError):
    """Raised when product-regulation facts are structurally invalid."""


class AnnexIInstrumentCatalog:
    """Lookup Annex I instruments without making legal classifications."""

    def __init__(
        self,
        *,
        schema_version: str,
        instruments: tuple[AnnexIInstrument, ...],
    ) -> None:
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise AnnexIInstrumentCatalogError(
                "catalog_schema_version must be a non-empty string"
            )
        if not instruments:
            raise AnnexIInstrumentCatalogError(
                "catalogue must contain at least one instrument"
            )
        self.schema_version = schema_version.strip()
        self._instruments_by_id: dict[str, AnnexIInstrument] = {}
        self._instrument_ids_by_alias: dict[str, list[str]] = {}
        seen_locations: set[tuple[AnnexISection, int]] = set()
        for instrument in instruments:
            if not isinstance(instrument, AnnexIInstrument):
                raise TypeError(
                    "instruments must contain AnnexIInstrument values"
                )
            if instrument.instrument_id in self._instruments_by_id:
                raise AnnexIInstrumentCatalogError(
                    f"duplicate instrument_id {instrument.instrument_id!r}"
                )
            location = (instrument.annex_section, instrument.annex_point)
            if location in seen_locations:
                raise AnnexIInstrumentCatalogError(
                    "duplicate Annex I section/point pair "
                    f"{instrument.annex_section.value}/{instrument.annex_point}"
                )
            self._instruments_by_id[instrument.instrument_id] = instrument
            seen_locations.add(location)
            self._index_aliases(instrument)

        self._ordered = tuple(
            sorted(
                self._instruments_by_id.values(),
                key=lambda item: (
                    0 if item.annex_section is AnnexISection.A else 1,
                    item.annex_point,
                    item.instrument_id,
                ),
            )
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnnexIInstrumentCatalog:
        """Build a validated catalogue from decoded JSON."""

        if not isinstance(payload, dict):
            raise AnnexIInstrumentCatalogError(
                "catalogue root must be an object"
            )
        raw_instruments = payload.get("instruments")
        if not isinstance(raw_instruments, list):
            raise AnnexIInstrumentCatalogError(
                "catalogue instruments must be a list"
            )
        instruments = tuple(
            cls._instrument_from_dict(item, index=index)
            for index, item in enumerate(raw_instruments)
        )
        return cls(
            schema_version=payload.get("catalog_schema_version"),
            instruments=instruments,
        )

    def all(self) -> tuple[AnnexIInstrument, ...]:
        """Return all instruments in deterministic Annex order."""

        return self._ordered

    def get(self, instrument_id: str) -> AnnexIInstrument:
        """Resolve a canonical ID and reject unknown values explicitly."""

        if not isinstance(instrument_id, str) or not instrument_id.strip():
            raise AnnexIInstrumentNotFoundError(
                "instrument_id must be a non-empty string"
            )
        try:
            return self._instruments_by_id[instrument_id.strip()]
        except KeyError as exc:
            raise AnnexIInstrumentNotFoundError(
                f"Annex I instrument {instrument_id!r} is not catalogued"
            ) from exc

    def resolve_alias(self, alias: str) -> AnnexIInstrument:
        """Resolve an exact controlled alias, never fuzzy text."""

        normalized = normalize_annex_i_alias(alias)
        instrument_ids = self._instrument_ids_by_alias.get(normalized, [])
        if not instrument_ids:
            raise AnnexIInstrumentAliasNotFoundError(
                f"Annex I alias {alias!r} is not catalogued"
            )
        if len(instrument_ids) > 1:
            raise AmbiguousAnnexIInstrumentAliasError(
                f"Annex I alias {alias!r} is ambiguous: "
                + ", ".join(instrument_ids)
            )
        return self._instruments_by_id[instrument_ids[0]]

    def list_by_section(
        self,
        section: AnnexISection | str,
    ) -> tuple[AnnexIInstrument, ...]:
        """Return one Annex section in deterministic point order."""

        try:
            normalized = (
                section
                if isinstance(section, AnnexISection)
                else AnnexISection(str(section).strip().upper())
            )
        except (TypeError, ValueError) as exc:
            raise AnnexIInstrumentCatalogError(
                "section must be Annex I section A or B"
            ) from exc
        return tuple(
            item for item in self._ordered if item.annex_section is normalized
        )

    def display_label(self, instrument_id: str, language: str = "en") -> str:
        """Return a bilingual label while retaining the same canonical ID."""

        return self.get(instrument_id).display_label(language)

    def _index_aliases(self, instrument: AnnexIInstrument) -> None:
        authored_aliases = (
            instrument.instrument_number,
            *instrument.aliases,
        )
        seen_for_instrument: set[str] = set()
        for alias in authored_aliases:
            normalized = normalize_annex_i_alias(alias)
            if normalized in seen_for_instrument:
                raise AnnexIInstrumentCatalogError(
                    f"instrument {instrument.instrument_id!r} contains "
                    f"duplicate normalized alias {alias!r}"
                )
            seen_for_instrument.add(normalized)
            self._instrument_ids_by_alias.setdefault(normalized, []).append(
                instrument.instrument_id
            )

    @staticmethod
    def _instrument_from_dict(
        payload: object,
        *,
        index: int,
    ) -> AnnexIInstrument:
        if not isinstance(payload, dict):
            raise AnnexIInstrumentCatalogError(
                f"catalogue instrument {index} must be an object"
            )
        required_strings = (
            "instrument_id",
            "annex_section",
            "instrument_type",
            "instrument_number",
            "official_title_en",
            "display_title_zh",
            "canonical_reference",
            "source_version",
            "official_source_uri",
        )
        values: dict[str, str] = {}
        for field_name in required_strings:
            value = payload.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise AnnexIInstrumentCatalogError(
                    f"catalogue instrument {index} has invalid {field_name}"
                )
            values[field_name] = value.strip()

        annex_point = payload.get("annex_point")
        if (
            isinstance(annex_point, bool)
            or not isinstance(annex_point, int)
            or annex_point <= 0
        ):
            raise AnnexIInstrumentCatalogError(
                f"catalogue instrument {index} has invalid annex_point"
            )
        product_categories = _required_string_tuple(
            payload.get("product_categories"),
            field_name="product_categories",
            index=index,
        )
        aliases = _required_string_tuple(
            payload.get("aliases"),
            field_name="aliases",
            index=index,
        )
        try:
            return AnnexIInstrument(
                instrument_id=values["instrument_id"],
                annex_section=AnnexISection(values["annex_section"].upper()),
                annex_point=annex_point,
                instrument_type=AnnexIInstrumentType(
                    values["instrument_type"].casefold()
                ),
                instrument_number=values["instrument_number"],
                official_title_en=values["official_title_en"],
                display_title_zh=values["display_title_zh"],
                canonical_reference=values["canonical_reference"],
                product_categories=product_categories,
                aliases=aliases,
                source_version=values["source_version"],
                official_source_uri=values["official_source_uri"],
            )
        except (TypeError, ValueError) as exc:
            raise AnnexIInstrumentCatalogError(
                f"catalogue instrument {index} is invalid: {exc}"
            ) from exc


def normalize_annex_i_alias(value: str) -> str:
    """Normalize controlled aliases deterministically without fuzzy matching."""

    if not isinstance(value, str) or not value.strip():
        raise AnnexIInstrumentAliasNotFoundError(
            "Annex I alias must be a non-empty string"
        )
    normalized = unicodedata.normalize("NFC", value).strip().casefold()
    return " ".join(normalized.split())


def load_annex_i_instrument_catalog(
    path: str | Path = DEFAULT_ANNEX_I_CATALOG_PATH,
) -> AnnexIInstrumentCatalog:
    """Load the version-controlled Annex I instrument catalogue."""

    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8") as catalog_file:
        payload = json.load(catalog_file)
    return AnnexIInstrumentCatalog.from_dict(payload)


def validate_product_regulation_facts(
    facts: ProductRegulationFacts,
    *,
    catalog: AnnexIInstrumentCatalog | None = None,
) -> AnnexIInstrument | None:
    """Validate fact shape and selected ID without deriving applicability."""

    if not isinstance(facts, ProductRegulationFacts):
        raise TypeError("facts must be ProductRegulationFacts")
    for field_name in (
        "ai_is_product",
        "ai_is_safety_component",
        "annex_i_instrument_confirmed",
        "third_party_conformity_required",
    ):
        if not isinstance(getattr(facts, field_name), TriState):
            raise InvalidProductRegulationFactsError(
                f"{field_name} must be a TriState"
            )
    if facts.product_type is not None and (
        not isinstance(facts.product_type, str)
        or not facts.product_type.strip()
    ):
        raise InvalidProductRegulationFactsError(
            "product_type must be a non-empty string or None"
        )
    if facts.annex_i_instrument is None:
        return None
    if (
        not isinstance(facts.annex_i_instrument, str)
        or not facts.annex_i_instrument.strip()
    ):
        raise InvalidProductRegulationFactsError(
            "annex_i_instrument must be a stable catalogue ID or None"
        )
    return (catalog or load_annex_i_instrument_catalog()).get(
        facts.annex_i_instrument
    )


def _required_string_tuple(
    value: object,
    *,
    field_name: str,
    index: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise AnnexIInstrumentCatalogError(
            f"catalogue instrument {index} has invalid {field_name}"
        )
    return tuple(item.strip() for item in value)


__all__ = [
    "AmbiguousAnnexIInstrumentAliasError",
    "AnnexIInstrumentAliasNotFoundError",
    "AnnexIInstrumentCatalog",
    "AnnexIInstrumentCatalogError",
    "AnnexIInstrumentNotFoundError",
    "DEFAULT_ANNEX_I_CATALOG_PATH",
    "InvalidProductRegulationFactsError",
    "load_annex_i_instrument_catalog",
    "normalize_annex_i_alias",
    "validate_product_regulation_facts",
]
