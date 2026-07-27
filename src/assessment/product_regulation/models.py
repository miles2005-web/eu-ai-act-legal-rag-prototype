"""Typed reference models for EU AI Act Annex I instruments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from src.assessment.models import SerializableModel


class AnnexISection(str, Enum):
    """Sections authored in Annex I of Regulation (EU) 2024/1689."""

    A = "A"
    B = "B"


class AnnexIInstrumentType(str, Enum):
    """Legal instrument types currently represented in Annex I."""

    DIRECTIVE = "directive"
    REGULATION = "regulation"


@dataclass(frozen=True, slots=True)
class AnnexIInstrument(SerializableModel):
    """One stable, non-conclusive reference to an Annex I instrument."""

    instrument_id: str
    annex_section: AnnexISection
    annex_point: int
    instrument_type: AnnexIInstrumentType
    instrument_number: str
    official_title_en: str
    display_title_zh: str
    canonical_reference: str
    product_categories: tuple[str, ...]
    aliases: tuple[str, ...]
    source_version: str
    official_source_uri: str

    _INSTRUMENT_ID_PATTERN = re.compile(
        r"^ANNEX_I_([AB])_(\d{2})_[A-Z0-9_]+$"
    )

    def __post_init__(self) -> None:
        for field_name in (
            "instrument_id",
            "instrument_number",
            "official_title_en",
            "display_title_zh",
            "canonical_reference",
            "source_version",
            "official_source_uri",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.annex_section, AnnexISection):
            raise TypeError("annex_section must be an AnnexISection")
        if (
            isinstance(self.annex_point, bool)
            or not isinstance(self.annex_point, int)
            or self.annex_point <= 0
        ):
            raise ValueError("annex_point must be a positive integer")
        if not isinstance(self.instrument_type, AnnexIInstrumentType):
            raise TypeError(
                "instrument_type must be an AnnexIInstrumentType"
            )
        identifier_match = self._INSTRUMENT_ID_PATTERN.fullmatch(
            self.instrument_id
        )
        if identifier_match is None:
            raise ValueError(
                "instrument_id must use the stable Annex I ID format"
            )
        if (
            identifier_match.group(1) != self.annex_section.value
            or int(identifier_match.group(2)) != self.annex_point
        ):
            raise ValueError(
                "instrument_id must match annex_section and annex_point"
            )
        for field_name in ("product_categories", "aliases"):
            values = getattr(self, field_name)
            if not values or any(
                not isinstance(value, str) or not value.strip()
                for value in values
            ):
                raise ValueError(
                    f"{field_name} must contain non-empty strings"
                )
        expected_reference = (
            f"Annex I, Section {self.annex_section.value}, "
            f"point {self.annex_point}"
        )
        if self.canonical_reference != expected_reference:
            raise ValueError(
                "canonical_reference must match annex_section and annex_point"
            )
        if not self.official_source_uri.startswith("https://"):
            raise ValueError("official_source_uri must be an HTTPS URI")

    def display_label(self, language: str = "en") -> str:
        """Return presentation text without changing canonical identity."""

        if language == "en":
            return self.official_title_en
        if language == "zh-CN":
            return self.display_title_zh
        raise ValueError("language must be 'en' or 'zh-CN'")


__all__ = [
    "AnnexIInstrument",
    "AnnexIInstrumentType",
    "AnnexISection",
]
