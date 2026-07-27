"""Version-controlled legal source catalog for evidence resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.assessment.evidence.models import AuthorityLevel


DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "legal_sources.json"
)


class LegalSourceCatalogError(ValueError):
    """Raised when legal source catalog data are invalid or ambiguous."""


@dataclass(frozen=True, slots=True)
class LegalSource:
    """Metadata for one stable legal instrument identity."""

    instrument_id: str
    title: str
    regulation_number: str
    authority_level: AuthorityLevel
    version: str
    official_uri: str
    jurisdiction: str
    language: str
    source_aliases: tuple[str, ...]


class LegalSourceCatalog:
    """Validated lookup by instrument ID or legacy source filename."""

    def __init__(
        self,
        *,
        schema_version: str,
        sources: tuple[LegalSource, ...],
    ) -> None:
        if not isinstance(schema_version, str) or not schema_version.strip():
            raise LegalSourceCatalogError(
                "catalog_schema_version must be a non-empty string"
            )
        if not sources:
            raise LegalSourceCatalogError("catalog must contain at least one source")

        self.schema_version = schema_version
        self._sources_by_id: dict[str, LegalSource] = {}
        self._instrument_id_by_alias: dict[str, str] = {}
        for source in sources:
            self._register(source)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LegalSourceCatalog:
        """Build and validate a catalog from decoded JSON data."""

        if not isinstance(payload, dict):
            raise LegalSourceCatalogError("catalog root must be an object")
        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, list):
            raise LegalSourceCatalogError("catalog sources must be a list")

        sources: list[LegalSource] = []
        for index, raw_source in enumerate(raw_sources):
            if not isinstance(raw_source, dict):
                raise LegalSourceCatalogError(
                    f"catalog source {index} must be an object"
                )
            sources.append(cls._source_from_dict(raw_source, index=index))
        return cls(
            schema_version=payload.get("catalog_schema_version"),
            sources=tuple(sources),
        )

    def get(self, instrument_id: str) -> LegalSource:
        """Return one source by canonical instrument ID."""

        key = self._normalize(instrument_id)
        try:
            return self._sources_by_id[key]
        except KeyError as exc:
            raise KeyError(
                f"Legal instrument {instrument_id!r} is not catalogued"
            ) from exc

    def resolve_alias(self, source_alias: str) -> LegalSource | None:
        """Resolve a legacy filename alias to its legal source metadata."""

        instrument_id = self._instrument_id_by_alias.get(
            self._normalize(source_alias)
        )
        return (
            self._sources_by_id[instrument_id]
            if instrument_id is not None
            else None
        )

    def all(self) -> tuple[LegalSource, ...]:
        """Return sources in authored catalog order."""

        return tuple(self._sources_by_id.values())

    def _register(self, source: LegalSource) -> None:
        instrument_key = self._normalize(source.instrument_id)
        if instrument_key in self._sources_by_id:
            raise LegalSourceCatalogError(
                f"duplicate instrument_id {source.instrument_id!r}"
            )
        self._sources_by_id[instrument_key] = source

        for alias in source.source_aliases:
            alias_key = self._normalize(alias)
            if alias_key in self._instrument_id_by_alias:
                raise LegalSourceCatalogError(
                    f"source alias {alias!r} belongs to multiple instruments"
                )
            self._instrument_id_by_alias[alias_key] = instrument_key

    @classmethod
    def _source_from_dict(
        cls,
        raw_source: dict[str, Any],
        *,
        index: int,
    ) -> LegalSource:
        required_strings = (
            "instrument_id",
            "title",
            "regulation_number",
            "version",
            "official_uri",
            "jurisdiction",
            "language",
        )
        values: dict[str, str] = {}
        for field_name in required_strings:
            value = raw_source.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise LegalSourceCatalogError(
                    f"catalog source {index} has invalid {field_name}"
                )
            values[field_name] = value.strip()
        if not values["official_uri"].startswith(("http://", "https://")):
            raise LegalSourceCatalogError(
                f"catalog source {index} has invalid official_uri"
            )

        raw_aliases = raw_source.get("source_aliases")
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise LegalSourceCatalogError(
                f"catalog source {index} must define source_aliases"
            )
        if any(not isinstance(alias, str) or not alias.strip() for alias in raw_aliases):
            raise LegalSourceCatalogError(
                f"catalog source {index} contains an invalid source alias"
            )
        aliases = tuple(alias.strip() for alias in raw_aliases)
        if len({cls._normalize(alias) for alias in aliases}) != len(aliases):
            raise LegalSourceCatalogError(
                f"catalog source {index} contains duplicate source aliases"
            )

        try:
            authority_level = AuthorityLevel(raw_source.get("authority_level"))
        except (TypeError, ValueError) as exc:
            raise LegalSourceCatalogError(
                f"catalog source {index} has invalid authority_level"
            ) from exc

        return LegalSource(
            authority_level=authority_level,
            source_aliases=aliases,
            **values,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise LegalSourceCatalogError("catalog lookup value must be non-empty")
        return " ".join(value.strip().casefold().split())


def load_legal_source_catalog(
    path: str | Path = DEFAULT_CATALOG_PATH,
) -> LegalSourceCatalog:
    """Load the version-controlled legal source catalog from JSON."""

    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8") as catalog_file:
        payload = json.load(catalog_file)
    return LegalSourceCatalog.from_dict(payload)
