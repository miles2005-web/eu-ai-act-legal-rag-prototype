"""Read-only adapters from existing legal stores to Evidence objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from src.assessment.evidence.catalog import (
    LegalSourceCatalog,
    load_legal_source_catalog,
)
from src.assessment.evidence.models import AuthorityLevel, Evidence


class VectorStoreFormatError(ValueError):
    """Raised when the existing JSON store does not match its record format."""


class LegalEvidenceRetriever(ABC):
    """Contract for converting a legal reference into supporting evidence."""

    @abstractmethod
    def retrieve(
        self,
        legal_source: str,
        citation: str,
        *,
        limit: int = 5,
    ) -> list[Evidence]:
        """Return evidence for one legal source and citation reference."""

        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _VectorStoreRecord:
    record_id: str
    document: str
    metadata: dict[str, Any]


class VectorStoreJSONEvidenceRetriever(LegalEvidenceRetriever):
    """Metadata-only adapter for the existing ``vector_store.json`` format.

    Embeddings in the source file are ignored. Matching uses only existing
    source and legal-reference metadata and preserves source record order.
    """

    DEFAULT_SOURCE_ALIASES = {
        "EU_AI_ACT": (
            "EU AI Act，Regulation (EU) 2024:1689.txt",
            "AI Act Annexes I-XIII.txt",
            "AI Act Recitals.txt",
        ),
    }
    DEFAULT_DOCUMENT_VERSIONS = {
        "EU_AI_ACT": "Regulation (EU) 2024/1689",
    }
    DEFAULT_AUTHORITY_LEVELS = {
        "EU_AI_ACT": AuthorityLevel.BINDING_LEGISLATION,
    }

    _ARTICLE_PATTERN = re.compile(r"\barticle\s+(\d+[a-z]?)", re.IGNORECASE)
    _ANNEX_PATTERN = re.compile(
        r"\bannex\s+([ivxlcdm]+|\d+[a-z]?)\b",
        re.IGNORECASE,
    )
    _RECITAL_PATTERN = re.compile(r"\brecital\s+(\d+[a-z]?)", re.IGNORECASE)

    def __init__(
        self,
        store_path: str | Path = "vector_store.json",
        *,
        source_aliases: Mapping[str, Iterable[str]] | None = None,
        document_versions: Mapping[str, str | None] | None = None,
        authority_levels: Mapping[str, AuthorityLevel] | None = None,
        source_catalog: LegalSourceCatalog | None = None,
    ) -> None:
        self._store_path = Path(store_path)
        self._source_catalog = source_catalog or load_legal_source_catalog()
        if not isinstance(self._source_catalog, LegalSourceCatalog):
            raise TypeError("source_catalog must be a LegalSourceCatalog")
        catalog_aliases = {
            source.instrument_id: source.source_aliases
            for source in self._source_catalog.all()
        }
        catalog_versions = {
            source.instrument_id: source.version
            for source in self._source_catalog.all()
        }
        catalog_authorities = {
            source.instrument_id: source.authority_level
            for source in self._source_catalog.all()
        }
        self._source_aliases = self._build_aliases(
            catalog_aliases,
            source_aliases,
        )
        self._document_versions = self._build_values(
            self.DEFAULT_DOCUMENT_VERSIONS,
            catalog_versions,
            document_versions,
        )
        self._authority_levels = self._build_values(
            self.DEFAULT_AUTHORITY_LEVELS,
            catalog_authorities,
            authority_levels,
        )
        if any(
            not isinstance(level, AuthorityLevel)
            for level in self._authority_levels.values()
        ):
            raise TypeError("authority_levels must contain AuthorityLevel values")
        self._records = self._load_records(self._store_path)

    def retrieve(
        self,
        legal_source: str,
        citation: str,
        *,
        limit: int = 5,
    ) -> list[Evidence]:
        """Convert metadata-matched store records into Evidence objects."""

        self._validate_reference(legal_source, citation, limit)
        allowed_sources = self._allowed_sources(legal_source)
        source_records = [
            record
            for record in self._records
            if self._record_matches_source(
                record,
                source_key=self._normalize(legal_source),
                allowed_sources=allowed_sources,
            )
        ]
        matching_records = self._filter_by_citation(source_records, citation)

        source_key = self._normalize(legal_source)
        document_version = self._document_versions.get(source_key)
        authority_level = self._authority_levels.get(
            source_key,
            AuthorityLevel.UNKNOWN,
        )
        return [
            self._to_evidence(
                record=record,
                legal_source=legal_source,
                citation=citation,
                document_version=document_version,
                authority_level=authority_level,
            )
            for record in matching_records[:limit]
        ]

    @classmethod
    def _load_records(cls, store_path: Path) -> tuple[_VectorStoreRecord, ...]:
        with store_path.open("r", encoding="utf-8") as store_file:
            raw_records = json.load(store_file)
        if not isinstance(raw_records, list):
            raise VectorStoreFormatError("vector store root must be a JSON list")

        records: list[_VectorStoreRecord] = []
        for index, item in enumerate(raw_records):
            if not isinstance(item, dict):
                raise VectorStoreFormatError(f"record {index} must be an object")
            record_id = item.get("id")
            document = item.get("document")
            metadata = item.get("metadata") or item.get("meta")
            if not isinstance(record_id, str) or not record_id.strip():
                raise VectorStoreFormatError(
                    f"record {index} must contain a non-empty string id"
                )
            if not isinstance(document, str) or not document.strip():
                raise VectorStoreFormatError(
                    f"record {index} must contain non-empty document text"
                )
            if not isinstance(metadata, dict):
                raise VectorStoreFormatError(
                    f"record {index} must contain metadata"
                )
            records.append(
                _VectorStoreRecord(
                    record_id=record_id,
                    document=document,
                    metadata=dict(metadata),
                )
            )
        return tuple(records)

    def _filter_by_citation(
        self,
        records: list[_VectorStoreRecord],
        citation: str,
    ) -> list[_VectorStoreRecord]:
        normalized_citation = self._normalize(citation)
        exact_matches = [
            record
            for record in records
            if self._normalize(record.metadata.get("canonical_citation"))
            == normalized_citation
        ]
        if exact_matches:
            return exact_matches

        article_match = self._ARTICLE_PATTERN.search(citation)
        if article_match:
            article_number = article_match.group(1).upper()
            return [
                record
                for record in records
                if self._metadata_reference(record, "article_number")
                == article_number
            ]

        annex_match = self._ANNEX_PATTERN.search(citation)
        if annex_match:
            annex_ref = annex_match.group(1).upper()
            return [
                record
                for record in records
                if self._metadata_reference(record, "annex_ref") == annex_ref
            ]

        recital_match = self._RECITAL_PATTERN.search(citation)
        if recital_match:
            recital_ref = recital_match.group(1).upper()
            return [
                record
                for record in records
                if self._metadata_reference(record, "recital_ref") == recital_ref
            ]

        return []

    def _to_evidence(
        self,
        *,
        record: _VectorStoreRecord,
        legal_source: str,
        citation: str,
        document_version: str | None,
        authority_level: AuthorityLevel,
    ) -> Evidence:
        identity = "|".join(
            (self._normalize(legal_source), self._normalize(citation), record.record_id)
        )
        evidence_id = f"vector-store:{sha256(identity.encode('utf-8')).hexdigest()[:24]}"
        return Evidence(
            evidence_id=evidence_id,
            legal_source=legal_source,
            citation=citation,
            excerpt=record.document,
            document_version=document_version,
            authority_level=authority_level,
        )

    def _allowed_sources(self, legal_source: str) -> frozenset[str]:
        source_key = self._normalize(legal_source)
        return self._source_aliases.get(source_key, frozenset((source_key,)))

    def _record_matches_source(
        self,
        record: _VectorStoreRecord,
        *,
        source_key: str,
        allowed_sources: frozenset[str],
    ) -> bool:
        instrument_id = self._normalize(record.metadata.get("instrument_id"))
        if instrument_id:
            return instrument_id == source_key
        return self._normalize(record.metadata.get("source")) in allowed_sources

    @classmethod
    def _build_aliases(
        cls,
        catalog_aliases: Mapping[str, Iterable[str]],
        custom_aliases: Mapping[str, Iterable[str]] | None,
    ) -> dict[str, frozenset[str]]:
        combined: dict[str, Iterable[str]] = dict(cls.DEFAULT_SOURCE_ALIASES)
        combined.update(catalog_aliases)
        if custom_aliases is not None:
            combined.update(custom_aliases)

        aliases: dict[str, frozenset[str]] = {}
        for legal_source, source_names in combined.items():
            if isinstance(source_names, str):
                raise TypeError("source alias values must be iterables of strings")
            normalized_names = frozenset(cls._normalize(name) for name in source_names)
            if not normalized_names or "" in normalized_names:
                raise ValueError("source aliases must contain non-empty strings")
            aliases[cls._normalize(legal_source)] = normalized_names
        return aliases

    @classmethod
    def _build_values(
        cls,
        defaults: Mapping[str, Any],
        catalog_values: Mapping[str, Any],
        custom_values: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        combined = dict(defaults)
        combined.update(catalog_values)
        if custom_values is not None:
            combined.update(custom_values)
        return {cls._normalize(key): value for key, value in combined.items()}

    @staticmethod
    def _validate_reference(legal_source: str, citation: str, limit: int) -> None:
        if not isinstance(legal_source, str) or not legal_source.strip():
            raise ValueError("legal_source must be a non-empty string")
        if not isinstance(citation, str) or not citation.strip():
            raise ValueError("citation must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")

    @staticmethod
    def _metadata_reference(record: _VectorStoreRecord, field_name: str) -> str:
        value = record.metadata.get(field_name)
        if value is None or str(value).casefold() == "none":
            return ""
        return str(value).strip().upper()

    @staticmethod
    def _normalize(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).strip().casefold().split())
