"""Versioned metadata foundation for future legal corpus builds."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re
import unicodedata

from src.assessment.evidence.models import AuthorityLevel
from src.assessment.models import SerializableModel


CORPUS_METADATA_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True, slots=True)
class CorpusMetadataV2(SerializableModel):
    """Instrument-aware metadata emitted by a future corpus v2 build."""

    instrument_id: str
    document_version: str
    canonical_citation: str
    authority_level: AuthorityLevel
    source_record_id: str
    stable_evidence_id: str
    metadata_schema_version: str = CORPUS_METADATA_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "instrument_id",
            "document_version",
            "canonical_citation",
            "source_record_id",
            "stable_evidence_id",
            "metadata_schema_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.authority_level, AuthorityLevel):
            raise TypeError("authority_level must be an AuthorityLevel")
        if self.metadata_schema_version != CORPUS_METADATA_SCHEMA_VERSION:
            raise ValueError(
                "metadata_schema_version must be "
                f"{CORPUS_METADATA_SCHEMA_VERSION!r}"
            )

    @classmethod
    def from_excerpt(
        cls,
        *,
        instrument_id: str,
        document_version: str,
        canonical_citation: str,
        authority_level: AuthorityLevel,
        excerpt: str,
    ) -> CorpusMetadataV2:
        """Create v2 metadata with reproducible content-based identities."""

        identity_digest = stable_evidence_digest(
            instrument_id=instrument_id,
            document_version=document_version,
            canonical_citation=canonical_citation,
            excerpt=excerpt,
        )
        return cls(
            instrument_id=_normalize_identity_value(
                instrument_id,
                field_name="instrument_id",
            ),
            document_version=_normalize_identity_value(
                document_version,
                field_name="document_version",
            ),
            canonical_citation=_normalize_identity_value(
                canonical_citation,
                field_name="canonical_citation",
            ),
            authority_level=authority_level,
            source_record_id=f"legal-chunk:v2:{identity_digest[:32]}",
            stable_evidence_id=f"evidence:v2:{identity_digest[:32]}",
        )


def normalize_evidence_excerpt(excerpt: str) -> str:
    """Normalize excerpt text for stable hashing without erasing punctuation."""

    if not isinstance(excerpt, str) or not excerpt.strip():
        raise ValueError("excerpt must be a non-empty string")
    normalized = unicodedata.normalize("NFC", excerpt)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_excerpt_hash(excerpt: str) -> str:
    """Return the full SHA-256 hash of a normalized legal excerpt."""

    normalized = normalize_evidence_excerpt(excerpt)
    return sha256(normalized.encode("utf-8")).hexdigest()


def stable_evidence_digest(
    *,
    instrument_id: str,
    document_version: str,
    canonical_citation: str,
    excerpt: str,
) -> str:
    """Hash the canonical v2 evidence identity components."""

    identity = [
        _normalize_identity_value(
            instrument_id,
            field_name="instrument_id",
        ),
        _normalize_identity_value(
            document_version,
            field_name="document_version",
        ),
        _normalize_identity_value(
            canonical_citation,
            field_name="canonical_citation",
        ),
        normalized_excerpt_hash(excerpt),
    ]
    canonical_identity = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(canonical_identity.encode("utf-8")).hexdigest()


def stable_evidence_id(
    *,
    instrument_id: str,
    document_version: str,
    canonical_citation: str,
    excerpt: str,
) -> str:
    """Return a namespaced stable evidence ID for v2 corpus records."""

    digest = stable_evidence_digest(
        instrument_id=instrument_id,
        document_version=document_version,
        canonical_citation=canonical_citation,
        excerpt=excerpt,
    )
    return f"evidence:v2:{digest[:32]}"


def _normalize_identity_value(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return " ".join(unicodedata.normalize("NFC", value).strip().split())
