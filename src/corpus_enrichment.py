"""Opt-in metadata-v2 enrichment for future legal corpus builds.

The existing ingestion and Chroma pipelines do not import this module, so
legacy builds retain their current output until a future pipeline explicitly
enables metadata-v2 enrichment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
import re
from typing import Any

from src.assessment.evidence.catalog import LegalSource
from src.assessment.evidence.corpus_metadata import CorpusMetadataV2


class CitationKind(str, Enum):
    """Supported structural citation types for primary EU legislation."""

    ARTICLE = "article"
    RECITAL = "recital"
    ANNEX = "annex"


@dataclass(frozen=True, slots=True)
class StructuredCitation:
    """Normalized components extracted from one canonical legal citation."""

    kind: CitationKind
    canonical_citation: str
    article_number: str | None = None
    paragraph_number: str | None = None
    point_label: str | None = None
    recital_ref: str | None = None
    annex_ref: str | None = None


_ARTICLE_CITATION = re.compile(
    r"^Article\s+(?P<article>\d+[a-z]?)"
    r"(?:"
    r"\s*\(\s*(?P<paragraph>\d+[a-z]?)\s*\)"
    r"(?:\s*\(\s*(?P<point>[a-z])\s*\))?"
    r"|\s*\(\s*(?P<direct_point>[a-z])\s*\)"
    r")?$",
    re.IGNORECASE,
)
_RECITAL_CITATION = re.compile(
    r"^Recital\s+(?P<recital>\d+[a-z]?)$",
    re.IGNORECASE,
)
_ANNEX_CITATION = re.compile(
    r"^Annex\s+(?P<annex>[ivxlcdm]+|\d+[a-z]?)$",
    re.IGNORECASE,
)


def parse_structured_citation(citation: str) -> StructuredCitation:
    """Parse and normalize one supported canonical legal citation.

    The parser is deliberately strict: it accepts only complete Article,
    Recital, or Annex references and rejects ranges or trailing prose. This
    keeps citation identity deterministic for metadata-v2 records.
    """

    if not isinstance(citation, str) or not citation.strip():
        raise ValueError("citation must be a non-empty string")
    normalized_input = " ".join(citation.strip().split())

    article_match = _ARTICLE_CITATION.fullmatch(normalized_input)
    if article_match:
        article_number = _normalize_number(article_match.group("article"))
        paragraph_number = _normalize_number(
            article_match.group("paragraph")
        )
        point_label = _normalize_point(
            article_match.group("point")
            or article_match.group("direct_point")
        )
        canonical = f"Article {article_number}"
        if paragraph_number is not None:
            canonical += f"({paragraph_number})"
        if point_label is not None:
            canonical += f"({point_label})"
        return StructuredCitation(
            kind=CitationKind.ARTICLE,
            canonical_citation=canonical,
            article_number=article_number,
            paragraph_number=paragraph_number,
            point_label=point_label,
        )

    recital_match = _RECITAL_CITATION.fullmatch(normalized_input)
    if recital_match:
        recital_ref = _normalize_number(recital_match.group("recital"))
        return StructuredCitation(
            kind=CitationKind.RECITAL,
            canonical_citation=f"Recital {recital_ref}",
            recital_ref=recital_ref,
        )

    annex_match = _ANNEX_CITATION.fullmatch(normalized_input)
    if annex_match:
        annex_ref = annex_match.group("annex").upper()
        return StructuredCitation(
            kind=CitationKind.ANNEX,
            canonical_citation=f"Annex {annex_ref}",
            annex_ref=annex_ref,
        )

    raise ValueError(
        "unsupported citation; expected Article N, Article N(P), "
        "Article N(P)(a), Article N(a), Recital N, or Annex N"
    )


def enrich_chunk_metadata_v2(
    chunk: Mapping[str, Any],
    *,
    legal_source: LegalSource,
) -> dict[str, Any]:
    """Return a copy of a structured chunk with complete metadata v2.

    Enrichment is intentionally explicit and non-mutating. The excerpt and
    canonical citation must already be final because both participate in the
    stable evidence identity.
    """

    if not isinstance(chunk, Mapping):
        raise TypeError("chunk must be a mapping")
    if not isinstance(legal_source, LegalSource):
        raise TypeError("legal_source must be a LegalSource")

    excerpt = chunk.get("text")
    if not isinstance(excerpt, str) or not excerpt.strip():
        raise ValueError("chunk text must be a non-empty string")

    raw_citation = chunk.get("canonical_citation")
    structured_citation = parse_structured_citation(raw_citation)
    corpus_metadata = CorpusMetadataV2.from_excerpt(
        instrument_id=legal_source.instrument_id,
        document_version=legal_source.version,
        canonical_citation=structured_citation.canonical_citation,
        authority_level=legal_source.authority_level,
        excerpt=excerpt,
    )

    enriched = dict(chunk)
    enriched.update(corpus_metadata.to_dict())
    enriched.update(_citation_metadata(structured_citation))
    return enriched


def enrich_chunks_metadata_v2(
    chunks: Iterable[Mapping[str, Any]],
    *,
    legal_source: LegalSource,
) -> list[dict[str, Any]]:
    """Enrich chunks in input order without changing the source objects."""

    if isinstance(chunks, (str, bytes, Mapping)):
        raise TypeError("chunks must be an iterable of chunk mappings")
    return [
        enrich_chunk_metadata_v2(chunk, legal_source=legal_source)
        for chunk in chunks
    ]


def _citation_metadata(citation: StructuredCitation) -> dict[str, str | None]:
    return {
        "canonical_citation": citation.canonical_citation,
        "article_number": citation.article_number,
        "paragraph_number": citation.paragraph_number,
        "point_label": citation.point_label,
        "recital_ref": citation.recital_ref,
        "annex_ref": citation.annex_ref,
    }


def _normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    return value[:-1] + value[-1].upper() if value[-1].isalpha() else value


def _normalize_point(value: str | None) -> str | None:
    return value.lower() if value is not None else None
