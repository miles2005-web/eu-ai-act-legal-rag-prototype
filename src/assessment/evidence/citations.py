"""Deterministic expansion of legal citation references for evidence lookup."""

from __future__ import annotations

import re


_ARTICLE_PARAGRAPH_RANGE = re.compile(
    r"^Article\s+(?P<article>\d+[a-z]?)"
    r"\(\s*(?P<start>\d+)\s*\)"
    r"\s*-\s*"
    r"\(\s*(?P<end>\d+)\s*\)$",
    re.IGNORECASE,
)
_ATOMIC_ARTICLE = re.compile(
    r"^(?:Article|Art\.)\s+(?P<article>\d+[a-z]?)"
    r"(?:\s*\(\s*(?P<paragraph>\d+[a-z]?)\s*\)"
    r"(?:\s*\(\s*(?P<point>[a-z])\s*\))?)?$",
    re.IGNORECASE,
)
_ATOMIC_ANNEX_POINT = re.compile(
    r"^Annex\s+(?P<annex>[ivxlcdm]+|\d+[a-z]?)\s*,?\s*"
    r"Section\s+(?P<section>[a-z])\s*,?\s*point\s+(?P<point>\d+)$",
    re.IGNORECASE,
)
_MAX_EXPANDED_CITATIONS = 100


def expand_citation_reference(citation: str) -> tuple[str, ...]:
    """Expand a supported range into atomic citations.

    Non-range references are returned unchanged. This preserves the authored
    legal basis while providing normalized lookup keys to retrieval and
    evidence binding layers.
    """

    if not isinstance(citation, str) or not citation.strip():
        raise ValueError("citation must be a non-empty string")
    authored_citation = citation.strip()
    match = _ARTICLE_PARAGRAPH_RANGE.fullmatch(authored_citation)
    if match is None:
        return (authored_citation,)

    article = _normalize_article_number(match.group("article"))
    start = int(match.group("start"))
    end = int(match.group("end"))
    if end < start:
        raise ValueError("citation range end must not precede its start")
    if end - start + 1 > _MAX_EXPANDED_CITATIONS:
        raise ValueError("citation range expands to too many references")
    return tuple(f"Article {article}({number})" for number in range(start, end + 1))


def normalize_atomic_citation(citation: str) -> str:
    """Normalize supported atomic references without fuzzy legal matching.

    Unknown or malformed references are returned with whitespace trimmed so
    callers fail closed on exact lookup rather than guessing a nearby Article
    or Annex point.
    """

    if not isinstance(citation, str) or not citation.strip():
        raise ValueError("citation must be a non-empty string")
    authored = " ".join(citation.strip().split())
    article = _ATOMIC_ARTICLE.fullmatch(authored)
    if article:
        normalized = f"Article {_normalize_article_number(article.group('article'))}"
        paragraph = article.group("paragraph")
        if paragraph is not None:
            normalized += f"({_normalize_article_number(paragraph)})"
        point = article.group("point")
        if point is not None:
            normalized += f"({point.casefold()})"
        return normalized

    annex = _ATOMIC_ANNEX_POINT.fullmatch(authored)
    if annex:
        return (
            f"Annex {annex.group('annex').upper()}, "
            f"Section {annex.group('section').upper()}, "
            f"point {int(annex.group('point'))}"
        )
    return authored


def is_strict_atomic_citation(citation: str) -> bool:
    """Return whether a citation requires an exact atomic corpus record."""

    normalized = normalize_atomic_citation(citation)
    article = _ATOMIC_ARTICLE.fullmatch(normalized)
    if article and article.group("paragraph") is not None:
        return True
    return _ATOMIC_ANNEX_POINT.fullmatch(normalized) is not None


def _normalize_article_number(value: str) -> str:
    return value[:-1] + value[-1].upper() if value[-1].isalpha() else value
