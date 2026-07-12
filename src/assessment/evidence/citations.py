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


def _normalize_article_number(value: str) -> str:
    return value[:-1] + value[-1].upper() if value[-1].isalpha() else value
