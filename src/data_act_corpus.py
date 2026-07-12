"""Isolated candidate-corpus builder for Regulation (EU) 2023/2854.

This module does not interact with Chroma or the existing ``vector_store.json``.
It prepares deterministic metadata-v2 JSON records from an explicitly supplied
Data Act source.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any

from src.assessment.evidence.catalog import (
    LegalSource,
    LegalSourceCatalog,
    load_legal_source_catalog,
)
from src.corpus_enrichment import enrich_chunk_metadata_v2
from src.ingest import parse_document
from src.legal_chunks import normalize_legal_text


DATA_ACT_INSTRUMENT_ID = "EU_DATA_ACT"
DEFAULT_EXISTING_STORE_PATH = Path("vector_store.json")

_ARTICLE = re.compile(r"^Article\s+(\d+[a-z]?)$", re.IGNORECASE)
_ANNEX = re.compile(
    r"^(?:ANNEX|Annex)\s+([IVXLCDM]+|\d+[a-z]?)$"
)
_RECITAL = re.compile(r"^\((\d+[a-z]?)\)\s+(.+)$")
_PARAGRAPH = re.compile(
    r"^(?:\((\d+[a-z]?)\)|(\d+[a-z]?)\.)\s*(?:\|\s*)?(.*)$",
    re.IGNORECASE,
)
_POINT = re.compile(r"^\(([a-z])\)\s*(?:\|\s*)?(.*)$", re.IGNORECASE)
_STRUCTURAL_HEADING = re.compile(
    r"^(?:CHAPTER|Chapter|SECTION|Section)\s+[0-9IVXLCDM]+$"
)
_DEFINITION_TERM = re.compile(
    r"[‘'\"“]([^’'\"”]+)[’'\"”]\s+means\b",
    re.IGNORECASE,
)


class DataActCandidateBuildError(ValueError):
    """Raised when a source cannot safely produce a Data Act candidate."""


def build_data_act_candidate_records(
    source_path: str | Path,
    *,
    source_catalog: LegalSourceCatalog | None = None,
) -> list[dict[str, Any]]:
    """Parse one catalogued Data Act source into metadata-v2 JSON records."""

    path = Path(source_path)
    legal_source = _resolve_data_act_source(
        path,
        source_catalog or load_legal_source_catalog(),
    )
    text = parse_document(path)
    if not text.strip():
        raise DataActCandidateBuildError("Data Act source contains no text")

    chunks = parse_data_act_chunks(text, source_name=path.name)
    if not chunks:
        raise DataActCandidateBuildError(
            "Data Act source contains no supported legal citations"
        )

    records: list[dict[str, Any]] = []
    for chunk in chunks:
        enriched = enrich_chunk_metadata_v2(
            chunk,
            legal_source=legal_source,
        )
        document = enriched.pop("text")
        source_record_id = enriched["source_record_id"]
        records.append(
            {
                "id": source_record_id,
                "document": document,
                "metadata": enriched,
            }
        )
    _validate_candidate_records(records)
    return records


def write_data_act_candidate_corpus(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    existing_store_path: str | Path = DEFAULT_EXISTING_STORE_PATH,
) -> Path:
    """Write records to an explicit candidate path, never the active store."""

    output = Path(output_path)
    active_store = Path(existing_store_path)
    if output.resolve() == active_store.resolve():
        raise DataActCandidateBuildError(
            "candidate output must not overwrite the existing vector store"
        )
    if not records:
        raise DataActCandidateBuildError("candidate corpus must not be empty")
    _validate_candidate_records(records)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def parse_data_act_chunks(
    text: str,
    *,
    source_name: str,
) -> list[dict[str, Any]]:
    """Create atomic citation-bearing chunks from prepared Data Act text."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError("source_name must be a non-empty string")

    lines = [
        line.strip()
        for line in normalize_legal_text(text).splitlines()
        if line.strip()
    ]
    chunks: list[dict[str, Any]] = []
    current_kind: str | None = None
    current_ref: str | None = None
    current_lines: list[str] = []
    operative_text_started = False

    def flush() -> None:
        nonlocal current_kind, current_ref, current_lines
        if current_kind == "recital" and current_ref is not None:
            chunks.append(
                _base_chunk(
                    source_name=source_name,
                    text="\n".join(current_lines),
                    canonical_citation=f"Recital {current_ref}",
                    recital_ref=current_ref,
                )
            )
        elif current_kind == "article" and current_ref is not None:
            chunks.extend(
                _parse_article_chunks(
                    article_number=current_ref,
                    lines=current_lines,
                    source_name=source_name,
                )
            )
        elif current_kind == "annex" and current_ref is not None:
            chunks.append(
                _base_chunk(
                    source_name=source_name,
                    text="\n".join(current_lines),
                    canonical_citation=f"Annex {current_ref}",
                    annex_ref=current_ref,
                )
            )
        current_kind = None
        current_ref = None
        current_lines = []

    for line in lines:
        article_match = _ARTICLE.fullmatch(line)
        if article_match:
            flush()
            operative_text_started = True
            current_kind = "article"
            current_ref = _normalize_number(article_match.group(1))
            current_lines = [line]
            continue

        annex_match = _ANNEX.fullmatch(line)
        if annex_match:
            flush()
            operative_text_started = True
            current_kind = "annex"
            current_ref = annex_match.group(1).upper()
            current_lines = [line]
            continue

        recital_match = (
            _RECITAL.fullmatch(line) if not operative_text_started else None
        )
        if recital_match:
            flush()
            current_kind = "recital"
            current_ref = _normalize_number(recital_match.group(1))
            current_lines = [line]
            continue

        if _STRUCTURAL_HEADING.fullmatch(line):
            flush()
            operative_text_started = True
            continue

        if current_kind is not None:
            current_lines.append(line)

    flush()
    for index, chunk in enumerate(chunks, start=1):
        chunk["chunk_id"] = index
    return chunks


def _parse_article_chunks(
    *,
    article_number: str,
    lines: list[str],
    source_name: str,
) -> list[dict[str, Any]]:
    article_heading = f"Article {article_number}"
    body = list(lines)
    if body and _ARTICLE.fullmatch(body[0]):
        body.pop(0)

    article_title: str | None = None
    if body and not _PARAGRAPH.fullmatch(body[0]) and not _POINT.fullmatch(body[0]):
        article_title = body.pop(0)

    chunks: list[dict[str, Any]] = []
    paragraph_number: str | None = None
    paragraph_lines: list[str] = []
    point_label: str | None = None
    point_lines: list[str] = []

    def flush_point() -> None:
        nonlocal point_label, point_lines
        if point_label is None:
            return
        if paragraph_number is None:
            raise DataActCandidateBuildError(
                f"Article {article_number} point ({point_label}) has no paragraph"
            )
        chunks.append(
            _article_chunk(
                source_name=source_name,
                article_number=article_number,
                article_title=article_title,
                paragraph_number=paragraph_number,
                point_label=point_label,
                text="\n".join(point_lines),
            )
        )
        point_label = None
        point_lines = []

    def flush_paragraph() -> None:
        nonlocal paragraph_number, paragraph_lines
        flush_point()
        if paragraph_number is not None and paragraph_lines:
            chunks.append(
                _article_chunk(
                    source_name=source_name,
                    article_number=article_number,
                    article_title=article_title,
                    paragraph_number=paragraph_number,
                    text="\n".join(paragraph_lines),
                )
            )
        paragraph_number = None
        paragraph_lines = []

    leading_lines: list[str] = []
    for line in body:
        paragraph_match = _PARAGRAPH.fullmatch(line)
        if paragraph_match:
            flush_paragraph()
            paragraph_number = _normalize_number(
                paragraph_match.group(1) or paragraph_match.group(2)
            )
            paragraph_lines = [line]
            continue

        point_match = _POINT.fullmatch(line)
        if point_match:
            flush_point()
            if paragraph_number is None:
                raise DataActCandidateBuildError(
                    f"Article {article_number} contains a point before a paragraph"
                )
            if paragraph_lines:
                chunks.append(
                    _article_chunk(
                        source_name=source_name,
                        article_number=article_number,
                        article_title=article_title,
                        paragraph_number=paragraph_number,
                        text="\n".join(paragraph_lines),
                    )
                )
                paragraph_lines = []
            point_label = point_match.group(1).lower()
            point_lines = [line]
            continue

        if point_label is not None:
            point_lines.append(line)
        elif paragraph_number is not None:
            paragraph_lines.append(line)
        else:
            leading_lines.append(line)

    flush_paragraph()
    if leading_lines:
        chunks.insert(
            0,
            _article_chunk(
                source_name=source_name,
                article_number=article_number,
                article_title=article_title,
                text="\n".join([article_heading, *leading_lines]),
            ),
        )
    if not chunks:
        article_text = "\n".join(
            part for part in (article_heading, article_title) if part
        )
        chunks.append(
            _article_chunk(
                source_name=source_name,
                article_number=article_number,
                article_title=article_title,
                text=article_text,
            )
        )
    return chunks


def _article_chunk(
    *,
    source_name: str,
    article_number: str,
    text: str,
    article_title: str | None,
    paragraph_number: str | None = None,
    point_label: str | None = None,
) -> dict[str, Any]:
    citation = f"Article {article_number}"
    parent_citation: str | None = None
    if paragraph_number is not None:
        parent_citation = citation
        citation += f"({paragraph_number})"
    if point_label is not None:
        parent_citation = citation
        citation += f"({point_label})"

    definition_term = None
    if article_number == "2" and paragraph_number is not None:
        match = _DEFINITION_TERM.search(text)
        definition_term = match.group(1).strip() if match else None

    return _base_chunk(
        source_name=source_name,
        text=text,
        canonical_citation=citation,
        article_number=article_number,
        article_title=article_title,
        paragraph_number=paragraph_number,
        point_label=point_label,
        parent_citation=parent_citation,
        definition_term=definition_term,
    )


def _base_chunk(
    *,
    source_name: str,
    text: str,
    canonical_citation: str,
    article_number: str | None = None,
    article_title: str | None = None,
    paragraph_number: str | None = None,
    point_label: str | None = None,
    parent_citation: str | None = None,
    definition_term: str | None = None,
    recital_ref: str | None = None,
    annex_ref: str | None = None,
) -> dict[str, Any]:
    if not text.strip():
        raise DataActCandidateBuildError(
            f"{canonical_citation} produced an empty legal excerpt"
        )
    return {
        "source": source_name,
        "text": text.strip(),
        "canonical_citation": canonical_citation,
        "article_number": article_number,
        "article_title": article_title,
        "paragraph_number": paragraph_number,
        "point_label": point_label,
        "parent_citation": parent_citation,
        "definition_term": definition_term,
        "recital_ref": recital_ref,
        "annex_ref": annex_ref,
    }


def _resolve_data_act_source(
    path: Path,
    source_catalog: LegalSourceCatalog,
) -> LegalSource:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in {".txt", ".pdf"}:
        raise DataActCandidateBuildError(
            "Data Act candidate source must be a .txt or .pdf file"
        )
    alias = path.with_suffix(".txt").name
    legal_source = source_catalog.resolve_alias(alias)
    if legal_source is None:
        raise DataActCandidateBuildError(
            f"source filename {alias!r} is not catalogued"
        )
    if legal_source.instrument_id != DATA_ACT_INSTRUMENT_ID:
        raise DataActCandidateBuildError(
            f"source resolves to {legal_source.instrument_id}, not EU_DATA_ACT"
        )
    return legal_source


def _validate_candidate_records(records: list[dict[str, Any]]) -> None:
    record_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise DataActCandidateBuildError(f"record {index} is not an object")
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise DataActCandidateBuildError(
                f"record {index} has no metadata object"
            )
        if metadata.get("instrument_id") != DATA_ACT_INSTRUMENT_ID:
            raise DataActCandidateBuildError(
                f"record {index} is not EU_DATA_ACT metadata"
            )
        record_id = metadata.get("source_record_id")
        evidence_id = metadata.get("stable_evidence_id")
        if record.get("id") != record_id:
            raise DataActCandidateBuildError(
                f"record {index} id does not match source_record_id"
            )
        if record_id in record_ids or evidence_id in evidence_ids:
            raise DataActCandidateBuildError(
                f"record {index} has a duplicate stable identity"
            )
        record_ids.add(record_id)
        evidence_ids.add(evidence_id)


def _normalize_number(value: str) -> str:
    return value[:-1] + value[-1].upper() if value[-1].isalpha() else value
