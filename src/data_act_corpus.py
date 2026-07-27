"""Isolated candidate-corpus builder for Regulation (EU) 2023/2854.

This module does not interact with Chroma or the existing ``vector_store.json``.
It prepares deterministic metadata-v2 JSON records from an explicitly supplied
Data Act source.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

from src.assessment.evidence.catalog import (
    LegalSource,
    LegalSourceCatalog,
    load_legal_source_catalog,
)
from src.assessment.evidence.corpus_metadata import (
    CorpusMetadataV2,
    normalized_excerpt_hash,
)
from src.assessment.evidence.models import AuthorityLevel
from src.corpus_enrichment import enrich_chunk_metadata_v2
from src.eurlex_html import (
    DATA_ACT_OFFICIAL_ELI,
    preprocess_eurlex_data_act_html,
)
from src.ingest import parse_document
from src.legal_chunks import normalize_legal_text


DATA_ACT_INSTRUMENT_ID = "EU_DATA_ACT"
DATA_ACT_CELEX = "32023R2854"
DATA_ACT_RELEVANCE_RECORD_COUNT = 3
DATA_ACT_RELEVANCE_CITATIONS = (
    "Article 1(1)(a)",
    "Article 2(5)",
    "Article 2(6)",
)
DATA_ACT_RELEVANCE_GENERATOR = "src.data_act_corpus"
DATA_ACT_RELEVANCE_GENERATOR_VERSION = "1.0.0"
DEFAULT_EXISTING_STORE_PATH = Path("vector_store.json")
DEFAULT_RUNTIME_EVIDENCE_PACK_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "legal_evidence"
    / "eu_data_act_relevance_metadata_v2.json"
)
DEFAULT_STABLE_ID_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "data_act_relevance_evidence_manifest.json"
)

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
    text = _read_data_act_source(path)
    if not text.strip():
        raise DataActCandidateBuildError("Data Act source contains no text")

    chunks = parse_data_act_chunks(text, source_name=path.name)
    if not chunks:
        raise DataActCandidateBuildError(
            "Data Act source contains no supported legal citations"
        )

    records: list[dict[str, Any]] = []
    official_html_source = path.suffix.casefold() in {".html", ".htm"}
    for chunk in chunks:
        enriched = enrich_chunk_metadata_v2(
            chunk,
            legal_source=legal_source,
        )
        document = enriched.pop("text")
        if official_html_source:
            enriched.update(
                {
                    "framework": DATA_ACT_INSTRUMENT_ID,
                    "official_instrument_identifier": (
                        legal_source.regulation_number
                    ),
                    "celex": DATA_ACT_CELEX,
                    "legal_reference": enriched["canonical_citation"],
                    "official_title": legal_source.title,
                    "official_source_uri": DATA_ACT_OFFICIAL_ELI,
                    "source_version": legal_source.version,
                    "jurisdiction": legal_source.jurisdiction,
                    "language": legal_source.language,
                    "authoritative_excerpt_hash": normalized_excerpt_hash(
                        document
                    ),
                    "source_isolation": "EU_DATA_ACT_OFFICIAL_ATOMIC_CORPUS",
                    "generation_tool": DATA_ACT_RELEVANCE_GENERATOR,
                    "generation_tool_version": (
                        DATA_ACT_RELEVANCE_GENERATOR_VERSION
                    ),
                    "record_provenance": {
                        "publisher": "EUR-Lex",
                        "celex": DATA_ACT_CELEX,
                        "official_uri": DATA_ACT_OFFICIAL_ELI,
                        "source_filename": path.name,
                    },
                }
            )
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


def select_data_act_relevance_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select the three authored runtime citations in deterministic order."""

    by_citation: dict[str, Mapping[str, Any]] = {}
    required = frozenset(DATA_ACT_RELEVANCE_CITATIONS)
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise DataActCandidateBuildError(
                "Evidence record has no metadata object"
            )
        citation = metadata.get("canonical_citation")
        if citation not in required:
            continue
        if citation in by_citation:
            raise DataActCandidateBuildError(
                f"candidate duplicates required runtime citation {citation!r}"
            )
        by_citation[str(citation)] = record
    missing = [
        citation
        for citation in DATA_ACT_RELEVANCE_CITATIONS
        if citation not in by_citation
    ]
    if missing:
        raise DataActCandidateBuildError(
            "Data Act candidate is missing required runtime citations: "
            + ", ".join(missing)
        )
    selected = [
        dict(by_citation[citation])
        for citation in DATA_ACT_RELEVANCE_CITATIONS
    ]
    validate_data_act_relevance_runtime_records(selected)
    return selected


def write_data_act_relevance_runtime_pack(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path = DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
) -> Path:
    """Write the reviewed embedding-free runtime pack deterministically."""

    selected = select_data_act_relevance_records(records)
    _validate_manifest_consistency(
        selected,
        _load_manifest(DEFAULT_STABLE_ID_MANIFEST_PATH),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_data_act_relevance_runtime_pack(
    path: str | Path = DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
) -> list[dict[str, Any]]:
    """Load and validate the committed Data Act runtime pack and manifest."""

    pack_path = Path(path)
    try:
        payload = json.loads(pack_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise DataActCandidateBuildError(
            f"runtime Evidence pack is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, list) or any(
        not isinstance(record, Mapping) for record in payload
    ):
        raise DataActCandidateBuildError(
            "runtime Evidence pack must be a JSON list of objects"
        )
    records = [dict(record) for record in payload]
    validate_data_act_relevance_runtime_records(records)
    _validate_manifest_consistency(
        records,
        _load_manifest(DEFAULT_STABLE_ID_MANIFEST_PATH),
    )
    return records


def compare_data_act_candidate_to_runtime_pack(
    candidate_records: Sequence[Mapping[str, Any]],
    runtime_pack_path: str | Path = DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
) -> None:
    """Fail when an official-source rebuild drifts from the runtime pack."""

    candidate = select_data_act_relevance_records(candidate_records)
    runtime = load_data_act_relevance_runtime_pack(runtime_pack_path)
    runtime_by_citation = _records_by_citation(runtime)
    compared_metadata = (
        "source",
        "canonical_citation",
        "stable_evidence_id",
        "source_record_id",
        "authoritative_excerpt_hash",
        "instrument_id",
        "framework",
        "document_version",
        "authority_level",
        "metadata_schema_version",
        "official_instrument_identifier",
        "celex",
        "legal_reference",
        "official_title",
        "official_source_uri",
        "source_version",
        "jurisdiction",
        "language",
        "source_isolation",
        "generation_tool",
        "generation_tool_version",
        "record_provenance",
        "article_number",
        "article_title",
        "paragraph_number",
        "point_label",
        "parent_citation",
        "definition_term",
    )
    for candidate_record in candidate:
        metadata = candidate_record["metadata"]
        citation = metadata["canonical_citation"]
        runtime_record = runtime_by_citation[citation]
        if candidate_record.get("document") != runtime_record.get("document"):
            raise DataActCandidateBuildError(
                f"authoritative excerpt drift for {citation}"
            )
        runtime_metadata = runtime_record["metadata"]
        mismatches = [
            field_name
            for field_name in compared_metadata
            if metadata.get(field_name) != runtime_metadata.get(field_name)
        ]
        if mismatches:
            raise DataActCandidateBuildError(
                f"runtime pack metadata drift for {citation}: "
                + ", ".join(mismatches)
            )


def validate_data_act_relevance_runtime_records(
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Validate exact citations, identities, provenance and pack isolation."""

    if len(records) != DATA_ACT_RELEVANCE_RECORD_COUNT:
        raise DataActCandidateBuildError(
            "Data Act relevance runtime pack must contain exactly "
            f"{DATA_ACT_RELEVANCE_RECORD_COUNT} atomic records"
        )
    by_citation = _records_by_citation(records)
    if tuple(by_citation) != DATA_ACT_RELEVANCE_CITATIONS:
        raise DataActCandidateBuildError(
            "Data Act relevance runtime citations or ordering are incorrect"
        )
    record_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for index, record in enumerate(records):
        metadata = record["metadata"]
        document = record.get("document")
        if not isinstance(document, str) or not document.strip():
            raise DataActCandidateBuildError(
                f"runtime record {index} has an empty excerpt"
            )
        expected_metadata = {
            "source": "Regulation_2023_2854.html",
            "instrument_id": DATA_ACT_INSTRUMENT_ID,
            "framework": DATA_ACT_INSTRUMENT_ID,
            "official_instrument_identifier": "Regulation (EU) 2023/2854",
            "document_version": "Regulation (EU) 2023/2854",
            "authority_level": "binding_legislation",
            "celex": DATA_ACT_CELEX,
            "official_title": "EU Data Act",
            "official_source_uri": DATA_ACT_OFFICIAL_ELI,
            "source_version": "Regulation (EU) 2023/2854",
            "jurisdiction": "EU",
            "language": "en",
            "metadata_schema_version": "2.0.0",
            "source_isolation": "EU_DATA_ACT_OFFICIAL_ATOMIC_CORPUS",
            "generation_tool": DATA_ACT_RELEVANCE_GENERATOR,
            "generation_tool_version": DATA_ACT_RELEVANCE_GENERATOR_VERSION,
        }
        mismatches = [
            key
            for key, expected in expected_metadata.items()
            if metadata.get(key) != expected
        ]
        if mismatches:
            raise DataActCandidateBuildError(
                f"runtime record {index} has invalid metadata: "
                + ", ".join(mismatches)
            )
        if "embedding" in record or "embedding" in metadata:
            raise DataActCandidateBuildError(
                f"runtime record {index} must not contain embeddings"
            )
        citation = metadata["canonical_citation"]
        if metadata.get("legal_reference") != citation:
            raise DataActCandidateBuildError(
                f"runtime record {index} has inconsistent legal_reference"
            )
        expected_structure = {
            "Article 1(1)(a)": {
                "article_number": "1",
                "paragraph_number": "1",
                "point_label": "a",
                "parent_citation": "Article 1(1)",
                "definition_term": None,
            },
            "Article 2(5)": {
                "article_number": "2",
                "paragraph_number": "5",
                "point_label": None,
                "parent_citation": "Article 2",
                "definition_term": "connected product",
            },
            "Article 2(6)": {
                "article_number": "2",
                "paragraph_number": "6",
                "point_label": None,
                "parent_citation": "Article 2",
                "definition_term": "related service",
            },
        }[citation]
        structural_mismatches = [
            key
            for key, expected in expected_structure.items()
            if metadata.get(key) != expected
        ]
        if structural_mismatches:
            raise DataActCandidateBuildError(
                f"runtime record {index} has invalid citation structure: "
                + ", ".join(structural_mismatches)
            )
        if metadata.get("authoritative_excerpt_hash") != normalized_excerpt_hash(
            document
        ):
            raise DataActCandidateBuildError(
                f"runtime record {index} has an incorrect excerpt hash"
            )
        try:
            expected_identity = CorpusMetadataV2.from_excerpt(
                instrument_id=DATA_ACT_INSTRUMENT_ID,
                document_version=metadata["document_version"],
                canonical_citation=citation,
                authority_level=AuthorityLevel(metadata["authority_level"]),
                excerpt=document,
            )
        except (TypeError, ValueError) as exc:
            raise DataActCandidateBuildError(
                f"runtime record {index} has invalid identity metadata: {exc}"
            ) from exc
        record_id = metadata.get("source_record_id")
        evidence_id = metadata.get("stable_evidence_id")
        if (
            record.get("id") != record_id
            or record_id != expected_identity.source_record_id
            or evidence_id != expected_identity.stable_evidence_id
        ):
            raise DataActCandidateBuildError(
                f"runtime record {index} stable identity does not match content"
            )
        if record_id in record_ids or evidence_id in evidence_ids:
            raise DataActCandidateBuildError(
                f"runtime record {index} has a duplicate stable identity"
            )
        record_ids.add(str(record_id))
        evidence_ids.add(str(evidence_id))
        provenance = metadata.get("record_provenance")
        expected_provenance = {
            "publisher": "EUR-Lex",
            "celex": DATA_ACT_CELEX,
            "official_uri": DATA_ACT_OFFICIAL_ELI,
            "source_filename": "Regulation_2023_2854.html",
        }
        if not isinstance(provenance, Mapping) or dict(provenance) != (
            expected_provenance
        ):
            raise DataActCandidateBuildError(
                f"runtime record {index} has invalid official provenance"
            )
        for value in _nested_strings(record):
            if value.startswith(("file://", "/Users/", "/home/")):
                raise DataActCandidateBuildError(
                    f"runtime record {index} contains a local filesystem path"
                )


def _records_by_citation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    by_citation: dict[str, Mapping[str, Any]] = {}
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise DataActCandidateBuildError("Evidence record has no metadata object")
        citation = metadata.get("canonical_citation")
        if not isinstance(citation, str) or not citation:
            raise DataActCandidateBuildError(
                "Evidence record has no canonical citation"
            )
        if citation in by_citation:
            raise DataActCandidateBuildError(
                f"duplicate canonical citation {citation!r}"
            )
        by_citation[citation] = record
    return by_citation


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataActCandidateBuildError(
            f"stable-ID manifest cannot be loaded: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise DataActCandidateBuildError(
            "stable-ID manifest root must be an object"
        )
    return payload


def _validate_manifest_consistency(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    expected_manifest_metadata = {
        "manifest_schema_version": "1.0.0",
        "instrument_id": DATA_ACT_INSTRUMENT_ID,
        "celex": DATA_ACT_CELEX,
        "document_version": "Regulation (EU) 2023/2854",
        "official_source_uri": DATA_ACT_OFFICIAL_ELI,
    }
    if any(
        manifest.get(key) != expected
        for key, expected in expected_manifest_metadata.items()
    ):
        raise DataActCandidateBuildError(
            "stable-ID manifest has invalid EU Data Act metadata"
        )
    raw_records = manifest.get("records")
    if not isinstance(raw_records, list):
        raise DataActCandidateBuildError(
            "stable-ID manifest records must be a list"
        )
    manifest_identities: dict[str, tuple[Any, ...]] = {}
    for item in raw_records:
        if not isinstance(item, Mapping):
            raise DataActCandidateBuildError(
                "stable-ID manifest record must be an object"
            )
        citation = item.get("canonical_citation")
        identity = (
            item.get("stable_evidence_id"),
            item.get("authoritative_excerpt_hash"),
            item.get("official_instrument_identifier"),
            item.get("document_version"),
        )
        if not isinstance(citation, str) or any(
            not isinstance(value, str) or not value for value in identity
        ):
            raise DataActCandidateBuildError(
                "stable-ID manifest record is incomplete"
            )
        if citation in manifest_identities:
            raise DataActCandidateBuildError(
                f"stable-ID manifest duplicates {citation!r}"
            )
        manifest_identities[citation] = identity
    record_identities = {
        citation: (
            record["metadata"].get("stable_evidence_id"),
            record["metadata"].get("authoritative_excerpt_hash"),
            record["metadata"].get("official_instrument_identifier"),
            record["metadata"].get("document_version"),
        )
        for citation, record in _records_by_citation(records).items()
    }
    if manifest_identities != record_identities:
        raise DataActCandidateBuildError(
            "stable-ID manifest differs from Evidence records"
        )


def _nested_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _nested_strings(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            yield from _nested_strings(item)


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
                if leading_lines:
                    chunks.append(
                        _article_chunk(
                            source_name=source_name,
                            article_number=article_number,
                            article_title=article_title,
                            text="\n".join(
                                [article_heading, *leading_lines]
                            ),
                        )
                    )
                    leading_lines = []
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
    if path.suffix.lower() not in {".txt", ".pdf", ".html", ".htm"}:
        raise DataActCandidateBuildError(
            "Data Act candidate source must be a .txt, .pdf, or EUR-Lex HTML file"
        )
    legal_source = source_catalog.resolve_alias(path.name)
    alias = path.name
    if legal_source is None and path.suffix.lower() in {".txt", ".pdf"}:
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


def _read_data_act_source(path: Path) -> str:
    if path.suffix.lower() in {".html", ".htm"}:
        return preprocess_eurlex_data_act_html(path)
    return parse_document(path)


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


__all__ = [
    "DATA_ACT_RELEVANCE_CITATIONS",
    "DATA_ACT_RELEVANCE_GENERATOR",
    "DATA_ACT_RELEVANCE_GENERATOR_VERSION",
    "DATA_ACT_RELEVANCE_RECORD_COUNT",
    "DEFAULT_RUNTIME_EVIDENCE_PACK_PATH",
    "DataActCandidateBuildError",
    "build_data_act_candidate_records",
    "compare_data_act_candidate_to_runtime_pack",
    "load_data_act_relevance_runtime_pack",
    "parse_data_act_chunks",
    "select_data_act_relevance_records",
    "validate_data_act_relevance_runtime_records",
    "write_data_act_candidate_corpus",
    "write_data_act_relevance_runtime_pack",
]
