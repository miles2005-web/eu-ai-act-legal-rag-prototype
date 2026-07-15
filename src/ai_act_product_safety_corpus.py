"""Isolated metadata-v2 corpus builder for AI Act Article 6(1)."""

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
from src.assessment.product_regulation import (
    AnnexIInstrument,
    AnnexIInstrumentCatalog,
    load_annex_i_instrument_catalog,
)
from src.corpus_enrichment import enrich_chunk_metadata_v2
from src.eurlex_ai_act import (
    AI_ACT_CELEX,
    AI_ACT_OFFICIAL_ELI,
    extract_ai_act_product_safety_source,
)


AI_ACT_INSTRUMENT_ID = "EU_AI_ACT"
AI_ACT_PRODUCT_SAFETY_RECORD_COUNT = 23
AI_ACT_PRODUCT_SAFETY_GENERATOR = "src.ai_act_product_safety_corpus"
AI_ACT_PRODUCT_SAFETY_GENERATOR_VERSION = "1.0.0"
DEFAULT_EXISTING_STORE_PATH = Path("vector_store.json")
DEFAULT_RUNTIME_EVIDENCE_PACK_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "legal_evidence"
    / "eu_ai_act_product_safety_metadata_v2.json"
)
DEFAULT_STABLE_ID_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "ai_act_product_safety_evidence_manifest.json"
)

_NUMBERED_ITEM = re.compile(r"^\((?P<number>\d+)\)\s+(?P<text>.+)$")
_LETTERED_ITEM = re.compile(r"^\((?P<point>[a-z])\)\s+(?P<text>.+)$", re.I)
_ANNEX_ITEM = re.compile(r"^(?P<point>\d+)\.\s+(?P<text>.+)$")
_SECTION_HEADING = re.compile(r"^Section\s+(?P<section>[AB])\.", re.I)


class AIActProductSafetyCorpusError(ValueError):
    """Raised when the official source cannot produce a safe candidate."""


def build_ai_act_product_safety_candidate_records(
    source_path: str | Path,
    *,
    source_catalog: LegalSourceCatalog | None = None,
    annex_catalog: AnnexIInstrumentCatalog | None = None,
) -> list[dict[str, Any]]:
    """Build 23 deterministic atomic records from the official AI Act HTML."""

    path = Path(source_path)
    legal_source = _resolve_ai_act_source(
        path,
        source_catalog or load_legal_source_catalog(),
    )
    instruments = annex_catalog or load_annex_i_instrument_catalog()
    official_source = extract_ai_act_product_safety_source(path)
    chunks = _atomic_product_safety_chunks(official_source.units, instruments)

    records: list[dict[str, Any]] = []
    for chunk in chunks:
        enriched = enrich_chunk_metadata_v2(chunk, legal_source=legal_source)
        document = enriched.pop("text")
        enriched.update(
            {
                "framework": AI_ACT_INSTRUMENT_ID,
                "official_instrument_identifier": legal_source.regulation_number,
                "celex": official_source.celex,
                "legal_reference": enriched["canonical_citation"],
                "official_title": official_source.official_title,
                "official_source_uri": official_source.canonical_uri,
                "source_version": legal_source.version,
                "jurisdiction": legal_source.jurisdiction,
                "language": legal_source.language,
                "authoritative_excerpt_hash": normalized_excerpt_hash(document),
                "source_isolation": "EU_AI_ACT_PRODUCT_SAFETY_ATOMIC_PACK",
                "generation_tool": AI_ACT_PRODUCT_SAFETY_GENERATOR,
                "generation_tool_version": (
                    AI_ACT_PRODUCT_SAFETY_GENERATOR_VERSION
                ),
                "record_provenance": {
                    "publisher": "EUR-Lex",
                    "celex": official_source.celex,
                    "official_uri": official_source.canonical_uri,
                    "source_filename": path.name,
                },
            }
        )
        records.append(
            {
                "id": enriched["source_record_id"],
                "document": document,
                "metadata": enriched,
            }
        )
    validate_ai_act_product_safety_candidate(records, instruments)
    return records


def write_ai_act_product_safety_candidate_corpus(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    existing_store_path: str | Path = DEFAULT_EXISTING_STORE_PATH,
) -> Path:
    """Write only to an explicit candidate path, never the active store."""

    output = Path(output_path)
    active_store = Path(existing_store_path)
    if output.resolve() == active_store.resolve():
        raise AIActProductSafetyCorpusError(
            "candidate output must not overwrite the existing vector store"
        )
    record_list = [dict(record) for record in records]
    validate_ai_act_product_safety_candidate(
        record_list,
        load_annex_i_instrument_catalog(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record_list, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def write_ai_act_product_safety_runtime_pack(
    records: Sequence[Mapping[str, Any]],
    output_path: str | Path = DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
) -> Path:
    """Write the reviewed embedding-free runtime pack deterministically."""

    record_list = [dict(record) for record in records]
    catalog = load_annex_i_instrument_catalog()
    validate_ai_act_product_safety_candidate(record_list, catalog)
    _validate_manifest_consistency(
        record_list,
        _load_manifest(DEFAULT_STABLE_ID_MANIFEST_PATH),
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record_list, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_ai_act_product_safety_runtime_pack(
    path: str | Path = DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
) -> list[dict[str, Any]]:
    """Load and validate the committed runtime pack and stable-ID manifest."""

    pack_path = Path(path)
    try:
        payload = json.loads(pack_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise AIActProductSafetyCorpusError(
            f"runtime Evidence pack is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise AIActProductSafetyCorpusError(
            "runtime Evidence pack root must be a JSON list"
        )
    records = [
        dict(record) if isinstance(record, Mapping) else record
        for record in payload
    ]
    if any(not isinstance(record, dict) for record in records):
        raise AIActProductSafetyCorpusError(
            "runtime Evidence pack records must be JSON objects"
        )
    validate_ai_act_product_safety_candidate(
        records,
        load_annex_i_instrument_catalog(),
    )
    _validate_manifest_consistency(
        records,
        _load_manifest(DEFAULT_STABLE_ID_MANIFEST_PATH),
    )
    return records


def compare_candidate_to_runtime_pack(
    candidate_records: Sequence[Mapping[str, Any]],
    runtime_pack_path: str | Path = DEFAULT_RUNTIME_EVIDENCE_PACK_PATH,
) -> None:
    """Fail when an official-source rebuild drifts from the runtime pack."""

    candidate = [dict(record) for record in candidate_records]
    validate_ai_act_product_safety_candidate(
        candidate,
        load_annex_i_instrument_catalog(),
    )
    runtime = load_ai_act_product_safety_runtime_pack(runtime_pack_path)
    candidate_by_citation = _records_by_citation(candidate)
    runtime_by_citation = _records_by_citation(runtime)
    if set(candidate_by_citation) != set(runtime_by_citation):
        raise AIActProductSafetyCorpusError(
            "candidate citations differ from the committed runtime pack"
        )
    compared_metadata = (
        "canonical_citation",
        "stable_evidence_id",
        "source_record_id",
        "authoritative_excerpt_hash",
        "instrument_id",
        "document_version",
        "annex_ref",
        "annex_section",
        "annex_point",
        "catalogue_instrument_id",
    )
    for citation, candidate_record in candidate_by_citation.items():
        runtime_record = runtime_by_citation[citation]
        if candidate_record.get("document") != runtime_record.get("document"):
            raise AIActProductSafetyCorpusError(
                f"authoritative excerpt drift for {citation}"
            )
        candidate_metadata = candidate_record["metadata"]
        runtime_metadata = runtime_record["metadata"]
        mismatches = [
            field_name
            for field_name in compared_metadata
            if candidate_metadata.get(field_name)
            != runtime_metadata.get(field_name)
        ]
        if mismatches:
            raise AIActProductSafetyCorpusError(
                f"runtime pack metadata drift for {citation}: "
                + ", ".join(mismatches)
            )


def validate_ai_act_product_safety_candidate(
    records: Sequence[Mapping[str, Any]],
    annex_catalog: AnnexIInstrumentCatalog,
) -> None:
    """Validate source isolation, identities and one-to-one catalogue mapping."""

    if len(records) != AI_ACT_PRODUCT_SAFETY_RECORD_COUNT:
        raise AIActProductSafetyCorpusError(
            "AI Act product-safety candidate must contain exactly "
            f"{AI_ACT_PRODUCT_SAFETY_RECORD_COUNT} atomic records"
        )
    citations: dict[str, Mapping[str, Any]] = {}
    record_ids: set[str] = set()
    evidence_ids: set[str] = set()
    for index, record in enumerate(records):
        metadata = record.get("metadata")
        document = record.get("document")
        if not isinstance(metadata, Mapping):
            raise AIActProductSafetyCorpusError(
                f"record {index} has no metadata object"
            )
        if not isinstance(document, str) or not document.strip():
            raise AIActProductSafetyCorpusError(
                f"record {index} has an empty authoritative excerpt"
            )
        if metadata.get("instrument_id") != AI_ACT_INSTRUMENT_ID:
            raise AIActProductSafetyCorpusError(
                f"record {index} is not isolated to EU_AI_ACT"
            )
        if metadata.get("framework") != AI_ACT_INSTRUMENT_ID:
            raise AIActProductSafetyCorpusError(
                f"record {index} has incorrect framework metadata"
            )
        if metadata.get("official_source_uri") != AI_ACT_OFFICIAL_ELI:
            raise AIActProductSafetyCorpusError(
                f"record {index} does not cite the official English source"
            )
        if metadata.get("celex") != AI_ACT_CELEX:
            raise AIActProductSafetyCorpusError(
                f"record {index} has incorrect CELEX metadata"
            )
        if metadata.get("language") != "en":
            raise AIActProductSafetyCorpusError(
                f"record {index} is not authoritative English Evidence"
            )
        if (
            metadata.get("generation_tool")
            != AI_ACT_PRODUCT_SAFETY_GENERATOR
            or metadata.get("generation_tool_version")
            != AI_ACT_PRODUCT_SAFETY_GENERATOR_VERSION
        ):
            raise AIActProductSafetyCorpusError(
                f"record {index} has invalid generation provenance"
            )
        citation = metadata.get("canonical_citation")
        if not isinstance(citation, str) or not citation:
            raise AIActProductSafetyCorpusError(
                f"record {index} has no canonical citation"
            )
        if citation in citations:
            raise AIActProductSafetyCorpusError(
                f"duplicate canonical citation {citation!r}"
            )
        citations[citation] = metadata
        record_id = metadata.get("source_record_id")
        evidence_id = metadata.get("stable_evidence_id")
        if record.get("id") != record_id:
            raise AIActProductSafetyCorpusError(
                f"record {index} id does not match source_record_id"
            )
        if record_id in record_ids or evidence_id in evidence_ids:
            raise AIActProductSafetyCorpusError(
                f"record {index} has a duplicate stable identity"
            )
        record_ids.add(str(record_id))
        evidence_ids.add(str(evidence_id))
        if metadata.get("authoritative_excerpt_hash") != normalized_excerpt_hash(
            document
        ):
            raise AIActProductSafetyCorpusError(
                f"record {index} has an incorrect excerpt hash"
            )
        try:
            expected_identity = CorpusMetadataV2.from_excerpt(
                instrument_id=str(metadata.get("instrument_id")),
                document_version=str(metadata.get("document_version")),
                canonical_citation=citation,
                authority_level=AuthorityLevel(
                    metadata.get("authority_level")
                ),
                excerpt=document,
            )
        except (TypeError, ValueError) as exc:
            raise AIActProductSafetyCorpusError(
                f"record {index} has invalid identity metadata: {exc}"
            ) from exc
        if (
            record_id != expected_identity.source_record_id
            or evidence_id != expected_identity.stable_evidence_id
        ):
            raise AIActProductSafetyCorpusError(
                f"record {index} stable identity does not match its content"
            )

    for citation in ("Article 3(14)", "Article 6(1)(a)", "Article 6(1)(b)"):
        if citation not in citations:
            raise AIActProductSafetyCorpusError(
                f"required atomic citation {citation!r} is missing"
            )

    expected_annex = {item.canonical_reference: item for item in annex_catalog.all()}
    actual_annex = {
        citation: metadata
        for citation, metadata in citations.items()
        if citation.startswith("Annex I,")
    }
    if set(actual_annex) != set(expected_annex):
        raise AIActProductSafetyCorpusError(
            "Annex I evidence citations do not match the catalogue one-to-one"
        )
    for citation, instrument in expected_annex.items():
        _validate_annex_metadata(actual_annex[citation], instrument)


def _atomic_product_safety_chunks(
    units: Mapping[str, tuple[str, ...]],
    annex_catalog: AnnexIInstrumentCatalog,
) -> list[dict[str, Any]]:
    article_3 = _find_numbered_item(units["art_3"], 14)
    article_6_a = _find_lettered_item(units["art_6"], "a")
    article_6_b = _find_lettered_item(units["art_6"], "b")
    chunks = [
        _chunk(
            "Article 3(14)",
            article_3,
            article_number="3",
            paragraph_number="14",
            definition_term="safety component",
        ),
        _chunk(
            "Article 6(1)(a)",
            article_6_a,
            article_number="6",
            paragraph_number="1",
            point_label="a",
        ),
        _chunk(
            "Article 6(1)(b)",
            article_6_b,
            article_number="6",
            paragraph_number="1",
            point_label="b",
        ),
    ]

    annex_items = _parse_annex_items(units["anx_I"])
    for instrument in annex_catalog.all():
        key = (instrument.annex_section.value, instrument.annex_point)
        try:
            excerpt = annex_items[key]
        except KeyError as exc:
            raise AIActProductSafetyCorpusError(
                f"official source is missing {instrument.canonical_reference}"
            ) from exc
        chunks.append(
            _chunk(
                instrument.canonical_reference,
                excerpt,
                annex_ref="I",
                annex_section=instrument.annex_section.value,
                annex_point=instrument.annex_point,
                catalogue_instrument_id=instrument.instrument_id,
                catalogue_instrument_type=instrument.instrument_type.value,
                catalogue_instrument_number=instrument.instrument_number,
                catalogue_official_title=instrument.official_title_en,
            )
        )
    return chunks


def _find_numbered_item(blocks: tuple[str, ...], number: int) -> str:
    for block in blocks:
        match = _NUMBERED_ITEM.fullmatch(block)
        if match and int(match.group("number")) == number:
            return block
    raise AIActProductSafetyCorpusError(
        f"official source is missing Article 3({number})"
    )


def _find_lettered_item(blocks: tuple[str, ...], point: str) -> str:
    paragraph_one_seen = False
    for block in blocks:
        if block.startswith("1."):
            paragraph_one_seen = True
            continue
        match = _LETTERED_ITEM.fullmatch(block)
        if (
            paragraph_one_seen
            and match
            and match.group("point").casefold() == point
        ):
            return block
    raise AIActProductSafetyCorpusError(
        f"official source is missing Article 6(1)({point})"
    )


def _parse_annex_items(
    blocks: tuple[str, ...],
) -> dict[tuple[str, int], str]:
    section: str | None = None
    items: dict[tuple[str, int], str] = {}
    for block in blocks:
        section_match = _SECTION_HEADING.match(block)
        if section_match:
            section = section_match.group("section").upper()
            continue
        item_match = _ANNEX_ITEM.fullmatch(block)
        if item_match and section is not None:
            point = int(item_match.group("point"))
            key = (section, point)
            if key in items:
                raise AIActProductSafetyCorpusError(
                    f"official source contains duplicate Annex I {section}/{point}"
                )
            items[key] = block
    return items


def _chunk(
    citation: str,
    excerpt: str,
    **metadata: Any,
) -> dict[str, Any]:
    return {
        "source": "Regulation_2024_1689.html",
        "text": excerpt,
        "canonical_citation": citation,
        **metadata,
    }


def _validate_annex_metadata(
    metadata: Mapping[str, Any],
    instrument: AnnexIInstrument,
) -> None:
    expected = {
        "annex_ref": "I",
        "annex_section": instrument.annex_section.value,
        "annex_point": instrument.annex_point,
        "catalogue_instrument_id": instrument.instrument_id,
        "catalogue_instrument_type": instrument.instrument_type.value,
        "catalogue_instrument_number": instrument.instrument_number,
        "catalogue_official_title": instrument.official_title_en,
    }
    mismatches = [
        key for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    ]
    if mismatches:
        raise AIActProductSafetyCorpusError(
            f"{instrument.canonical_reference} metadata disagrees with catalogue: "
            + ", ".join(mismatches)
        )


def _records_by_citation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    by_citation: dict[str, Mapping[str, Any]] = {}
    for record in records:
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise AIActProductSafetyCorpusError(
                "Evidence record has no metadata object"
            )
        citation = metadata.get("canonical_citation")
        if not isinstance(citation, str) or not citation:
            raise AIActProductSafetyCorpusError(
                "Evidence record has no canonical citation"
            )
        if citation in by_citation:
            raise AIActProductSafetyCorpusError(
                f"duplicate canonical citation {citation!r}"
            )
        by_citation[citation] = record
    return by_citation


def _load_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AIActProductSafetyCorpusError(
            f"stable-ID manifest cannot be loaded: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise AIActProductSafetyCorpusError(
            "stable-ID manifest root must be an object"
        )
    return payload


def _validate_manifest_consistency(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    if manifest.get("instrument_id") != AI_ACT_INSTRUMENT_ID:
        raise AIActProductSafetyCorpusError(
            "stable-ID manifest is not for EU_AI_ACT"
        )
    raw_manifest_records = manifest.get("records")
    if not isinstance(raw_manifest_records, list):
        raise AIActProductSafetyCorpusError(
            "stable-ID manifest records must be a list"
        )
    manifest_by_citation: dict[str, tuple[str, str]] = {}
    for item in raw_manifest_records:
        if not isinstance(item, Mapping):
            raise AIActProductSafetyCorpusError(
                "stable-ID manifest record must be an object"
            )
        citation = item.get("canonical_citation")
        evidence_id = item.get("stable_evidence_id")
        excerpt_hash = item.get("authoritative_excerpt_hash")
        if (
            not isinstance(citation, str)
            or not isinstance(evidence_id, str)
            or not isinstance(excerpt_hash, str)
        ):
            raise AIActProductSafetyCorpusError(
                "stable-ID manifest record is incomplete"
            )
        if citation in manifest_by_citation:
            raise AIActProductSafetyCorpusError(
                f"stable-ID manifest duplicates {citation!r}"
            )
        manifest_by_citation[citation] = (evidence_id, excerpt_hash)
    record_identities = {
        citation: (
            record["metadata"].get("stable_evidence_id"),
            record["metadata"].get("authoritative_excerpt_hash"),
        )
        for citation, record in _records_by_citation(records).items()
    }
    if manifest_by_citation != record_identities:
        raise AIActProductSafetyCorpusError(
            "stable-ID manifest differs from Evidence records"
        )


def _resolve_ai_act_source(
    path: Path,
    source_catalog: LegalSourceCatalog,
) -> LegalSource:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.casefold() not in {".html", ".htm"}:
        raise AIActProductSafetyCorpusError(
            "AI Act product-safety source must be official EUR-Lex HTML"
        )
    source = source_catalog.resolve_alias(path.name)
    if source is None:
        raise AIActProductSafetyCorpusError(
            f"source filename {path.name!r} is not catalogued"
        )
    if source.instrument_id != AI_ACT_INSTRUMENT_ID:
        raise AIActProductSafetyCorpusError(
            f"source resolves to {source.instrument_id}, not EU_AI_ACT"
        )
    return source


__all__ = [
    "AI_ACT_PRODUCT_SAFETY_RECORD_COUNT",
    "DEFAULT_RUNTIME_EVIDENCE_PACK_PATH",
    "AIActProductSafetyCorpusError",
    "build_ai_act_product_safety_candidate_records",
    "compare_candidate_to_runtime_pack",
    "load_ai_act_product_safety_runtime_pack",
    "validate_ai_act_product_safety_candidate",
    "write_ai_act_product_safety_candidate_corpus",
    "write_ai_act_product_safety_runtime_pack",
]
