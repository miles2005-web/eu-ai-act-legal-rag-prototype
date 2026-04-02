from __future__ import annotations

import math
import re
from typing import Iterable


DEFAULT_MAX_CHARS = 1200
DEFAULT_MIN_CHARS = 350

OCR_PHRASE_REPLACEMENTS = {
    r"\bA\s+cting\b": "Acting",
    r"\bA\s+rtificial\b": "Artificial",
    r"\bAr\s+tif\s+icial\b": "Artificial",
    r"\bAr\s+ticle\b": "Article",
    r"\bAr\s+ticles\b": "Articles",
    r"\bC hapter\b": "Chapter",
    r"\bClassif\s+ication\b": "Classification",
    r"\bconf\s+or\s+mity\b": "conformity",
    r"\bDirec?tiv\s+es\b": "Directives",
    r"\bdo\s+wn\b": "down",
    r"\bEUR\s+OPEAN\b": "EUROPEAN",
    r"\bf\s+or\b": "for",
    r"\bf\s+ollo\s+wing\b": "following",
    r"\bP\s+ARLIAMENT\b": "PARLIAMENT",
    r"\bCO\s+UNCIL\b": "COUNCIL",
    r"\bhar\s+monised\b": "harmonised",
    r"\bhar\s+monisation\b": "harmonisation",
    r"\binte\s+lligence\b": "intelligence",
    r"\binte\s+nded\b": "intended",
    r"\binte\s+rnal\b": "internal",
    r"\bleg\s+al\b": "legal",
    r"\bmark\s+et\b": "market",
    r"\bpar\s+ticular\b": "particular",
    r"\bpur\s+pose\b": "purpose",
    r"\brefe\s+r\s+red\b": "referred",
    r"\br\s+ights\b": "rights",
    r"\br\s+isk\b": "risk",
    r"\br\s+ules\b": "rules",
    r"\bser\s+vice\b": "service",
    r"\bsys\s+tem\b": "system",
    r"\bsys\s+tems\b": "systems",
    r"\bsyste\s+m\b": "system",
    r"\bsyste\s+ms\b": "systems",
    r"\bte\s+xt\b": "text",
    r"\bT\s+reaty\b": "Treaty",
    r"\bthird-par\s+ty\b": "third-party",
    r"\btr\s+ustwor\s+thy\b": "trustworthy",
    r"\buni\s+form\b": "uniform",
    r"\bUni\s+on\b": "Union",
    r"\bwat\s+er\s+marks\b": "water marks",
    r"\bidentifica\s+tions\b": "identifications",
    r"\bcr\s+ypt\s+ographic\b": "cryptographic",
    r"\bprovin\s+g\b": "proving",
    r"\bprove\s+nance\b": "provenance",
    r"\bcont\s+ent\b": "content",
}


def normalize_legal_text(text: str) -> str:
    """
    Lightly clean OCR-style spacing and repeated page noise.

    This is intentionally conservative. It fixes repeated header/footer noise
    and a small set of high-confidence OCR breaks without changing the
    legal structure-aware chunking logic.
    """
    replacements = {
        r"\bAr\s+ticle\b": "Article",
        r"\bar\s+ticle\b": "article",
        r"\bAnn\s+ex\b": "Annex",
        r"\bann\s+ex\b": "annex",
        r"\bRec\s+ital\b": "Recital",
        r"\bCHAPT\s+ER\b": "CHAPTER",
        r"\bChap\s+ter\b": "Chapter",
        r"\bSECTI\s+ON\b": "SECTION",
        r"\bSect\s+ion\b": "Section",
        r"\bsy\s+stems\b": "systems",
        r"\bhigh-r\s+isk\b": "high-risk",
    }

    cleaned = clean_extracted_text(text)
    for pattern, replacement in replacements.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def clean_extracted_text(text: str) -> str:
    """Remove repeated page noise and repair a small set of OCR word breaks."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    kept_lines: list[str] = []

    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if _is_page_noise_line(line):
            continue
        kept_lines.append(line)

    cleaned = "\n".join(kept_lines)

    for pattern, replacement in OCR_PHRASE_REPLACEMENTS.items():
        cleaned = re.sub(pattern, replacement, cleaned)

    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def tokenize(text: str) -> list[str]:
    """Turn text into lowercase word tokens."""
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def extract_query_metadata(query: str) -> dict[str, str | None]:
    """Detect simple legal references in the user's question."""
    normalized_query = normalize_legal_text(query)

    article_match = re.search(r"\barticle\s+(\d+[a-z]?)\b", normalized_query, re.IGNORECASE)
    annex_match = re.search(r"\bannex\s+([ivxlcdm]+|\d+[a-z]?)\b", normalized_query, re.IGNORECASE)
    recital_match = re.search(
        r"\brecital\s+(\d+[a-z]?)\b|\((\d+[a-z]?)\)",
        normalized_query,
        re.IGNORECASE,
    )

    recital_number = None
    if recital_match:
        recital_number = recital_match.group(1) or recital_match.group(2)

    return {
        "article_number": _normalize_ref(article_match.group(1) if article_match else None),
        "annex_ref": _normalize_ref(annex_match.group(1) if annex_match else None),
        "recital_ref": _normalize_ref(recital_number),
    }


def is_provider_obligations_query(query: str) -> bool:
    """
    Detect broad questions about provider obligations for high-risk AI systems.

    This stays intentionally lightweight and rule-based.
    """
    normalized_query = normalize_legal_text(query).lower()

    provider_terms = ("provider", "providers")
    obligations_terms = (
        "obligation",
        "obligations",
        "requirement",
        "requirements",
        "duty",
        "duties",
        "must",
        "need to",
        "have to",
        "compliance",
    )
    high_risk_terms = ("high-risk", "high risk")

    return (
        any(term in normalized_query for term in provider_terms)
        and any(term in normalized_query for term in obligations_terms)
        and any(term in normalized_query for term in high_risk_terms)
    )


def build_structured_chunks(
    text: str,
    source: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[dict]:
    """
    Split a legal text into chunks aligned to headings when possible.

    The parser tracks the current chapter, section, article, annex, and recital
    context and attaches simple metadata to every chunk.
    """
    normalized_text = normalize_legal_text(text)
    lines = [line.strip() for line in normalized_text.splitlines()]

    blocks: list[dict] = []
    current_lines: list[str] = []
    context = _empty_context()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        boundary_info = _parse_boundary(line, context)
        if boundary_info:
            if current_lines:
                blocks.append(_build_block(current_lines, context))
                current_lines = []
            context = _updated_context(context, boundary_info)

        current_lines.append(line)

    if current_lines:
        blocks.append(_build_block(current_lines, context))

    chunks: list[dict] = []
    chunk_id = 1

    for block in blocks:
        for piece in _split_block_text(block["text"], max_chars=max_chars, min_chars=min_chars):
            chunks.append(
                {
                    "source": source,
                    "chunk_id": chunk_id,
                    "text": piece,
                    "chapter_heading": block["chapter_heading"],
                    "section_heading": block["section_heading"],
                    "article_number": block["article_number"],
                    "annex_ref": block["annex_ref"],
                    "recital_ref": block["recital_ref"],
                }
            )
            chunk_id += 1

    return chunks


def score_chunk(query: str, chunk: dict) -> float:
    """
    Score a chunk by keyword overlap plus simple metadata preferences.

    Metadata matches are intentionally strong so queries like "Article 6"
    or "Annex III" prefer the structurally correct chunks.
    """
    query_tokens = tokenize(normalize_legal_text(query))
    chunk_tokens = tokenize(chunk["text"])

    if not query_tokens or not chunk_tokens:
        return 0.0

    query_words = set(query_tokens)
    chunk_words = set(chunk_tokens)
    overlap = query_words.intersection(chunk_words)

    if not overlap:
        return 0.0

    score = len(overlap) / math.sqrt(len(chunk_words))
    query_meta = extract_query_metadata(query)

    if query_meta["article_number"] and query_meta["article_number"] == chunk.get("article_number"):
        score += 4.0
        if chunk["text"].lstrip().lower().startswith(f"article {query_meta['article_number'].lower()}"):
            score += 0.35
    if query_meta["annex_ref"] and query_meta["annex_ref"] == chunk.get("annex_ref"):
        score += 4.0
        if chunk["text"].lstrip().lower().startswith(f"annex {query_meta['annex_ref'].lower()}"):
            score += 0.35
    if query_meta["recital_ref"] and query_meta["recital_ref"] == chunk.get("recital_ref"):
        score += 3.5
        if chunk["text"].lstrip().startswith(f"({query_meta['recital_ref']})"):
            score += 0.25

    if query_meta["article_number"] and chunk.get("article_number"):
        score += 0.4
    if query_meta["annex_ref"] and chunk.get("annex_ref"):
        score += 0.4
    if query_meta["recital_ref"] and chunk.get("recital_ref"):
        score += 0.3

    if is_provider_obligations_query(query):
        obligations_articles = {"9", "10", "11", "13", "14"}
        if chunk.get("article_number") in obligations_articles:
            score += 1.75

        chunk_text = chunk["text"].lower()
        if "high-risk ai system" in chunk_text or "high-risk ai systems" in chunk_text:
            score += 0.45
        if "provider" in chunk_text or "providers" in chunk_text:
            score += 0.35
        if "requirement" in chunk_text or "requirements" in chunk_text:
            score += 0.25

    return score


def _empty_context() -> dict[str, str | None]:
    return {
        "chapter_heading": None,
        "section_heading": None,
        "article_number": None,
        "annex_ref": None,
        "recital_ref": None,
    }


def _parse_boundary(
    line: str, context: dict[str, str | None]
) -> dict[str, str | None] | None:
    chapter_match = re.match(r"^(CHAPTER|Chapter)\s+([IVXLCDM]+)\s*$", line)
    if chapter_match:
        return {"type": "chapter", "value": f"Chapter {chapter_match.group(2)}"}

    section_match = re.match(r"^(SECTION|Section)\s+([0-9IVXLCDM]+)\s*$", line)
    if section_match:
        return {"type": "section", "value": f"Section {section_match.group(2)}"}

    article_match = re.match(r"^Article\s+(\d+[a-z]?)\s*$", line, re.IGNORECASE)
    if article_match:
        return {"type": "article", "value": _normalize_ref(article_match.group(1))}

    annex_match = re.match(r"^(ANNEX|Annex)\s+([IVXLCDM]+|\d+[a-z]?)\s*$", line)
    if annex_match:
        return {"type": "annex", "value": _normalize_ref(annex_match.group(2))}

    recital_allowed = (
        context["chapter_heading"] is None
        and context["section_heading"] is None
        and context["article_number"] is None
        and context["annex_ref"] is None
    )
    recital_match = re.match(r"^\((\d+[a-z]?)\)\s+.*$", line) if recital_allowed else None
    if recital_match:
        return {"type": "recital", "value": _normalize_ref(recital_match.group(1))}

    return None


def _updated_context(
    current: dict[str, str | None], boundary_info: dict[str, str | None]
) -> dict[str, str | None]:
    updated = dict(current)
    boundary_type = boundary_info["type"]
    boundary_value = boundary_info["value"]

    if boundary_type == "chapter":
        updated["chapter_heading"] = boundary_value
        updated["section_heading"] = None
        updated["article_number"] = None
        updated["annex_ref"] = None
        updated["recital_ref"] = None
    elif boundary_type == "section":
        updated["section_heading"] = boundary_value
        updated["article_number"] = None
        updated["annex_ref"] = None
        updated["recital_ref"] = None
    elif boundary_type == "article":
        updated["article_number"] = boundary_value
        updated["annex_ref"] = None
        updated["recital_ref"] = None
    elif boundary_type == "annex":
        updated["annex_ref"] = boundary_value
        updated["article_number"] = None
        updated["recital_ref"] = None
    elif boundary_type == "recital":
        updated["recital_ref"] = boundary_value
        updated["article_number"] = None
        updated["annex_ref"] = None

    return updated


def _build_block(lines: list[str], context: dict[str, str | None]) -> dict:
    text = "\n".join(_collapse_blank_lines(lines)).strip()
    return {
        "text": text,
        "chapter_heading": context["chapter_heading"],
        "section_heading": context["section_heading"],
        "article_number": context["article_number"],
        "annex_ref": context["annex_ref"],
        "recital_ref": context["recital_ref"],
    }


def _collapse_blank_lines(lines: Iterable[str]) -> list[str]:
    collapsed: list[str] = []
    previous_blank = False

    for line in lines:
        is_blank = not line
        if is_blank and previous_blank:
            continue
        collapsed.append(line)
        previous_blank = is_blank

    return collapsed


def _split_block_text(text: str, max_chars: int, min_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        return _split_long_text(text, max_chars=max_chars)

    pieces: list[str] = []
    current_piece = ""

    for paragraph in paragraphs:
        candidate = paragraph if not current_piece else f"{current_piece}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            current_piece = candidate
            continue

        if current_piece:
            pieces.append(current_piece)
            current_piece = paragraph
        else:
            pieces.extend(_split_long_text(paragraph, max_chars=max_chars))
            current_piece = ""

    if current_piece:
        pieces.append(current_piece)

    return _merge_small_pieces(pieces, max_chars=max_chars, min_chars=min_chars)


def _split_long_text(text: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current_piece = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        candidate = sentence if not current_piece else f"{current_piece} {sentence}"
        if len(candidate) <= max_chars:
            current_piece = candidate
            continue

        if current_piece:
            pieces.append(current_piece)

        if len(sentence) <= max_chars:
            current_piece = sentence
            continue

        start = 0
        while start < len(sentence):
            end = start + max_chars
            slice_end = sentence.rfind(" ", start, end)
            if slice_end <= start:
                slice_end = end
            pieces.append(sentence[start:slice_end].strip())
            start = slice_end

        current_piece = ""

    if current_piece:
        pieces.append(current_piece)

    return [piece for piece in pieces if piece]


def _merge_small_pieces(pieces: list[str], max_chars: int, min_chars: int) -> list[str]:
    merged: list[str] = []

    for piece in pieces:
        if merged and len(piece) < min_chars:
            candidate = f"{merged[-1]}\n\n{piece}"
            if len(candidate) <= max_chars:
                merged[-1] = candidate
                continue
        merged.append(piece)

    return merged


def _normalize_ref(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().upper()


def _is_page_noise_line(line: str) -> bool:
    if not line:
        return False

    page_noise_patterns = [
        r"^Offi\s*cial\s+Jour\s*nal$",
        r"^of\s+the\s+European\s+Union$",
        r"^EN$",
        r"^L\s+ser\s*ies$",
        r"^\d{4}/\d+$",
        r"^\d{1,2}\.\d{1,2}\.\d{4}$",
        r"^OJ\s+L,\s*\d{1,2}\.\d{1,2}\.\d{4}$",
        r"^\d+/\d+\s+ELI:\s*http://data\.europa\.eu/eli/reg/\d{4}/\d+/oj$",
        r"^ELI:\s*http://data\.europa\.eu/eli/reg/\d{4}/\d+/oj\s+\d+/\d+$",
    ]

    return any(re.match(pattern, line, re.IGNORECASE) for pattern in page_noise_patterns)
