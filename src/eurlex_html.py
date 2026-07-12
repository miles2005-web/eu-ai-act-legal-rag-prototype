"""Isolated EUR-Lex XHTML preprocessing for legal corpus candidates."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re


DATA_ACT_OFFICIAL_ELI = "https://eur-lex.europa.eu/eli/reg/2023/2854/oj/eng"
_SUPPORTED_UNIT_ID = re.compile(r"^(?:rct|art|anx)_", re.IGNORECASE)
_RECITAL_MARKER = re.compile(r"^\((\d+[a-z]?)\)$", re.IGNORECASE)
_ARTICLE_HEADING = re.compile(r"^Article\s+\d+[a-z]?$", re.IGNORECASE)
_ANNEX_HEADING = re.compile(
    r"^Annex\s+(?:[IVXLCDM]+|\d+[a-z]?)$",
    re.IGNORECASE,
)


class EurLexHTMLFormatError(ValueError):
    """Raised when saved EUR-Lex HTML lacks the expected legal structure."""


class _EurLexLegalTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.canonical_uri: str | None = None
        self.div_depth = 0
        self.active_unit_depth: int | None = None
        self.active_unit_id: str | None = None
        self.active_lines: list[str] = []
        self.units: list[tuple[str, list[str]]] = []
        self.paragraph_depth = 0
        self.paragraph_text: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = dict(attrs)
        normalized_tag = tag.casefold()
        if normalized_tag == "link" and attributes.get("rel") == "canonical":
            self.canonical_uri = attributes.get("href")

        if normalized_tag == "div":
            self.div_depth += 1
            element_id = attributes.get("id", "")
            classes = set((attributes.get("class") or "").split())
            if (
                self.active_unit_depth is None
                and "eli-subdivision" in classes
                and _SUPPORTED_UNIT_ID.match(element_id)
            ):
                self.active_unit_depth = self.div_depth
                self.active_unit_id = element_id
                self.active_lines = []

        if normalized_tag == "p" and self.active_unit_depth is not None:
            self.paragraph_depth += 1
            if self.paragraph_depth == 1:
                self.paragraph_text = []

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if (
            normalized_tag == "p"
            and self.active_unit_depth is not None
            and self.paragraph_depth > 0
        ):
            if self.paragraph_depth == 1:
                text = _normalize_html_text("".join(self.paragraph_text))
                if text:
                    self.active_lines.append(text)
                self.paragraph_text = []
            self.paragraph_depth -= 1

        if normalized_tag == "div":
            if (
                self.active_unit_depth is not None
                and self.div_depth == self.active_unit_depth
            ):
                self.units.append(
                    (self.active_unit_id or "", list(self.active_lines))
                )
                self.active_unit_depth = None
                self.active_unit_id = None
                self.active_lines = []
                self.paragraph_depth = 0
                self.paragraph_text = []
            self.div_depth = max(0, self.div_depth - 1)

    def handle_data(self, data: str) -> None:
        if self.active_unit_depth is not None and self.paragraph_depth > 0:
            self.paragraph_text.append(data)


def preprocess_eurlex_data_act_html(path: str | Path) -> str:
    """Convert saved EUR-Lex Data Act XHTML into structure-safe plain text."""

    source_path = Path(path)
    if source_path.suffix.casefold() not in {".html", ".htm"}:
        raise ValueError("EUR-Lex source must be an .html or .htm file")
    parser = _EurLexLegalTextParser()
    parser.feed(source_path.read_text(encoding="utf-8"))
    parser.close()

    canonical = (parser.canonical_uri or "").rstrip("/")
    expected = DATA_ACT_OFFICIAL_ELI.rstrip("/")
    if canonical != expected:
        raise EurLexHTMLFormatError(
            "EUR-Lex source canonical URI is not the English Official Journal "
            "Data Act text"
        )

    output_units: list[str] = []
    article_count = 0
    recital_count = 0
    for unit_id, lines in parser.units:
        if unit_id.casefold().startswith("rct_"):
            recital = _prepared_recital(lines)
            if recital is not None:
                output_units.append(recital)
                recital_count += 1
            continue

        if unit_id.casefold().startswith("art_"):
            if not lines or not _ARTICLE_HEADING.fullmatch(lines[0]):
                raise EurLexHTMLFormatError(
                    f"EUR-Lex unit {unit_id!r} has no Article heading"
                )
            output_units.append("\n".join(lines))
            article_count += 1
            continue

        if unit_id.casefold().startswith("anx_"):
            if lines and _ANNEX_HEADING.fullmatch(lines[0]):
                output_units.append("\n".join(lines))

    if recital_count == 0 or article_count == 0:
        raise EurLexHTMLFormatError(
            "EUR-Lex source contains no usable Recitals or Articles"
        )
    return "\n\n".join(output_units).strip()


def _prepared_recital(lines: list[str]) -> str | None:
    if len(lines) < 2 or not _RECITAL_MARKER.fullmatch(lines[0]):
        return None
    return f"{lines[0]} {' '.join(lines[1:])}"


def _normalize_html_text(value: str) -> str:
    return " ".join(value.replace("\xa0", " ").split())
