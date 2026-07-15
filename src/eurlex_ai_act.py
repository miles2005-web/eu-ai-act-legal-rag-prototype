"""Isolated extraction of product-safety provisions from EUR-Lex AI Act HTML."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


AI_ACT_OFFICIAL_ELI = "https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng"
AI_ACT_CELEX = "32024R1689"


class EurLexAIActFormatError(ValueError):
    """Raised when saved EUR-Lex HTML lacks the required official structure."""


@dataclass(frozen=True, slots=True)
class EurLexAIActSource:
    """Validated official-source identity plus extracted legal units."""

    canonical_uri: str
    celex: str
    official_title: str
    units: dict[str, tuple[str, ...]]


class _TargetUnitParser(HTMLParser):
    """Capture normalized paragraph and table blocks from selected legal units."""

    _TARGETS = frozenset(("art_3", "art_6", "anx_I"))

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.canonical_uri: str | None = None
        self.document_title: str | None = None
        self.document_id: str | None = None
        self.div_depth = 0
        self.active_unit: str | None = None
        self.active_depth: int | None = None
        self.units: dict[str, list[str]] = {}
        self.table_depth = 0
        self.table_text: list[str] = []
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
        if normalized_tag == "meta":
            if attributes.get("name") == "WT.z_docTitle":
                self.document_title = attributes.get("content")
            elif attributes.get("name") == "WT.z_docID":
                self.document_id = attributes.get("content")

        if normalized_tag == "div":
            self.div_depth += 1
            element_id = attributes.get("id", "")
            if self.active_unit is None and element_id in self._TARGETS:
                self.active_unit = element_id
                self.active_depth = self.div_depth
                self.units[element_id] = []

        if self.active_unit is None:
            return
        if normalized_tag == "table":
            self.table_depth += 1
            if self.table_depth == 1:
                self.table_text = []
        elif normalized_tag in {"td", "th"} and self.table_depth > 0:
            # EUR-Lex uses adjacent table cells for point labels and text.
            # An explicit separator keeps minified/saved HTML equivalent to
            # the formatted Official Journal source.
            self.table_text.append(" ")
        elif normalized_tag == "p" and self.table_depth == 0:
            self.paragraph_depth += 1
            if self.paragraph_depth == 1:
                self.paragraph_text = []

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.casefold()
        if self.active_unit is not None:
            if normalized_tag == "table" and self.table_depth > 0:
                if self.table_depth == 1:
                    self._append_block("".join(self.table_text))
                    self.table_text = []
                self.table_depth -= 1
            elif (
                normalized_tag == "p"
                and self.table_depth == 0
                and self.paragraph_depth > 0
            ):
                if self.paragraph_depth == 1:
                    self._append_block("".join(self.paragraph_text))
                    self.paragraph_text = []
                self.paragraph_depth -= 1

        if normalized_tag == "div":
            if (
                self.active_unit is not None
                and self.active_depth == self.div_depth
            ):
                self.active_unit = None
                self.active_depth = None
                self.table_depth = 0
                self.table_text = []
                self.paragraph_depth = 0
                self.paragraph_text = []
            self.div_depth = max(0, self.div_depth - 1)

    def handle_data(self, data: str) -> None:
        if self.active_unit is None:
            return
        if self.table_depth > 0:
            self.table_text.append(data)
        elif self.paragraph_depth > 0:
            self.paragraph_text.append(data)

    def _append_block(self, value: str) -> None:
        normalized = normalize_eurlex_text(value)
        if normalized and self.active_unit is not None:
            self.units[self.active_unit].append(normalized)


def extract_ai_act_product_safety_units(
    path: str | Path,
) -> dict[str, tuple[str, ...]]:
    """Extract Article 3, Article 6 and Annex I from official English HTML."""

    return extract_ai_act_product_safety_source(path).units


def extract_ai_act_product_safety_source(
    path: str | Path,
) -> EurLexAIActSource:
    """Validate official identity and extract Article 3, Article 6 and Annex I."""

    source_path = Path(path)
    if source_path.suffix.casefold() not in {".html", ".htm"}:
        raise ValueError("EUR-Lex AI Act source must be an .html or .htm file")
    parser = _TargetUnitParser()
    parser.feed(source_path.read_text(encoding="utf-8"))
    parser.close()

    canonical = (parser.canonical_uri or "").rstrip("/")
    if canonical != AI_ACT_OFFICIAL_ELI.rstrip("/"):
        raise EurLexAIActFormatError(
            "EUR-Lex source canonical URI is not the English Official Journal "
            "text of Regulation (EU) 2024/1689"
        )
    if parser.document_id != AI_ACT_CELEX:
        raise EurLexAIActFormatError(
            f"EUR-Lex source CELEX is not {AI_ACT_CELEX}"
        )
    official_title = normalize_eurlex_text(parser.document_title or "")
    if not official_title.startswith("Regulation (EU) 2024/1689"):
        raise EurLexAIActFormatError(
            "EUR-Lex source has no valid official Regulation title"
        )
    missing = _TargetUnitParser._TARGETS.difference(parser.units)
    if missing:
        raise EurLexAIActFormatError(
            "EUR-Lex AI Act source is missing required units: "
            + ", ".join(sorted(missing))
        )
    empty = [unit for unit, blocks in parser.units.items() if not blocks]
    if empty:
        raise EurLexAIActFormatError(
            "EUR-Lex AI Act source contains empty required units: "
            + ", ".join(sorted(empty))
        )
    return EurLexAIActSource(
        canonical_uri=canonical,
        celex=parser.document_id,
        official_title=official_title,
        units={unit: tuple(blocks) for unit, blocks in parser.units.items()},
    )


def normalize_eurlex_text(value: str) -> str:
    """Normalize layout whitespace while preserving authoritative wording."""

    return " ".join(value.replace("\xa0", " ").split())


__all__ = [
    "AI_ACT_CELEX",
    "AI_ACT_OFFICIAL_ELI",
    "EurLexAIActSource",
    "EurLexAIActFormatError",
    "extract_ai_act_product_safety_units",
    "extract_ai_act_product_safety_source",
    "normalize_eurlex_text",
]
