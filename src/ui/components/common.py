"""Small presentation primitives shared across assessment screens."""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from html import escape
from typing import Any, Iterable

import streamlit as st

from src.assessment.evidence import Evidence
from src.assessment.findings import FindingStatus
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import TriState
from src.assessment.product_regulation import (
    AnnexIInstrumentNotFoundError,
    load_annex_i_instrument_catalog,
)
from src.ui.i18n import DEFAULT_LANGUAGE, t


_STATUS_TONES = {
    FindingStatus.APPLIES: "danger",
    FindingStatus.DOES_NOT_APPLY: "neutral",
    FindingStatus.POTENTIALLY_APPLIES: "warning",
    FindingStatus.UNDETERMINED: "warning",
    FindingStatus.NOT_ASSESSED: "neutral",
}


def framework_label(
    framework: RegulatoryFramework,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Return a stable human-readable regulatory framework label."""

    if not isinstance(framework, RegulatoryFramework):
        raise TypeError("framework must be a RegulatoryFramework")
    return t(f"framework.{framework.value}", language)


def status_label(
    status: FindingStatus,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Return a restrained display label without changing legal meaning."""

    if not isinstance(status, FindingStatus):
        raise TypeError("status must be a FindingStatus")
    return t(f"status.{status.value}", language)


def status_tone(status: FindingStatus) -> str:
    """Map a legal finding status to a presentation-only semantic tone."""

    if not isinstance(status, FindingStatus):
        raise TypeError("status must be a FindingStatus")
    return _STATUS_TONES[status]


def fact_label(fact_path: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Return a compliance-user label for a raw AssessmentFacts path."""

    if not isinstance(fact_path, str) or not fact_path.strip():
        raise ValueError("fact_path must be a non-empty string")
    normalized = fact_path.strip()
    translation_key = f"fact.{normalized}"
    translated = t(translation_key, language)
    if translated != translation_key:
        return translated
    return normalized.rsplit(".", maxsplit=1)[-1].replace("_", " ").title()


def rule_label(rule_id: str | None, language: str = DEFAULT_LANGUAGE) -> str:
    """Return a readable title for a versioned rule identifier."""

    if not rule_id:
        return t("navigation.assessment", language)
    translation_key = f"rule.{rule_id}"
    translated = t(translation_key, language)
    return translated if translated != translation_key else rule_id.replace(
        "_", " "
    ).title()


def fact_value(
    facts: object,
    fact_path: str,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Read one fact path for presentation without mutating the fact model."""

    value: Any = facts
    try:
        for segment in fact_path.split("."):
            value = getattr(value, segment)
    except AttributeError:
        return t("value.not_recorded", language)
    if (
        fact_path == "product_regulation.annex_i_instrument"
        and isinstance(value, str)
        and value
    ):
        try:
            instrument = load_annex_i_instrument_catalog().get(value)
        except AnnexIInstrumentNotFoundError:
            return t("value.not_recorded", language)
        return (
            f"{instrument.display_label(language)} — "
            f"{instrument.canonical_reference}"
        )
    return _display_value(value, language)


def reasoning_state(
    result: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> tuple[str, str]:
    """Map a raw rule trace result to a visible state label and CSS tone."""

    if not result or "unknown" in result.casefold():
        return t("state.unknown", language), "unknown"
    normalized = result.casefold()
    if (
        normalized.startswith(("no_", "not_"))
        or "_not_" in normalized
        or "not_matched" in normalized
    ):
        return t("state.not_matched", language), "not-matched"
    return t("state.matched", language), "matched"


def readable_result(
    result: str | None,
    language: str = DEFAULT_LANGUAGE,
) -> str:
    """Humanize a raw rule result while preserving it elsewhere for audit."""

    if not result:
        return t("value.not_recorded", language)
    return result.replace("_", " ").replace(",", ", ").capitalize()


def group_evidence_by_citation(
    evidence_records: Iterable[Evidence],
) -> list[tuple[str, list[Evidence]]]:
    """Group Evidence by citation while preserving every record and order."""

    grouped: OrderedDict[str, list[Evidence]] = OrderedDict()
    for evidence in evidence_records:
        if not isinstance(evidence, Evidence):
            raise TypeError("evidence_records must contain Evidence instances")
        grouped.setdefault(evidence.citation, []).append(evidence)
    return list(grouped.items())


def _display_value(value: Any, language: str) -> str:
    if value is None:
        return t("value.not_recorded", language)
    if isinstance(value, TriState):
        return {
            TriState.YES: t("value.yes", language),
            TriState.NO: t("value.no", language),
            TriState.UNKNOWN: t("value.unknown", language),
        }[value]
    if isinstance(value, Enum):
        domain_key = f"domain.{value.value}"
        localized_domain = t(domain_key, language)
        if localized_domain != domain_key:
            return localized_domain
        return value.value.replace("_", " ").title()
    if isinstance(value, list):
        if not value:
            return t("value.none_recorded", language)
        return ", ".join(_display_value(item, language) for item in value)
    if isinstance(value, str):
        return value or t("value.not_recorded", language)
    return str(value)


def render_framework_badge(
    framework: RegulatoryFramework,
    *,
    tone: str = "accent",
    language: str = DEFAULT_LANGUAGE,
) -> None:
    """Render a compact framework badge."""

    _render_badge(framework_label(framework, language), tone=tone)


def render_status_badge(
    status: FindingStatus,
    language: str = DEFAULT_LANGUAGE,
) -> None:
    """Render one finding status badge."""

    _render_badge(status_label(status, language), tone=status_tone(status))


def render_section_header(
    title: str,
    *,
    description: str | None = None,
    eyebrow: str | None = None,
) -> None:
    """Render a consistent section heading with optional supporting copy."""

    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    eyebrow_html = (
        f'<div class="ui-section-eyebrow">{escape(eyebrow.strip())}</div>'
        if eyebrow and eyebrow.strip()
        else ""
    )
    description_html = (
        '<p class="ui-section-description">'
        f"{escape(description.strip())}</p>"
        if description and description.strip()
        else ""
    )
    st.markdown(
        '<section class="ui-section-header">'
        f"{eyebrow_html}"
        f'<h2 class="ui-section-title">{escape(title.strip())}</h2>'
        f"{description_html}"
        "</section>",
        unsafe_allow_html=True,
    )


def _render_badge(label: str, *, tone: str) -> None:
    allowed_tones = {"neutral", "accent", "success", "warning", "danger"}
    normalized_tone = tone if tone in allowed_tones else "neutral"
    st.markdown(
        f'<span class="ui-badge ui-badge--{normalized_tone}">'
        f"{escape(label)}</span>",
        unsafe_allow_html=True,
    )
