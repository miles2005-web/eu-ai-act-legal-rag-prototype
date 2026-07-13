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


_FRAMEWORK_LABELS = {
    RegulatoryFramework.EU_AI_ACT: "EU AI Act",
    RegulatoryFramework.GDPR: "GDPR",
    RegulatoryFramework.EU_DATA_ACT: "EU Data Act",
    RegulatoryFramework.UNKNOWN: "Other framework",
}
_STATUS_TONES = {
    FindingStatus.APPLIES: "danger",
    FindingStatus.DOES_NOT_APPLY: "neutral",
    FindingStatus.POTENTIALLY_APPLIES: "warning",
    FindingStatus.UNDETERMINED: "warning",
    FindingStatus.NOT_ASSESSED: "neutral",
}
_FACT_LABELS = {
    "use_context.domain": "Employment context",
    "use_context.task": "AI system function",
    "use_context.materially_influences_decision": (
        "Material influence on decisions"
    ),
    "data_protection.personal_data_processed": "Personal data processing",
    "data_protection.automated_individual_decision": (
        "Automated individual decision"
    ),
    "data_protection.special_category_data_processed": (
        "Special-category data processing"
    ),
    "data_act.connected_product": "Connected product",
    "data_act.related_service": "Related service",
    "data_act.data_generated": "Product or service data generated",
    "data_act.data_holder_identified": "Data holder identified",
    "data_act.user_or_third_party_access_request": (
        "User or third-party data access request"
    ),
}
_RULE_LABELS = {
    "AI_ACT_HIGH_RISK_EMPLOYMENT": "Employment high-risk screening",
    "GDPR_ARTICLE22_RELEVANCE": "GDPR automated-decision relevance screening",
    "EU_DATA_ACT_RELEVANCE": "EU Data Act relevance screening",
}


def framework_label(framework: RegulatoryFramework) -> str:
    """Return a stable human-readable regulatory framework label."""

    if not isinstance(framework, RegulatoryFramework):
        raise TypeError("framework must be a RegulatoryFramework")
    return _FRAMEWORK_LABELS[framework]


def status_label(status: FindingStatus) -> str:
    """Return a restrained display label without changing legal meaning."""

    if not isinstance(status, FindingStatus):
        raise TypeError("status must be a FindingStatus")
    return status.value.replace("_", " ").title()


def status_tone(status: FindingStatus) -> str:
    """Map a legal finding status to a presentation-only semantic tone."""

    if not isinstance(status, FindingStatus):
        raise TypeError("status must be a FindingStatus")
    return _STATUS_TONES[status]


def fact_label(fact_path: str) -> str:
    """Return a compliance-user label for a raw AssessmentFacts path."""

    if not isinstance(fact_path, str) or not fact_path.strip():
        raise ValueError("fact_path must be a non-empty string")
    normalized = fact_path.strip()
    if normalized in _FACT_LABELS:
        return _FACT_LABELS[normalized]
    return normalized.rsplit(".", maxsplit=1)[-1].replace("_", " ").title()


def rule_label(rule_id: str | None) -> str:
    """Return a readable title for a versioned rule identifier."""

    if not rule_id:
        return "Assessment rule"
    return _RULE_LABELS.get(
        rule_id,
        rule_id.replace("_", " ").title(),
    )


def fact_value(facts: object, fact_path: str) -> str:
    """Read one fact path for presentation without mutating the fact model."""

    value: Any = facts
    try:
        for segment in fact_path.split("."):
            value = getattr(value, segment)
    except AttributeError:
        return "Not recorded"
    return _display_value(value)


def reasoning_state(result: str | None) -> tuple[str, str]:
    """Map a raw rule trace result to a visible state label and CSS tone."""

    if not result or "unknown" in result.casefold():
        return "Unknown", "unknown"
    normalized = result.casefold()
    if (
        normalized.startswith(("no_", "not_"))
        or "_not_" in normalized
        or "not_matched" in normalized
    ):
        return "Not matched", "not-matched"
    return "Matched", "matched"


def readable_result(result: str | None) -> str:
    """Humanize a raw rule result while preserving it elsewhere for audit."""

    if not result:
        return "No recorded result"
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


def _display_value(value: Any) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, TriState):
        return {
            TriState.YES: "Yes",
            TriState.NO: "No",
            TriState.UNKNOWN: "Unknown",
        }[value]
    if isinstance(value, Enum):
        return value.value.replace("_", " ").title()
    if isinstance(value, list):
        if not value:
            return "None recorded"
        return ", ".join(_display_value(item) for item in value)
    if isinstance(value, str):
        return value or "Not recorded"
    return str(value)


def render_framework_badge(
    framework: RegulatoryFramework,
    *,
    tone: str = "accent",
) -> None:
    """Render a compact framework badge."""

    _render_badge(framework_label(framework), tone=tone)


def render_status_badge(status: FindingStatus) -> None:
    """Render one finding status badge."""

    _render_badge(status_label(status), tone=status_tone(status))


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
