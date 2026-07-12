"""Small presentation primitives shared across assessment screens."""

from __future__ import annotations

from html import escape

import streamlit as st

from src.assessment.findings import FindingStatus
from src.assessment.frameworks import RegulatoryFramework


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
