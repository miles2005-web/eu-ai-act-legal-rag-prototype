"""Enterprise-style cards for structured assessment output."""

from __future__ import annotations

from collections import Counter
from html import escape

import streamlit as st

from src.assessment.evidence import Evidence
from src.assessment.findings import Finding
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.report import AssessmentReport
from src.ui.components.common import (
    framework_label,
    render_framework_badge,
    render_status_badge,
    status_label,
)
from src.ui.i18n import DEFAULT_LANGUAGE, t, t_or


def render_assessment_summary_card(
    report: AssessmentReport,
    language: str = DEFAULT_LANGUAGE,
) -> None:
    """Render report-wide metrics without privileging the first finding."""

    if not isinstance(report, AssessmentReport):
        raise TypeError("report must be an AssessmentReport")
    with st.container(border=True):
        st.markdown(f'#### {t("component.overview", language)}')
        framework_count = len(report.assessed_frameworks)
        if framework_count == 0:
            framework_count = len(
                {finding.framework for finding in report.findings}
            )
        columns = st.columns(4)
        columns[0].metric(t("component.frameworks_assessed", language), framework_count)
        columns[1].metric(t("component.findings", language), len(report.findings))
        columns[2].metric(
            t("component.evidence_records", language),
            len(report.evidence),
        )
        columns[3].metric(
            t("component.missing_information", language),
            len(report.missing_information),
        )
        st.caption(report.summary)


def render_framework_card(
    framework: RegulatoryFramework,
    findings: list[Finding],
    language: str = DEFAULT_LANGUAGE,
) -> None:
    """Render a compact framework-level finding summary."""

    if not isinstance(framework, RegulatoryFramework):
        raise TypeError("framework must be a RegulatoryFramework")
    if any(not isinstance(finding, Finding) for finding in findings):
        raise TypeError("findings must contain Finding instances")
    statuses = Counter(finding.status for finding in findings)
    review_count = sum(
        1 for finding in findings if finding.requires_legal_review
    )
    with st.container(border=True):
        heading_column, badge_column = st.columns([4, 1])
        heading_column.markdown(f"#### {framework_label(framework, language)}")
        with badge_column:
            render_framework_badge(framework, language=language)
        st.caption(
            t(
                "component.framework_summary",
                language,
                findings=len(findings),
                reviews=review_count,
            )
        )
        if not findings:
            st.write(t("component.no_findings", language))
            return
        st.write(
            " · ".join(
                f"{status_label(status, language)}: {count}"
                for status, count in statuses.items()
            )
        )


def render_finding_card(
    finding: Finding,
    *,
    evidence_count: int | None = None,
    language: str = DEFAULT_LANGUAGE,
) -> None:
    """Render one legal finding and its existing reasoning trace."""

    if not isinstance(finding, Finding):
        raise TypeError("finding must be a Finding")
    with st.container(border=True):
        title_column, status_column = st.columns([4, 1])
        finding_key = f"finding.{finding.rule_id}.{finding.status.value}"
        title_column.markdown(
            f'### {t_or(f"{finding_key}.title", finding.title, language)}'
        )
        title_column.caption(
            f'{t("component.rule", language)} '
            f'{finding.rule_id or t("value.not_recorded", language)} · '
            f'{t("component.version", language)} '
            f'{finding.rule_version or t("value.not_recorded", language)}'
        )
        with status_column:
            render_status_badge(finding.status, language)
        render_framework_badge(
            finding.framework,
            tone="neutral",
            language=language,
        )
        st.write(t_or(f"{finding_key}.summary", finding.summary, language))

        metadata_columns = st.columns(3)
        metadata_columns[0].caption(t("component.issue_code", language))
        metadata_columns[0].code(finding.issue_code, language=None)
        metadata_columns[1].caption(t("component.facts_referenced", language))
        metadata_columns[1].write(str(len(finding.fact_refs)))
        metadata_columns[2].caption(t("component.bound_evidence", language))
        metadata_columns[2].write(
            str(evidence_count) if evidence_count is not None else "—"
        )

        st.markdown(f'**{t("component.legal_basis", language)}**')
        if finding.legal_basis:
            for basis in finding.legal_basis:
                instrument = t_or(
                    f"framework.{basis.instrument}",
                    basis.instrument,
                    language,
                )
                st.write(f"- {basis.citation} — {instrument}")
        else:
            st.caption(t("component.no_legal_basis", language))

        if finding.requires_legal_review:
            st.warning(
                t("component.review_required", language)
            )

        with st.expander(t("component.reasoning_trace", language), expanded=False):
            if not finding.trace:
                st.caption(t("component.no_trace", language))
            for index, trace_entry in enumerate(finding.trace, start=1):
                st.markdown(
                    f"**{index}. {trace_entry.description}**"
                )
                if trace_entry.result is not None:
                    st.write(
                        f'{t("component.result", language)}: '
                        f"`{trace_entry.result}`"
                    )
                if trace_entry.fact_refs:
                    st.caption(
                        f'{t("component.facts", language)}: '
                        + ", ".join(trace_entry.fact_refs)
                    )


def render_evidence_trace_card(
    evidence: Evidence,
    *,
    requested_citation: str | None = None,
    record_number: int = 1,
    expanded: bool = False,
    language: str = DEFAULT_LANGUAGE,
) -> None:
    """Render one versioned Evidence record with trace metadata."""

    if not isinstance(evidence, Evidence):
        raise TypeError("evidence must be an Evidence")
    if not isinstance(record_number, int) or isinstance(record_number, bool):
        raise TypeError("record_number must be an integer")
    if record_number <= 0:
        raise ValueError("record_number must be greater than zero")
    label = t("evidence.official_item", language, number=record_number)
    with st.expander(label, expanded=expanded):
        if requested_citation and requested_citation != evidence.citation:
            st.caption(
                f'{t("component.requested_basis", language)}: '
                f"{requested_citation} · "
                f'{t("component.resolved_citation", language)}: '
                f"{evidence.citation}"
            )
        columns = st.columns(2)
        columns[0].markdown(
            f'**{t("evidence.authority", language)}**  \n'
            f'{t(f"authority.{evidence.authority_level.value}", language)}'
        )
        columns[1].markdown(
            f'**{t("evidence.version", language)}**  \n'
            f'{evidence.document_version or t("evidence.not_recorded", language)}'
        )
        st.markdown(f'**{t("evidence.original", language)}**')
        st.write(evidence.excerpt)
        st.markdown(f'**{t("evidence.stable_id", language)}**')
        st.markdown(
            f'<span class="ui-mono">{escape(evidence.evidence_id)}</span>',
            unsafe_allow_html=True,
        )
