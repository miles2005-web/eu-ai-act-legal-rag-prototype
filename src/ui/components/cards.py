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


def render_assessment_summary_card(report: AssessmentReport) -> None:
    """Render report-wide metrics without privileging the first finding."""

    if not isinstance(report, AssessmentReport):
        raise TypeError("report must be an AssessmentReport")
    with st.container(border=True):
        st.markdown("#### Assessment overview")
        framework_count = len(report.assessed_frameworks)
        if framework_count == 0:
            framework_count = len(
                {finding.framework for finding in report.findings}
            )
        columns = st.columns(4)
        columns[0].metric("Frameworks assessed", framework_count)
        columns[1].metric("Findings", len(report.findings))
        columns[2].metric("Evidence records", len(report.evidence))
        columns[3].metric(
            "Missing information",
            len(report.missing_information),
        )
        st.caption(report.summary)


def render_framework_card(
    framework: RegulatoryFramework,
    findings: list[Finding],
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
        heading_column.markdown(f"#### {framework_label(framework)}")
        with badge_column:
            render_framework_badge(framework)
        st.caption(
            f"{len(findings)} finding(s) · {review_count} requiring legal review"
        )
        if not findings:
            st.write("No findings recorded for this framework.")
            return
        st.write(
            " · ".join(
                f"{status_label(status)}: {count}"
                for status, count in statuses.items()
            )
        )


def render_finding_card(
    finding: Finding,
    *,
    evidence_count: int | None = None,
) -> None:
    """Render one legal finding and its existing reasoning trace."""

    if not isinstance(finding, Finding):
        raise TypeError("finding must be a Finding")
    with st.container(border=True):
        title_column, status_column = st.columns([4, 1])
        title_column.markdown(f"### {finding.title}")
        title_column.caption(
            f"Rule {finding.rule_id or 'Not recorded'} · "
            f"Version {finding.rule_version or 'Not recorded'}"
        )
        with status_column:
            render_status_badge(finding.status)
        render_framework_badge(finding.framework, tone="neutral")
        st.write(finding.summary)

        metadata_columns = st.columns(3)
        metadata_columns[0].caption("Issue code")
        metadata_columns[0].code(finding.issue_code, language=None)
        metadata_columns[1].caption("Facts referenced")
        metadata_columns[1].write(str(len(finding.fact_refs)))
        metadata_columns[2].caption("Bound evidence")
        metadata_columns[2].write(
            str(evidence_count) if evidence_count is not None else "—"
        )

        st.markdown("**Legal basis**")
        if finding.legal_basis:
            for basis in finding.legal_basis:
                st.write(f"- {basis.citation} — `{basis.instrument}`")
        else:
            st.caption("No legal basis recorded.")

        if finding.requires_legal_review:
            st.warning(
                "Further legal review is required. This is a preliminary "
                "assessment, not a definitive legal conclusion."
            )

        with st.expander("Reasoning trace", expanded=False):
            if not finding.trace:
                st.caption("No reasoning trace recorded.")
            for index, trace_entry in enumerate(finding.trace, start=1):
                st.markdown(
                    f"**{index}. {trace_entry.description}**"
                )
                if trace_entry.result is not None:
                    st.write(f"Result: `{trace_entry.result}`")
                if trace_entry.fact_refs:
                    st.caption("Facts: " + ", ".join(trace_entry.fact_refs))


def render_evidence_trace_card(
    evidence: Evidence,
    *,
    requested_citation: str | None = None,
    expanded: bool = False,
) -> None:
    """Render one versioned Evidence record with trace metadata."""

    if not isinstance(evidence, Evidence):
        raise TypeError("evidence must be an Evidence")
    label = f"{evidence.citation} · {evidence.legal_source}"
    with st.expander(label, expanded=expanded):
        if requested_citation and requested_citation != evidence.citation:
            st.caption(
                f"Requested legal basis: {requested_citation} · "
                f"Resolved citation: {evidence.citation}"
            )
        columns = st.columns(2)
        columns[0].markdown(
            f"**Authority**  \n{evidence.authority_level.value}"
        )
        columns[1].markdown(
            "**Document version**  \n"
            f"{evidence.document_version or 'Not recorded'}"
        )
        st.markdown("**Authoritative excerpt**")
        st.write(evidence.excerpt)
        st.markdown("**Stable Evidence ID**")
        st.markdown(
            f'<span class="ui-mono">{escape(evidence.evidence_id)}</span>',
            unsafe_allow_html=True,
        )
