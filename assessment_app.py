"""Minimal Streamlit interface for the structured assessment workflow."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import streamlit as st

from scripts.run_demo_assessment import (
    _assign_tri_state_values,
    build_assessment_facts,
    load_fixture,
)
from src.assessment.demo import AssessmentWorkflowBundle, create_assessment_workflow
from src.assessment.facts import UseDomain
from src.assessment.models import TriState
from src.assessment.report import AssessmentReport
from src.ui import apply_enterprise_styles
from src.ui.components import (
    render_section_header,
)
from src.ui.components.common import (
    fact_label,
    fact_value,
    framework_label,
    render_framework_badge,
    render_status_badge,
    group_evidence_by_citation,
    readable_result,
    reasoning_state,
    rule_label,
)


APP_TITLE = "EU Digital Regulation Assessment Platform"
PROJECT_ROOT = Path(__file__).resolve().parent
INDUSTRIAL_FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "industrial_ai_case.json"
)
VIEW_LANDING = "landing"
VIEW_WORKSPACE = "workspace"
VIEW_RESULTS = "results"
VIEW_EVIDENCE = "evidence"
DOMAIN_LABELS = {
    UseDomain.UNKNOWN: "Unknown / not yet provided",
    UseDomain.EMPLOYMENT: "Employment",
    UseDomain.BIOMETRICS: "Biometrics",
    UseDomain.CRITICAL_INFRASTRUCTURE: "Critical infrastructure",
    UseDomain.EDUCATION: "Education",
    UseDomain.ESSENTIAL_SERVICES: "Essential services",
    UseDomain.LAW_ENFORCEMENT: "Law enforcement",
    UseDomain.MIGRATION_ASYLUM_BORDER_CONTROL: "Migration, asylum, or border control",
    UseDomain.JUSTICE_DEMOCRATIC_PROCESSES: "Justice or democratic processes",
    UseDomain.PRODUCT_SAFETY: "Product safety",
    UseDomain.OTHER: "Other",
}
TRI_STATE_LABELS = {
    TriState.UNKNOWN: "Unknown / not yet provided",
    TriState.YES: "Yes",
    TriState.NO: "No",
}


def get_workflow_bundle() -> AssessmentWorkflowBundle:
    """Create one in-memory workflow bundle for the current UI session."""

    if "assessment_workflow_bundle" not in st.session_state:
        st.session_state.assessment_workflow_bundle = create_assessment_workflow()
    return st.session_state.assessment_workflow_bundle


def initialize_ui_state() -> None:
    """Initialize presentation-only navigation state."""

    st.session_state.setdefault("assessment_view", VIEW_LANDING)
    st.session_state.setdefault("demo_scenario", None)
    st.session_state.setdefault("selected_finding_id", None)


def navigate(view: str) -> None:
    """Navigate between workspace views without changing domain state."""

    st.session_state.assessment_view = view
    st.rerun()


def load_recruitment_demo(bundle: AssessmentWorkflowBundle) -> None:
    """Create a case from the existing pure-data recruitment demo fixture."""

    payload = load_fixture()
    assessment_case = bundle.case_service.create_case(
        payload["scenario"]["name"],
        description=payload["scenario"]["description"],
        facts=build_assessment_facts(payload["facts"]),
    )
    st.session_state.assessment_case_id = assessment_case.case_id
    st.session_state.assessment_report = None
    st.session_state.demo_loaded = True
    st.session_state.demo_scenario = "recruitment"
    st.session_state.assessment_view = VIEW_WORKSPACE
    st.rerun()


def load_industrial_demo(bundle: AssessmentWorkflowBundle) -> None:
    """Create a case from the existing industrial AI fixture."""

    payload = json.loads(INDUSTRIAL_FIXTURE_PATH.read_text(encoding="utf-8"))
    facts = build_assessment_facts(payload["facts"])
    _assign_tri_state_values(
        facts.data_protection,
        payload["facts"]["data_protection"],
    )
    _assign_tri_state_values(facts.data_act, payload["facts"]["data_act"])
    assessment_case = bundle.case_service.create_case(
        payload["scenario"]["name"],
        description=payload["scenario"]["description"],
        facts=facts,
    )
    st.session_state.assessment_case_id = assessment_case.case_id
    st.session_state.assessment_report = None
    st.session_state.demo_loaded = True
    st.session_state.demo_scenario = "industrial"
    st.session_state.assessment_view = VIEW_WORKSPACE
    st.rerun()


def render_sidebar(
    bundle: AssessmentWorkflowBundle,
    case_id: str | None,
    report: AssessmentReport | None,
) -> None:
    """Show presentation-only workflow progress for the current session."""

    with st.sidebar:
        st.markdown(
            """
            <div class="ui-product-mark">
              <div class="ui-product-mark__eyebrow">Legal engineering</div>
              <div class="ui-product-mark__name">EU Digital Regulation</div>
              <div class="ui-product-mark__meta">Assessment Platform · Prototype</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="ui-nav-label">Workspace</div>', unsafe_allow_html=True)
        current_view = st.session_state.get("assessment_view", VIEW_LANDING)
        if st.button(
            "Assessment",
            use_container_width=True,
            disabled=case_id is None,
            type="primary" if current_view == VIEW_WORKSPACE else "secondary",
        ):
            navigate(VIEW_WORKSPACE)
        if st.button(
            "Demo cases",
            use_container_width=True,
            type="primary" if current_view == VIEW_LANDING else "secondary",
        ):
            navigate(VIEW_LANDING)
        if st.button(
            "Findings",
            use_container_width=True,
            disabled=report is None,
            type="primary" if current_view == VIEW_RESULTS else "secondary",
        ):
            navigate(VIEW_RESULTS)
        if st.button(
            "Evidence trace",
            use_container_width=True,
            disabled=report is None,
            type="primary" if current_view == VIEW_EVIDENCE else "secondary",
        ):
            navigate(VIEW_EVIDENCE)

        st.markdown('<div class="ui-nav-label ui-nav-label--reference">Reference</div>', unsafe_allow_html=True)
        with st.expander("Regulatory frameworks", expanded=False):
            st.caption("EU AI Act · GDPR · EU Data Act")
            st.write("Deterministic screening with versioned legal authority.")
        with st.expander("Technical details", expanded=False):
            if case_id is None:
                st.caption("Create or open a case to inspect technical details.")
            else:
                assessment_case = bundle.case_service.get_case(case_id)
                st.caption("Case ID")
                st.code(assessment_case.case_id, language=None)
                st.caption(
                    f"Facts schema {assessment_case.current_facts.schema_version}"
                )
                if report is not None:
                    st.caption(
                        f"Report {report.report_version} · Engine {report.engine_version}"
                    )

        st.markdown('<div class="ui-progress-heading">Assessment progress</div>', unsafe_allow_html=True)
        case_complete = case_id is not None
        facts_complete = False
        if case_id is not None:
            facts = bundle.case_service.get_case(case_id).current_facts
            facts_complete = all(
                (
                    facts.use_context.domain is not UseDomain.UNKNOWN,
                    bool(facts.use_context.task),
                    facts.use_context.materially_influences_decision
                    is not TriState.UNKNOWN,
                )
            )

        statuses = (
            ("Case created", case_complete),
            ("Required facts provided", facts_complete),
            ("Assessment completed", report is not None),
        )
        for label, complete in statuses:
            state = "Complete" if complete else "Pending"
            icon = "✓" if complete else "○"
            st.markdown(
                '<div class="ui-progress-row">'
                f'<span aria-hidden="true">{icon}</span>'
                f'<span>{label}</span><span class="ui-progress-state">{state}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        st.progress(sum(complete for _, complete in statuses) / len(statuses))

        if case_id is not None and st.button("Start a new assessment"):
            st.session_state.assessment_workflow_bundle = (
                create_assessment_workflow()
            )
            st.session_state.assessment_case_id = None
            st.session_state.assessment_report = None
            st.session_state.demo_loaded = False
            st.session_state.demo_scenario = None
            st.session_state.selected_finding_id = None
            st.session_state.assessment_view = VIEW_LANDING
            st.rerun()


def render_case_creation(bundle: AssessmentWorkflowBundle) -> None:
    """Collect case identity data and create an in-memory assessment case."""

    st.markdown(
        """
        <section class="ui-hero">
          <div class="ui-hero__eyebrow">EU compliance intelligence</div>
          <h1 class="ui-hero__title">EU Digital Regulation<br>Assessment Platform</h1>
          <p class="ui-hero__summary">
            Turn system facts into preliminary, evidence-grounded assessments
            across Europe's core digital regulatory frameworks.
          </p>
          <div class="ui-framework-strip">
            <span class="ui-badge ui-badge--accent">EU AI Act</span>
            <span class="ui-badge ui-badge--neutral">GDPR</span>
            <span class="ui-badge ui-badge--neutral">EU Data Act</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Prototype only — this output does not constitute legal advice.")

    capability_columns = st.columns(4)
    capabilities = (
        (
            "01",
            "Establish the facts",
            "Capture legally relevant system, use-case, and data facts without inference.",
        ),
        (
            "02",
            "Run versioned rules",
            "Apply deterministic regulatory screening with explicit rule metadata.",
        ),
        (
            "03",
            "Ground every finding",
            "Resolve authoritative provisions from instrument-aware legal corpora.",
        ),
        (
            "04",
            "Review the trace",
            "Connect facts, findings, legal basis, and stable evidence identities.",
        ),
    )
    for column, (number, heading, description) in zip(
        capability_columns,
        capabilities,
        strict=True,
    ):
        with column:
            st.markdown(
                '<div class="ui-capability-block">'
                '<div class="ui-capability-number">'
                f"{number}</div>"
                '<div class="ui-capability-title">'
                f"{heading}</div>"
                '<p class="ui-capability-copy">'
                f"{description}</p></div>",
                unsafe_allow_html=True,
            )

    active_case_id = st.session_state.get("assessment_case_id")
    if active_case_id is not None:
        active_case = bundle.case_service.get_case(active_case_id)
        render_section_header(
            "Current case",
            eyebrow="Continue assessment",
        )
        with st.container(border=True):
            case_column, action_column = st.columns([4, 1])
            case_column.markdown(f"### {active_case.name}")
            case_column.write(
                active_case.description or "No case description provided."
            )
            with action_column:
                if st.button(
                    "Open workspace",
                    type="primary",
                    use_container_width=True,
                ):
                    navigate(VIEW_WORKSPACE)

    render_section_header(
        "Choose a demonstration scenario",
        eyebrow="Guided assessment",
        description=(
            "Start with a prepared factual record to explore the compliance "
            "workspace and report experience."
        ),
    )
    recruitment_column, industrial_column = st.columns(2)
    with recruitment_column:
        with st.container(border=True):
            st.markdown(
                """
                <div class="ui-demo-label">Employment · EU AI Act</div>
                <div class="ui-demo-title">Recruitment AI Screening</div>
                <div class="ui-demo-copy">
                  Assess an AI system that screens CVs, ranks candidates, and
                  materially influences access to employment.
                </div>
                <div class="ui-demo-meta">
                  Prepared facts · High-risk classification screening
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Open recruitment demo",
                type="primary",
                use_container_width=True,
            ):
                load_recruitment_demo(bundle)
    with industrial_column:
        with st.container(border=True):
            st.markdown(
                """
                <div class="ui-demo-label">Industrial data · EU Data Act</div>
                <div class="ui-demo-title">Industrial AI Monitoring</div>
                <div class="ui-demo-copy">
                  Assess connected machinery monitoring that generates operational
                  data requested by an external maintenance provider.
                </div>
                <div class="ui-demo-meta">
                  Prepared facts · Connected-product relevance screening
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "Open industrial demo",
                type="primary",
                use_container_width=True,
            ):
                load_industrial_demo(bundle)

    render_section_header(
        "Create a blank assessment",
        eyebrow="New case",
        description="Start with an empty factual record for your own AI system.",
    )
    with st.form("create_assessment_case"):
        name = st.text_input(
            "Case name",
            placeholder="Example: Recruitment screening system",
        )
        description = st.text_area(
            "Description",
            placeholder="Briefly describe the AI system and assessment purpose.",
        )
        submitted = st.form_submit_button("Create case", type="primary")

    if submitted:
        if not name.strip():
            st.error("Enter a case name before creating the case.")
            return
        assessment_case = bundle.case_service.create_case(
            name.strip(),
            description=description.strip() or None,
        )
        st.session_state.assessment_case_id = assessment_case.case_id
        st.session_state.assessment_report = None
        st.session_state.demo_loaded = False
        st.session_state.demo_scenario = None
        st.session_state.assessment_view = VIEW_WORKSPACE
        st.rerun()


def render_case_context(bundle: AssessmentWorkflowBundle, case_id: str) -> None:
    """Render the active case header and a concise system profile."""

    assessment_case = bundle.case_service.get_case(case_id)
    facts = assessment_case.current_facts
    st.caption("ACTIVE ASSESSMENT CASE")
    st.markdown(f"# {assessment_case.name}")
    if assessment_case.description:
        st.markdown(
            f'<p class="ui-case-description">{escape(assessment_case.description)}</p>',
            unsafe_allow_html=True,
        )

    report_ready = st.session_state.get("assessment_report") is not None
    status = "Report available" if report_ready else "Facts in progress"
    demo = st.session_state.get("demo_scenario")
    scenario = f"{demo.title()} demo" if demo else "Custom case"
    st.markdown(
        '<section class="ui-system-summary" aria-label="System summary">'
        '<div class="ui-system-summary__item">'
        '<span class="ui-system-summary__label">System</span>'
        f'<strong>{escape(facts.system.name or "Unnamed AI system")}</strong>'
        f'<span>{escape(facts.system.intended_purpose or "Purpose not yet provided.")}</span>'
        "</div>"
        '<div class="ui-system-summary__item">'
        '<span class="ui-system-summary__label">Use context</span>'
        f'<strong>{escape(DOMAIN_LABELS[facts.use_context.domain])}</strong>'
        f'<span>{escape(facts.use_context.task or "Task not yet provided.")}</span>'
        "</div>"
        '<div class="ui-system-summary__item">'
        '<span class="ui-system-summary__label">Assessment</span>'
        f"<strong>{status}</strong><span>{scenario}</span>"
        "</div></section>",
        unsafe_allow_html=True,
    )
    with st.expander("Technical details", expanded=False):
        detail_columns = st.columns(3)
        detail_columns[0].caption("Case ID")
        detail_columns[0].code(assessment_case.case_id, language=None)
        detail_columns[1].caption("Case schema")
        detail_columns[1].code(assessment_case.schema_version, language=None)
        detail_columns[2].caption("Facts schema")
        detail_columns[2].code(facts.schema_version, language=None)


def render_fact_collection(bundle: AssessmentWorkflowBundle, case_id: str) -> None:
    """Collect the facts required by the registered employment rule."""

    assessment_case = bundle.case_service.get_case(case_id)
    facts = assessment_case.current_facts

    render_section_header(
        "Provide assessment facts",
        eyebrow="Step 2",
        description=(
            "Unknown answers remain visible as missing information and do not "
            "create a legal finding."
        ),
    )
    if st.session_state.get("demo_loaded"):
        demo_name = (
            "Industrial AI Monitoring"
            if st.session_state.get("demo_scenario") == "industrial"
            else "Recruitment AI Screening"
        )
        st.info(
            f"{demo_name} demo loaded. Review or edit the populated "
            "facts before running the assessment."
        )
    domains = list(DOMAIN_LABELS)
    influence_options = list(TRI_STATE_LABELS)
    with st.form("assessment_facts"):
        domain = st.selectbox(
            "In which context is the AI system used?",
            options=domains,
            index=domains.index(facts.use_context.domain),
            format_func=DOMAIN_LABELS.get,
        )
        task = st.text_area(
            "What task does the AI system perform?",
            value=facts.use_context.task or "",
            placeholder=(
                "Example: Screens CVs and ranks candidates for recruiter review."
            ),
        )
        materially_influences = st.selectbox(
            "Does the AI output materially influence an employment-related decision?",
            options=influence_options,
            index=influence_options.index(
                facts.use_context.materially_influences_decision
            ),
            format_func=TRI_STATE_LABELS.get,
        )
        facts_submitted = st.form_submit_button("Save facts", type="primary")

    if facts_submitted:
        facts.use_context.domain = domain
        facts.use_context.task = task.strip() or None
        facts.use_context.materially_influences_decision = materially_influences
        bundle.case_service.update_facts(case_id, facts)
        st.session_state.assessment_report = None
        st.success("Facts saved.")


def render_assessment_action(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
) -> None:
    """Execute the existing assessment workflow on user request."""

    render_section_header(
        "Run assessment",
        eyebrow="Step 3",
        description=(
            "Apply the configured rules, resolve legal evidence, and build a "
            "traceable preliminary report."
        ),
    )
    if st.button("Run assessment", type="primary", use_container_width=True):
        with st.spinner("Running assessment and resolving legal evidence..."):
            st.session_state.assessment_report = bundle.workflow.run(case_id)
        st.session_state.assessment_view = VIEW_RESULTS
        st.rerun()


def evidence_for_finding(report: AssessmentReport, finding_id: str) -> list:
    """Return bound Evidence in report binding order for presentation."""

    binding = next(
        (
            item
            for item in report.evidence_bindings
            if item.finding_id == finding_id
        ),
        None,
    )
    evidence_by_id = {item.evidence_id: item for item in report.evidence}
    return [
        evidence_by_id[evidence_id]
        for evidence_id in (binding.evidence_refs if binding else [])
        if evidence_id in evidence_by_id
    ]


def render_decision_path(finding, facts) -> None:
    """Render a finding trace as human-readable decision stages."""

    st.markdown("### Decision path")
    if not finding.trace:
        st.caption("No reasoning stages were recorded for this finding.")
        return
    for index, entry in enumerate(finding.trace, start=1):
        label = fact_label(entry.fact_refs[0]) if entry.fact_refs else f"Condition {index}"
        state_label, state_tone = reasoning_state(entry.result)
        value = (
            fact_value(facts, entry.fact_refs[0])
            if entry.fact_refs
            else readable_result(entry.result)
        )
        st.markdown(
            '<div class="ui-decision-step">'
            '<div class="ui-decision-step__index">'
            f"{index:02d}</div>"
            '<div class="ui-decision-step__content">'
            '<div class="ui-decision-step__heading">'
            f"<strong>{escape(label)}</strong>"
            f'<span class="ui-state ui-state--{state_tone}">{state_label}</span>'
            "</div>"
            f'<div class="ui-decision-step__value">{escape(value)}</div>'
            f'<p>{escape(entry.description)}</p>'
            "</div></div>",
            unsafe_allow_html=True,
        )


def render_result_finding_card(
    finding,
    bound_evidence: list,
    facts,
    schema_version: str,
) -> None:
    """Present one finding in user-first compliance review order."""

    with st.container():
        status_column, framework_column, count_column = st.columns([2, 2, 3])
        with status_column:
            render_status_badge(finding.status)
        with framework_column:
            render_framework_badge(finding.framework, tone="neutral")
        count_column.markdown(
            '<div class="ui-finding-evidence-count">'
            f"{len(bound_evidence)} supporting record(s)</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="ui-finding-kicker">Preliminary conclusion</div>'
            f'<h2 class="ui-finding-title ui-finding-title--hero">'
            f"{escape(finding.title)}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="ui-finding-summary">{escape(finding.summary)}</p>',
            unsafe_allow_html=True,
        )

        if finding.requires_legal_review:
            st.warning(
                "Further legal review is required before treating this "
                "preliminary result as a final classification."
            )

        render_decision_path(finding, facts)

        st.markdown("### Legal authority")
        if finding.legal_basis:
            for basis in finding.legal_basis:
                st.markdown(
                    '<div class="ui-legal-basis-row">'
                    f'<span class="ui-legal-basis-citation">{escape(basis.citation)}</span>'
                    f'<span class="ui-legal-basis-instrument">{escape(basis.instrument)}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No legal basis recorded for this finding.")

        st.markdown("### Evidence summary")
        if not bound_evidence:
            st.info("No supporting evidence is currently bound to this finding.")
        else:
            citation_groups = group_evidence_by_citation(bound_evidence)
            st.markdown(
                '<div class="ui-evidence-overview">'
                f'<strong>{len(bound_evidence)} supporting records</strong>'
                f'<span>{len(citation_groups)} citations represented</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            for citation, records in citation_groups:
                st.markdown(
                    '<div class="ui-evidence-citation-summary">'
                    f"<strong>{escape(citation)}</strong>"
                    f"<span>{len(records)} supporting excerpt(s)</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            if st.button(
                "Inspect this finding in Evidence trace",
                key=f"open-evidence-{finding.finding_id}",
            ):
                st.session_state.selected_finding_id = finding.finding_id
                navigate(VIEW_EVIDENCE)

        with st.expander("Technical details", expanded=False):
            metadata_columns = st.columns(3)
            metadata_columns[0].caption("Rule ID")
            metadata_columns[0].code(finding.rule_id or "Not recorded")
            metadata_columns[1].caption("Rule version")
            metadata_columns[1].code(finding.rule_version or "Not recorded")
            metadata_columns[2].caption("Issue code")
            metadata_columns[2].code(finding.issue_code)
            st.caption("Facts schema")
            st.code(schema_version, language=None)
            if finding.fact_refs:
                st.caption("Raw fact keys")
                for fact_path in finding.fact_refs:
                    st.code(fact_path, language=None)
            if finding.reason_codes:
                st.caption("Raw reason codes")
                for reason_code in finding.reason_codes:
                    st.code(reason_code, language=None)
    st.divider()


def render_trace_stage(
    number: str,
    title: str,
    description: str,
) -> None:
    """Render the heading for one stage of the vertical audit trace."""

    st.markdown(
        '<div class="ui-trace-stage-header">'
        f'<div class="ui-trace-stage-number">{number}</div>'
        '<div class="ui-trace-stage-heading">'
        f'<h2 class="ui-trace-stage-title">{escape(title)}</h2>'
        f'<div class="ui-trace-stage-description">{escape(description)}</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


def render_trace_connector() -> None:
    """Render a restrained directional connector between audit stages."""

    st.markdown(
        '<div class="ui-trace-connector" aria-hidden="true">↓</div>',
        unsafe_allow_html=True,
    )


def render_audit_evidence_card(evidence, *, record_number: int) -> None:
    """Render atomic Evidence as an audit-ready record."""

    label = f"Excerpt {record_number} · {evidence.legal_source}"
    with st.expander(label, expanded=False):
        st.markdown(
            '<div class="ui-evidence-trace-meta">'
            f'<span>Authority · {escape(evidence.authority_level.value.replace("_", " ").title())}</span>'
            f'<span>Version · {escape(evidence.document_version or "Not recorded")}</span>'
            f'<span>Source · {escape(evidence.legal_source)}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Authoritative excerpt**")
        st.write(evidence.excerpt)
        st.caption("Stable Evidence ID")
        st.code(evidence.evidence_id, language=None)


def render_report(report: AssessmentReport, facts) -> None:
    """Present structured assessment results without evidence duplication."""

    render_section_header(
        "Assessment report",
        eyebrow="Preliminary result",
        description=(
            "Review framework findings, reasoning, and supporting legal "
            "authority."
        ),
    )
    framework_names = ", ".join(
        framework_label(framework)
        for framework in report.assessed_frameworks
    ) or "No framework recorded"
    st.markdown(
        '<div class="ui-report-context">'
        f'<span>{len(report.findings)} finding(s)</span>'
        f'<span>{len(report.evidence)} evidence record(s)</span>'
        f'<span>{len(report.missing_information)} information gap(s)</span>'
        f'<span>{framework_names}</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    render_section_header("Findings")
    if not report.findings:
        st.info("No legal finding was produced from the currently available facts.")
    for finding in report.findings:
        render_result_finding_card(
            finding,
            evidence_for_finding(report, finding.finding_id),
            facts,
            facts.schema_version,
        )

    render_section_header("Missing information")
    if report.missing_information:
        for item in report.missing_information:
            st.write(
                f"- **{fact_label(item.fact_path)}** — "
                f"{item.reason.value.replace('_', ' ')}"
            )
    else:
        st.write("None identified for the rules currently assessed.")

    if report.recommendations:
        st.subheader("Recommendations")
        for recommendation in report.recommendations:
            st.write(f"- {recommendation}")

    with st.expander("Report technical details", expanded=False):
        st.caption("Report ID")
        st.code(report.report_id, language=None)
        st.caption(
            f"Report version {report.report_version} · "
            f"Engine {report.engine_version} · "
            f"Generated {report.generated_at.isoformat()}"
        )
        if report.missing_information:
            st.caption("Raw missing fact keys")
            for item in report.missing_information:
                st.code(item.fact_path, language=None)

def render_compliance_chain(finding, bound_evidence: list, facts) -> None:
    """Render one finding as a centered, sequential compliance trace."""

    with st.container():
        header_column, status_column = st.columns([4, 1])
        header_column.caption("SELECTED FINDING")
        header_column.markdown(f"## {finding.title}")
        header_column.write(finding.summary)
        with status_column:
            render_status_badge(finding.status)

    render_trace_connector()
    render_trace_stage(
        "01",
        "Facts",
        "The factual inputs used by the assessment rule.",
    )
    with st.container():
        if finding.fact_refs:
            for fact_path in finding.fact_refs:
                st.markdown(
                    '<div class="ui-trace-fact-row">'
                    '<span class="ui-trace-fact-dot"></span>'
                    '<span class="ui-trace-fact-content">'
                    f"<strong>{escape(fact_label(fact_path))}</strong>"
                    f"<span>{escape(fact_value(facts, fact_path))}</span>"
                    "</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No fact references were recorded.")

    render_trace_connector()
    render_trace_stage(
        "02",
        "Rule evaluation",
        "The versioned assessment logic and recorded reasoning sequence.",
    )
    with st.container():
        st.markdown(f"### {rule_label(finding.rule_id)}")
        if finding.trace:
            render_decision_path(finding, facts)
        else:
            st.caption("No reasoning trace was recorded.")
        with st.expander("Rule technical details", expanded=False):
            st.caption("Rule ID and version")
            st.code(
                f"{finding.rule_id or 'Not recorded'} · "
                f"{finding.rule_version or 'Not recorded'}",
                language=None,
            )
            if finding.reason_codes:
                st.caption("Raw reason codes")
                for reason_code in finding.reason_codes:
                    st.code(reason_code, language=None)

    render_trace_connector()
    render_trace_stage(
        "03",
        "Legal basis",
        "The authored legal references supporting the finding.",
    )
    with st.container():
        if finding.legal_basis:
            for basis in finding.legal_basis:
                st.markdown(
                    '<div class="ui-legal-basis-row">'
                    f'<span class="ui-legal-basis-citation">{escape(basis.citation)}</span>'
                    f'<span class="ui-legal-basis-instrument">{escape(basis.instrument)}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No legal basis was recorded.")

    render_trace_connector()
    render_trace_stage(
        "04",
        "Source evidence",
        "Atomic, versioned legal excerpts bound to this finding.",
    )
    if not bound_evidence:
        st.info("No supporting evidence is bound to this finding.")
    for citation, records in group_evidence_by_citation(bound_evidence):
        st.markdown(
            '<section class="ui-evidence-group">'
            '<div class="ui-evidence-group__heading">'
            f"<strong>{escape(citation)}</strong>"
            f"<span>{len(records)} supporting excerpt(s)</span>"
            "</div></section>",
            unsafe_allow_html=True,
        )
        for record_number, evidence in enumerate(records, start=1):
            render_audit_evidence_card(
                evidence,
                record_number=record_number,
            )

    with st.expander("Evidence technical details", expanded=False):
        st.caption("Raw fact keys")
        for fact_path in finding.fact_refs:
            st.code(fact_path, language=None)
        st.caption("All bound stable Evidence IDs")
        for evidence in bound_evidence:
            st.code(evidence.evidence_id, language=None)


def render_evidence_workspace(report: AssessmentReport, facts) -> None:
    """Render finding-specific evidence relationships and stable identities."""

    render_section_header(
        "Evidence trace",
        eyebrow="Legal authority",
        description=(
            "Inspect the relationship between findings, versioned rules, legal "
            "basis references, and atomic corpus evidence."
        ),
    )
    if not report.findings:
        st.info("No finding is available for evidence tracing.")
        return

    finding_by_id = {finding.finding_id: finding for finding in report.findings}
    options = list(finding_by_id)
    selected_id = st.session_state.get("selected_finding_id")
    if selected_id not in finding_by_id:
        selected_id = options[0]
    selected_id = st.selectbox(
        "Select finding",
        options=options,
        index=options.index(selected_id),
        format_func=lambda finding_id: finding_by_id[finding_id].title,
    )
    st.session_state.selected_finding_id = selected_id
    finding = finding_by_id[selected_id]

    bound_evidence = evidence_for_finding(report, selected_id)
    _, chain_column, _ = st.columns([1, 6, 1])
    with chain_column:
        render_compliance_chain(finding, bound_evidence, facts)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")
    apply_enterprise_styles()
    initialize_ui_state()

    try:
        bundle = get_workflow_bundle()
    except (OSError, ValueError) as exc:
        st.error(f"The assessment workflow could not be initialized: {exc}")
        st.stop()

    case_id = st.session_state.get("assessment_case_id")
    report = st.session_state.get("assessment_report")
    render_sidebar(bundle, case_id, report)
    view = st.session_state.get("assessment_view", VIEW_LANDING)

    if view == VIEW_LANDING:
        render_case_creation(bundle)
        return

    if case_id is None:
        st.session_state.assessment_view = VIEW_LANDING
        st.rerun()

    render_case_context(bundle, case_id)
    if view == VIEW_WORKSPACE:
        render_fact_collection(bundle, case_id)
        render_assessment_action(bundle, case_id)
        return

    if report is None:
        st.info("Run the assessment before opening results or evidence trace.")
        if st.button("Return to assessment workspace", type="primary"):
            navigate(VIEW_WORKSPACE)
        return

    facts = bundle.case_service.get_case(case_id).current_facts
    if view == VIEW_RESULTS:
        render_report(report, facts)
        return

    if view == VIEW_EVIDENCE:
        render_evidence_workspace(report, facts)
        return

    st.session_state.assessment_view = VIEW_LANDING
    st.rerun()


if __name__ == "__main__":
    main()
