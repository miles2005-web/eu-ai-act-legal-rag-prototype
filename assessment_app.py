"""Minimal Streamlit interface for the structured assessment workflow."""

from __future__ import annotations

import json
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
    render_assessment_summary_card,
    render_evidence_trace_card,
    render_finding_card,
    render_framework_card,
    render_section_header,
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
        st.caption("PRODUCT NAVIGATION")
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
            "Regulatory frameworks",
            use_container_width=True,
            disabled=report is None,
            type="primary" if current_view == VIEW_RESULTS else "secondary",
        ):
            navigate(VIEW_RESULTS)
        if st.button(
            "Evidence engine",
            use_container_width=True,
            disabled=report is None,
            type="primary" if current_view == VIEW_EVIDENCE else "secondary",
        ):
            navigate(VIEW_EVIDENCE)

        st.divider()
        st.markdown("#### Assessment progress")
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
            st.write(f"{'✅' if complete else '○'} {label}")
        st.progress(sum(complete for _, complete in statuses) / len(statuses))

        st.divider()
        st.caption("REGULATORY COVERAGE")
        st.write("EU AI Act · GDPR · EU Data Act")
        st.caption("Structured rules and versioned legal evidence")

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
            with st.container(border=True):
                st.markdown(
                    '<div class="ui-capability-number">'
                    f"{number}</div>"
                    '<div class="ui-capability-title">'
                    f"{heading}</div>"
                    '<p class="ui-capability-copy">'
                    f"{description}</p>",
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
    title_column, action_column = st.columns([4, 1])
    with title_column:
        st.caption("ACTIVE ASSESSMENT CASE")
        st.markdown(f"# {assessment_case.name}")
        if assessment_case.description:
            st.write(assessment_case.description)
    with action_column:
        st.metric("Schema version", assessment_case.schema_version)

    profile_column, scope_column, status_column = st.columns(3)
    with profile_column:
        with st.container(border=True):
            st.caption("SYSTEM PROFILE")
            st.markdown(f"**{facts.system.name or 'Unnamed AI system'}**")
            st.write(facts.system.intended_purpose or "Purpose not yet provided.")
    with scope_column:
        with st.container(border=True):
            st.caption("USE CONTEXT")
            st.markdown(f"**{DOMAIN_LABELS[facts.use_context.domain]}**")
            st.write(facts.use_context.task or "Task not yet provided.")
    with status_column:
        with st.container(border=True):
            st.caption("ASSESSMENT STATUS")
            st.markdown(
                "**Report available**"
                if st.session_state.get("assessment_report") is not None
                else "**Facts in progress**"
            )
            demo = st.session_state.get("demo_scenario")
            st.write(f"Scenario: {demo.title()}" if demo else "Custom case")


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


def render_report(report: AssessmentReport) -> None:
    """Present structured assessment results without evidence duplication."""

    render_section_header(
        "Assessment report",
        eyebrow="Preliminary result",
        description=(
            "Review framework findings, reasoning, and supporting legal "
            "authority."
        ),
    )
    render_assessment_summary_card(report)

    if report.findings_by_framework:
        render_section_header("Framework overview")
        framework_columns = st.columns(
            min(len(report.findings_by_framework), 3)
        )
        for index, group in enumerate(report.findings_by_framework):
            with framework_columns[index % len(framework_columns)]:
                render_framework_card(group.framework, group.findings)

    render_section_header("Findings")
    if not report.findings:
        st.info("No legal finding was produced from the currently available facts.")
    for finding in report.findings:
        binding = next(
            (
                item
                for item in report.evidence_bindings
                if item.finding_id == finding.finding_id
            ),
            None,
        )
        render_finding_card(
            finding,
            evidence_count=len(binding.evidence_refs) if binding else 0,
        )

    render_section_header("Missing information")
    if report.missing_information:
        for item in report.missing_information:
            st.write(f"- `{item.fact_path}` ({item.reason.value})")
    else:
        st.write("None identified for the rules currently assessed.")

    if report.recommendations:
        st.subheader("Recommendations")
        for recommendation in report.recommendations:
            st.write(f"- {recommendation}")

    st.caption(
        f"Report {report.report_id} · Engine {report.engine_version} · "
        f"Generated {report.generated_at.isoformat()}"
    )

    if st.button(
        "Open evidence trace",
        type="primary",
        use_container_width=True,
    ):
        navigate(VIEW_EVIDENCE)


def render_evidence_workspace(report: AssessmentReport) -> None:
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

    binding = next(
        (
            item
            for item in report.evidence_bindings
            if item.finding_id == selected_id
        ),
        None,
    )
    evidence_by_id = {item.evidence_id: item for item in report.evidence}
    bound_evidence = [
        evidence_by_id[evidence_id]
        for evidence_id in (binding.evidence_refs if binding else [])
        if evidence_id in evidence_by_id
    ]

    overview_column, trace_column = st.columns([2, 3])
    with overview_column:
        render_finding_card(finding, evidence_count=len(bound_evidence))
    with trace_column:
        with st.container(border=True):
            st.caption("TRACEABILITY CHAIN")
            st.markdown("#### Fact → Rule → Legal basis → Evidence")
            st.write("**Facts referenced**")
            for fact_path in finding.fact_refs:
                st.code(fact_path, language=None)
            st.write("**Rule metadata**")
            st.write(
                f"`{finding.rule_id or 'Not recorded'}` · "
                f"Version `{finding.rule_version or 'Not recorded'}`"
            )
            st.write("**Authored legal basis**")
            for basis in finding.legal_basis:
                st.write(f"- {basis.instrument} · {basis.citation}")

    render_section_header(
        "Bound evidence",
        description=f"{len(bound_evidence)} atomic evidence record(s).",
    )
    if not bound_evidence:
        st.info("No supporting evidence is bound to this finding.")
    for evidence in bound_evidence:
        render_evidence_trace_card(evidence)


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

    if view == VIEW_RESULTS:
        render_report(report)
        return

    if view == VIEW_EVIDENCE:
        render_evidence_workspace(report)
        return

    st.session_state.assessment_view = VIEW_LANDING
    st.rerun()


if __name__ == "__main__":
    main()
