"""Minimal Streamlit interface for the structured assessment workflow."""

from __future__ import annotations

import streamlit as st

from scripts.run_demo_assessment import build_assessment_facts, load_fixture
from src.assessment.demo import AssessmentWorkflowBundle, create_assessment_workflow
from src.assessment.facts import UseDomain
from src.assessment.models import TriState
from src.assessment.report import AssessmentReport


APP_TITLE = "EU AI Act Compliance Assessment Platform"
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
    st.rerun()


def render_sidebar(
    bundle: AssessmentWorkflowBundle,
    case_id: str | None,
    report: AssessmentReport | None,
) -> None:
    """Show presentation-only workflow progress for the current session."""

    with st.sidebar:
        st.header("Assessment progress")
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
        st.caption("Current assessment scope")
        st.write("EU AI Act employment high-risk screening")
        st.caption("Article 6 · Annex III point 4(a)")

        if case_id is not None and st.button("Start a new assessment"):
            st.session_state.assessment_workflow_bundle = (
                create_assessment_workflow()
            )
            st.session_state.assessment_case_id = None
            st.session_state.assessment_report = None
            st.session_state.demo_loaded = False
            st.rerun()


def render_case_creation(bundle: AssessmentWorkflowBundle) -> None:
    """Collect case identity data and create an in-memory assessment case."""

    st.subheader("Start an assessment")
    st.write(
        "Explore the prepared recruitment scenario or create a case using your "
        "own facts."
    )
    if st.button(
        "Load Recruitment AI Screening Demo",
        type="primary",
        use_container_width=True,
    ):
        load_recruitment_demo(bundle)

    st.caption(
        "The demo describes an AI system that screens CVs, ranks candidates, "
        "and materially influences recruitment decisions."
    )
    st.divider()
    st.markdown("#### Create a blank case")
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
        st.rerun()


def render_fact_collection(bundle: AssessmentWorkflowBundle, case_id: str) -> None:
    """Collect the facts required by the registered employment rule."""

    assessment_case = bundle.case_service.get_case(case_id)
    facts = assessment_case.current_facts

    st.subheader("2. Provide assessment facts")
    if st.session_state.get("demo_loaded"):
        st.info(
            "Recruitment AI Screening Demo loaded. Review or edit the populated "
            "facts before running the assessment."
        )
    st.caption(
        "Unknown answers are preserved as missing information and do not create "
        "a legal finding."
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

    st.subheader("3. Run assessment")
    if st.button("Run assessment", type="primary", use_container_width=True):
        with st.spinner("Running assessment and resolving legal evidence..."):
            st.session_state.assessment_report = bundle.workflow.run(case_id)
        st.rerun()


def render_report(report: AssessmentReport) -> None:
    """Present the structured report without adding legal conclusions."""

    st.divider()
    st.header("Assessment report")

    primary_finding = report.findings[0] if report.findings else None
    classification = (
        primary_finding.status.value.replace("_", " ").title()
        if primary_finding is not None
        else "Not determined"
    )
    review_status = (
        "Legal review required"
        if primary_finding is not None and primary_finding.requires_legal_review
        else "No review flag raised"
    )
    classification_column, review_column, evidence_column = st.columns(3)
    classification_column.metric("Preliminary classification", classification)
    review_column.metric("Review status", review_status)
    evidence_column.metric("Evidence records", len(report.evidence))

    st.subheader("Report summary")
    st.write(report.summary)

    st.subheader("Findings")
    if not report.findings:
        st.info("No legal finding was produced from the currently available facts.")
    for finding in report.findings:
        st.markdown(f"### {finding.title}")
        st.write(
            "**Preliminary classification:** "
            f"`{finding.status.value.replace('_', ' ')}`"
        )
        st.write(finding.summary)
        st.markdown("**Legal basis**")
        for basis in finding.legal_basis:
            st.write(f"- {basis.citation} — `{basis.instrument}`")
        if finding.requires_legal_review:
            st.warning(
                "Review status: further legal review is required. This is a "
                "preliminary classification, not a definitive legal conclusion."
            )
        else:
            st.success("Review status: no additional review flag was raised.")
        with st.expander("Reasoning trace", expanded=True):
            for trace_entry in finding.trace:
                st.markdown(f"**{trace_entry.description}**")
                st.write(f"Result: `{trace_entry.result}`")
                if trace_entry.fact_refs:
                    st.caption("Facts: " + ", ".join(trace_entry.fact_refs))

    st.subheader("Evidence")
    st.caption(f"{len(report.evidence)} supporting evidence record(s)")
    if not report.evidence:
        st.info("No supporting evidence was resolved for the current findings.")
    for evidence in report.evidence:
        with st.expander(f"{evidence.citation} — {evidence.legal_source}"):
            st.write(f"**Authority:** {evidence.authority_level.value}")
            if evidence.document_version:
                st.write(f"**Document version:** {evidence.document_version}")
            st.write(evidence.excerpt)

    st.subheader("Missing information")
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


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="centered")
    st.title(APP_TITLE)
    st.write(
        "A legal engineering prototype for preliminary, evidence-grounded "
        "assessment under the EU AI Act."
    )
    st.caption("Prototype only — this output does not constitute legal advice.")

    capability_columns = st.columns(4)
    capabilities = (
        ("Structured facts", "Capture legally relevant facts without inference."),
        ("Rule-based assessment", "Apply deterministic, versioned legal rules."),
        ("Legal evidence", "Retrieve supporting provisions from the legal corpus."),
        ("Traceable reports", "Connect facts, findings, reasoning, and authority."),
    )
    for column, (heading, description) in zip(
        capability_columns,
        capabilities,
        strict=True,
    ):
        with column:
            st.markdown(f"**{heading}**")
            st.caption(description)

    try:
        bundle = get_workflow_bundle()
    except (OSError, ValueError) as exc:
        st.error(f"The assessment workflow could not be initialized: {exc}")
        st.stop()

    case_id = st.session_state.get("assessment_case_id")
    report = st.session_state.get("assessment_report")
    render_sidebar(bundle, case_id, report)
    if case_id is None:
        render_case_creation(bundle)
        return

    assessment_case = bundle.case_service.get_case(case_id)
    st.success(f"Current case: {assessment_case.name}")
    render_fact_collection(bundle, case_id)
    render_assessment_action(bundle, case_id)

    if report is not None:
        render_report(report)


if __name__ == "__main__":
    main()
