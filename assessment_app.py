"""Minimal Streamlit interface for the structured assessment workflow."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import streamlit as st

from scripts.run_demo_assessment import (
    build_assessment_facts,
    load_fixture,
)
from src.assessment.demo import AssessmentWorkflowBundle, create_assessment_workflow
from src.assessment.facts import UseDomain
from src.assessment.findings import FindingStatus
from src.assessment.frameworks import RegulatoryFramework
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
    status_label,
)
from src.ui.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_LABELS,
    SUPPORTED_LANGUAGES,
    count_text,
    t,
    t_or,
)
from src.ui.normalization import (
    NormalizationStatus,
    apply_normalized_input,
    normalize_legal_input,
)


PROJECT_ROOT = Path(__file__).resolve().parent
INDUSTRIAL_FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "industrial_ai_case.json"
)
VIEW_LANDING = "landing"
VIEW_WORKSPACE = "workspace"
VIEW_RESULTS = "results"
VIEW_EVIDENCE = "evidence"
CASE_FORM_STATE_KEYS = (
    "assessment_fact_domain",
    "assessment_fact_task",
    "assessment_fact_material_influence",
    "assessment_confirm_ambiguous_task",
    "assessment_input_normalization",
)
PRIMARY_FINDING_STATUSES = frozenset(
    (
        FindingStatus.APPLIES,
        FindingStatus.POTENTIALLY_APPLIES,
        FindingStatus.UNDETERMINED,
    )
)


def domain_label(domain: UseDomain, language: str) -> str:
    return t(f"domain.{domain.value}", language)


def tri_state_label(value: TriState, language: str) -> str:
    return t(f"value.{value.value}", language)


def finding_title(finding, language: str) -> str:
    key = f"finding.{finding.rule_id}.{finding.status.value}.title"
    return t_or(key, finding.title, language)


def finding_summary(finding, language: str) -> str:
    key = f"finding.{finding.rule_id}.{finding.status.value}.summary"
    return t_or(key, finding.summary, language)


def trace_description(entry, language: str) -> str:
    if not entry.fact_refs:
        return entry.description
    return t_or(
        f"trace.description.{entry.fact_refs[0]}",
        entry.description,
        language,
    )


def instrument_label(instrument: str, language: str) -> str:
    return t_or(f"framework.{instrument}", instrument, language)


def scenario_text(field: str, fallback: str, language: str) -> str:
    """Localize predefined demo content without changing fixture facts."""

    scenario_id = st.session_state.get("demo_scenario_id")
    if not scenario_id:
        return fallback
    return t_or(f"scenario.{scenario_id}.{field}", fallback, language)


def displayed_task(facts, language: str) -> str:
    """Prefer localized demo copy or the original user-entered task text."""

    demo_task = scenario_text(
        "task",
        facts.use_context.task or t("case.task.missing", language),
        language,
    )
    if st.session_state.get("demo_scenario_id"):
        return demo_task
    metadata = st.session_state.get("assessment_input_normalization") or {}
    original = metadata.get("original_text")
    return original or demo_task


def task_input_value(facts, language: str) -> str:
    """Return editable original text while keeping canonical facts separate."""

    metadata = st.session_state.get("assessment_input_normalization") or {}
    original = metadata.get("original_text")
    if original is not None:
        return original
    return scenario_text("task", facts.use_context.task or "", language)


def sync_predefined_task_input(facts, language: str) -> None:
    """Relocalize untouched demo text without replacing user-entered input."""

    scenario_id = st.session_state.get("demo_scenario_id")
    if not scenario_id:
        return
    widget_key = "assessment_fact_task"
    current = st.session_state.get(widget_key)
    variants = {
        facts.use_context.task or "",
        t_or(f"scenario.{scenario_id}.task", "", "en"),
        t_or(f"scenario.{scenario_id}.task", "", "zh-CN"),
    }
    if current is None or current in variants:
        st.session_state[widget_key] = scenario_text(
            "task",
            facts.use_context.task or "",
            language,
        )


def get_workflow_bundle() -> AssessmentWorkflowBundle:
    """Create one in-memory workflow bundle for the current UI session."""

    if "assessment_workflow_bundle" not in st.session_state:
        st.session_state.assessment_workflow_bundle = create_assessment_workflow()
    return st.session_state.assessment_workflow_bundle


def initialize_ui_state() -> None:
    """Initialize presentation-only navigation state."""

    st.session_state.setdefault("assessment_view", VIEW_LANDING)
    st.session_state.setdefault("demo_scenario", None)
    st.session_state.setdefault("demo_scenario_id", None)
    st.session_state.setdefault("selected_finding_id", None)
    st.session_state.setdefault("ui_language", DEFAULT_LANGUAGE)


def navigate(view: str) -> None:
    """Navigate between workspace views without changing domain state."""

    st.session_state.assessment_view = view
    st.rerun()


def reset_case_dependent_state(*, view: str) -> AssessmentWorkflowBundle:
    """Replace all case-owned state while preserving presentation preferences."""

    for key in CASE_FORM_STATE_KEYS:
        st.session_state.pop(key, None)
    bundle = create_assessment_workflow()
    st.session_state.assessment_workflow_bundle = bundle
    st.session_state.assessment_case_id = None
    st.session_state.assessment_report = None
    st.session_state.selected_finding_id = None
    st.session_state.demo_loaded = False
    st.session_state.demo_scenario = None
    st.session_state.demo_scenario_id = None
    st.session_state.assessment_view = view
    return bundle


def report_belongs_to_case(
    bundle: AssessmentWorkflowBundle,
    report: AssessmentReport,
    case_id: str,
) -> bool:
    """Verify the report's immutable run snapshot belongs to the active case."""

    try:
        assessment_run = bundle.workflow.get_run(
            report.assessment_run_reference
        )
    except (KeyError, ValueError):
        return False
    return assessment_run.case_id == case_id


def clear_mismatched_report(
    bundle: AssessmentWorkflowBundle,
    case_id: str | None,
    report: AssessmentReport | None,
) -> AssessmentReport | None:
    """Prevent a report from one case or bundle rendering under another case."""

    if report is None:
        return None
    if case_id is not None and report_belongs_to_case(bundle, report, case_id):
        return report
    st.session_state.assessment_report = None
    st.session_state.selected_finding_id = None
    if st.session_state.get("assessment_view") in (VIEW_RESULTS, VIEW_EVIDENCE):
        st.session_state.assessment_view = (
            VIEW_WORKSPACE if case_id is not None else VIEW_LANDING
        )
    return None


def ordered_findings_for_presentation(findings: list) -> tuple[list, list]:
    """Separate substantive preliminary findings from screened-out records."""

    primary = [
        finding
        for finding in findings
        if finding.status in PRIMARY_FINDING_STATUSES
    ]
    screened_out = [
        finding
        for finding in findings
        if finding.status not in PRIMARY_FINDING_STATUSES
    ]
    return primary, screened_out


def prioritize_selected_finding(findings: list, selected_id: str | None) -> list:
    """Place the selected finding first without changing report ordering."""

    if not selected_id:
        return list(findings)
    return sorted(
        findings,
        key=lambda finding: finding.finding_id != selected_id,
    )


def scoped_recommendations(
    report: AssessmentReport,
    findings: list,
    missing_information: list,
    language: str,
) -> list[str]:
    """Build identifier-free presentation advice for one framework scope."""

    recommendations = [
        t(
            "recommendation.requirement",
            language,
            fact=fact_label(item.fact_path, language),
            framework=framework_label(item.framework, language),
        )
        for item in missing_information
    ]
    for finding in findings:
        if finding.requires_legal_review:
            recommendations.append(
                t_or(
                    f"recommendation.review.{finding.rule_id}",
                    t("finding.review_warning", language),
                    language,
                )
            )

    bound_finding_ids = {
        binding.finding_id for binding in report.evidence_bindings
    }
    for finding in findings:
        if finding.legal_basis and finding.finding_id not in bound_finding_ids:
            recommendations.append(
                t("recommendation.supporting_authority_readable", language)
            )

    frameworks = {
        *(finding.framework for finding in findings),
        *(item.framework for item in missing_information),
    }
    for failure in report.execution_failures:
        if failure.framework in frameworks:
            recommendations.append(
                t(
                    "recommendation.execution_failure_readable",
                    language,
                    rule=rule_label(failure.rule_id, language),
                )
            )
    return list(dict.fromkeys(recommendations))


def framework_screen_status_key(
    findings: list,
    missing_information: list,
    failures: list,
) -> str:
    """Return a presentation status without changing assessment outcomes."""

    if any(finding.status in PRIMARY_FINDING_STATUSES for finding in findings):
        return "framework_screen.potentially_relevant"
    if missing_information:
        return "framework_screen.additional_facts"
    if failures:
        return "framework_screen.failure"
    if findings:
        return "framework_screen.screened_out"
    return "framework_screen.completed"


def render_missing_item(item, language: str, *, scope: str) -> None:
    """Render a localized requirement while retaining raw keys elsewhere."""

    st.markdown(
        f'<div class="ui-{escape(scope)}-missing">'
        f'<strong>{escape(fact_label(item.fact_path, language))}</strong>'
        f' — {escape(t(f"missing_reason.{item.reason.value}", language))}'
        "</div>",
        unsafe_allow_html=True,
    )


def render_recommendation(
    recommendation: str,
    *,
    scope: str,
) -> None:
    """Render identifier-free advice with an explicit presentation scope."""

    st.markdown(
        f'<div class="ui-{escape(scope)}-recommendation">'
        f"{escape(recommendation)}</div>",
        unsafe_allow_html=True,
    )


def load_recruitment_demo() -> None:
    """Create a case from the existing pure-data recruitment demo fixture."""

    payload = load_fixture()
    bundle = reset_case_dependent_state(view=VIEW_WORKSPACE)
    assessment_case = bundle.case_service.create_case(
        payload["scenario"]["name"],
        description=payload["scenario"]["description"],
        facts=build_assessment_facts(payload["facts"]),
    )
    st.session_state.assessment_case_id = assessment_case.case_id
    st.session_state.demo_loaded = True
    st.session_state.demo_scenario = "recruitment"
    st.session_state.demo_scenario_id = payload["scenario_id"]
    st.rerun()


def load_industrial_demo() -> None:
    """Create a case from the existing industrial AI fixture."""

    payload = json.loads(INDUSTRIAL_FIXTURE_PATH.read_text(encoding="utf-8"))
    facts = build_assessment_facts(payload["facts"])
    bundle = reset_case_dependent_state(view=VIEW_WORKSPACE)
    assessment_case = bundle.case_service.create_case(
        payload["scenario"]["name"],
        description=payload["scenario"]["description"],
        facts=facts,
    )
    st.session_state.assessment_case_id = assessment_case.case_id
    st.session_state.demo_loaded = True
    st.session_state.demo_scenario = "industrial"
    st.session_state.demo_scenario_id = payload["scenario_id"]
    st.rerun()


def render_sidebar(
    bundle: AssessmentWorkflowBundle,
    case_id: str | None,
    report: AssessmentReport | None,
    language: str,
) -> None:
    """Show presentation-only workflow progress for the current session."""

    with st.sidebar:
        st.markdown(
            f"""
            <div class="ui-product-mark">
              <div class="ui-product-mark__eyebrow">{t("product.eyebrow", language)}</div>
              <div class="ui-product-mark__name">{t("product.name", language)}</div>
              <div class="ui-product-mark__meta">{t("product.meta", language)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_language = st.radio(
            t("language.selector", language),
            options=SUPPORTED_LANGUAGES,
            format_func=LANGUAGE_LABELS.get,
            horizontal=True,
            key="ui_language",
            label_visibility="collapsed",
        )
        language = selected_language
        st.markdown(
            f'<div class="ui-nav-label">{t("navigation.workspace", language)}</div>',
            unsafe_allow_html=True,
        )
        current_view = st.session_state.get("assessment_view", VIEW_LANDING)
        if st.button(
            t("navigation.assessment", language),
            use_container_width=True,
            disabled=case_id is None,
            type="primary" if current_view == VIEW_WORKSPACE else "secondary",
        ):
            navigate(VIEW_WORKSPACE)
        if st.button(
            t("navigation.demo_cases", language),
            use_container_width=True,
            type="primary" if current_view == VIEW_LANDING else "secondary",
        ):
            navigate(VIEW_LANDING)
        if st.button(
            t("navigation.findings", language),
            use_container_width=True,
            disabled=report is None,
            type="primary" if current_view == VIEW_RESULTS else "secondary",
        ):
            navigate(VIEW_RESULTS)
        if st.button(
            t("navigation.evidence_trace", language),
            use_container_width=True,
            disabled=report is None,
            type="primary" if current_view == VIEW_EVIDENCE else "secondary",
        ):
            navigate(VIEW_EVIDENCE)

        st.markdown(
            '<div class="ui-nav-label ui-nav-label--reference">'
            f'{t("navigation.reference", language)}</div>',
            unsafe_allow_html=True,
        )
        with st.expander(t("navigation.frameworks", language), expanded=False):
            st.caption(
                " · ".join(
                    framework_label(framework, language)
                    for framework in (
                        RegulatoryFramework.EU_AI_ACT,
                        RegulatoryFramework.GDPR,
                        RegulatoryFramework.EU_DATA_ACT,
                    )
                )
            )
            st.write(t("reference.frameworks.copy", language))
        with st.expander(t("navigation.technical_details", language), expanded=False):
            if case_id is None:
                st.caption(t("reference.case_required", language))
            else:
                assessment_case = bundle.case_service.get_case(case_id)
                st.caption(t("technical.case_id", language))
                st.code(assessment_case.case_id, language=None)
                st.caption(
                    f'{t("technical.facts_schema", language)} '
                    f"{assessment_case.current_facts.schema_version}"
                )
                if report is not None:
                    st.caption(
                        t(
                            "technical.report",
                            language,
                            version=report.report_version,
                            engine=report.engine_version,
                        )
                    )

        st.markdown(
            f'<div class="ui-progress-heading">{t("progress.title", language)}</div>',
            unsafe_allow_html=True,
        )
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
            (t("progress.case", language), case_complete),
            (t("progress.facts", language), facts_complete),
            (t("progress.assessment", language), report is not None),
        )
        for label, complete in statuses:
            state = t(
                "progress.complete" if complete else "progress.pending",
                language,
            )
            icon = "✓" if complete else "○"
            st.markdown(
                '<div class="ui-progress-row">'
                f'<span aria-hidden="true">{icon}</span>'
                f'<span>{label}</span><span class="ui-progress-state">{state}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        st.progress(sum(complete for _, complete in statuses) / len(statuses))

        if case_id is not None and st.button(
            t("navigation.new_assessment", language)
        ):
            reset_case_dependent_state(view=VIEW_LANDING)
            st.rerun()


def render_case_creation(bundle: AssessmentWorkflowBundle, language: str) -> None:
    """Collect case identity data and create an in-memory assessment case."""

    st.markdown(
        f"""
        <section class="ui-hero">
          <div class="ui-hero__eyebrow">{t("landing.eyebrow", language)}</div>
          <h1 class="ui-hero__title">{t("landing.title", language)}</h1>
          <p class="ui-hero__summary">
            {t("landing.subtitle", language)}
          </p>
          <div class="ui-framework-strip">
            <span class="ui-badge ui-badge--accent">{t("framework.EU_AI_ACT", language)}</span>
            <span class="ui-badge ui-badge--neutral">{t("framework.GDPR", language)}</span>
            <span class="ui-badge ui-badge--neutral">{t("framework.EU_DATA_ACT", language)}</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.caption(t("landing.disclaimer", language))

    capability_columns = st.columns(4)
    capabilities = (
        (
            "01",
            t("capability.facts.title", language),
            t("capability.facts.copy", language),
        ),
        (
            "02",
            t("capability.rules.title", language),
            t("capability.rules.copy", language),
        ),
        (
            "03",
            t("capability.evidence.title", language),
            t("capability.evidence.copy", language),
        ),
        (
            "04",
            t("capability.trace.title", language),
            t("capability.trace.copy", language),
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
            t("landing.current_case", language),
            eyebrow=t("landing.continue", language),
        )
        with st.container(border=True):
            case_column, action_column = st.columns([4, 1])
            case_column.markdown(
                f'### {scenario_text("case_name", active_case.name, language)}'
            )
            case_column.write(
                scenario_text(
                    "description",
                    active_case.description
                    or t("landing.no_description", language),
                    language,
                )
            )
            with action_column:
                if st.button(
                    t("landing.open_workspace", language),
                    type="primary",
                    use_container_width=True,
                ):
                    navigate(VIEW_WORKSPACE)

    render_section_header(
        t("demo.section.title", language),
        eyebrow=t("demo.section.eyebrow", language),
        description=t("demo.section.copy", language),
    )
    recruitment_column, industrial_column = st.columns(2)
    with recruitment_column:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="ui-demo-label">{t("demo.recruitment.label", language)}</div>
                <div class="ui-demo-title">{t("demo.recruitment.title", language)}</div>
                <div class="ui-demo-copy">
                  {t("demo.recruitment.copy", language)}
                </div>
                <div class="ui-demo-meta">
                  {t("demo.recruitment.meta", language)}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                t("demo.recruitment.open", language),
                type="primary",
                use_container_width=True,
            ):
                load_recruitment_demo()
    with industrial_column:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="ui-demo-label">{t("demo.industrial.label", language)}</div>
                <div class="ui-demo-title">{t("demo.industrial.title", language)}</div>
                <div class="ui-demo-copy">
                  {t("demo.industrial.copy", language)}
                </div>
                <div class="ui-demo-meta">
                  {t("demo.industrial.meta", language)}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                t("demo.industrial.open", language),
                type="primary",
                use_container_width=True,
            ):
                load_industrial_demo()

    render_section_header(
        t("case.new.title", language),
        eyebrow=t("case.new.eyebrow", language),
        description=t("case.new.copy", language),
    )
    with st.form("create_assessment_case", clear_on_submit=True):
        name = st.text_input(
            t("case.name", language),
            placeholder=t("case.name.placeholder", language),
        )
        description = st.text_area(
            t("case.description", language),
            placeholder=t("case.description.placeholder", language),
        )
        submitted = st.form_submit_button(t("case.create", language), type="primary")

    if submitted:
        if not name.strip():
            st.error(t("case.name.required", language))
            return
        bundle = reset_case_dependent_state(view=VIEW_WORKSPACE)
        assessment_case = bundle.case_service.create_case(
            name.strip(),
            description=description.strip() or None,
        )
        st.session_state.assessment_case_id = assessment_case.case_id
        st.rerun()


def render_case_context(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    language: str,
) -> None:
    """Render the active case header and a concise system profile."""

    assessment_case = bundle.case_service.get_case(case_id)
    facts = assessment_case.current_facts
    st.caption(t("case.active", language).upper())
    display_name = scenario_text(
        "case_name",
        assessment_case.name,
        language,
    )
    display_description = scenario_text(
        "description",
        assessment_case.description or "",
        language,
    )
    st.markdown(f"# {display_name}")
    if assessment_case.description:
        st.markdown(
            f'<p class="ui-case-description">{escape(display_description)}</p>',
            unsafe_allow_html=True,
        )

    report_ready = st.session_state.get("assessment_report") is not None
    status = t(
        "case.report_available" if report_ready else "case.facts_in_progress",
        language,
    )
    demo = st.session_state.get("demo_scenario")
    scenario = (
        t(
            "case.demo",
            language,
            name=scenario_text(
                "case_name",
                t(f"demo.{demo}.title", language),
                language,
            ),
        )
        if demo
        else t("case.custom", language)
    )
    intended_purpose = scenario_text(
        "purpose",
        facts.system.intended_purpose or t("case.purpose.missing", language),
        language,
    )
    st.markdown(
        '<section class="ui-system-summary" '
        f'aria-label="{escape(t("case.system_summary_aria", language))}">'
        '<div class="ui-system-summary__item">'
        f'<span class="ui-system-summary__label">{t("case.system", language)}</span>'
        f'<strong>{escape(facts.system.name or t("case.system.unnamed", language))}</strong>'
        f'<span>{escape(intended_purpose)}</span>'
        "</div>"
        '<div class="ui-system-summary__item">'
        f'<span class="ui-system-summary__label">{t("case.use_context", language)}</span>'
        f'<strong>{escape(domain_label(facts.use_context.domain, language))}</strong>'
        f'<span>{escape(displayed_task(facts, language))}</span>'
        "</div>"
        '<div class="ui-system-summary__item">'
        f'<span class="ui-system-summary__label">{t("case.assessment", language)}</span>'
        f"<strong>{status}</strong><span>{scenario}</span>"
        "</div></section>",
        unsafe_allow_html=True,
    )
    with st.expander(t("navigation.technical_details", language), expanded=False):
        detail_columns = st.columns(3)
        detail_columns[0].caption(t("technical.case_id", language))
        detail_columns[0].code(assessment_case.case_id, language=None)
        detail_columns[1].caption(t("technical.case_schema", language))
        detail_columns[1].code(assessment_case.schema_version, language=None)
        detail_columns[2].caption(t("technical.facts_schema", language))
        detail_columns[2].code(facts.schema_version, language=None)


def render_fact_collection(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    language: str,
) -> None:
    """Collect the facts required by the registered employment rule."""

    assessment_case = bundle.case_service.get_case(case_id)
    facts = assessment_case.current_facts

    render_section_header(
        t("facts.title", language),
        eyebrow=t("facts.eyebrow", language),
        description=t("facts.copy", language),
    )
    if st.session_state.get("demo_loaded"):
        demo_name = scenario_text(
            "case_name",
            t(
                "demo.industrial.title"
                if st.session_state.get("demo_scenario") == "industrial"
                else "demo.recruitment.title",
                language,
            ),
            language,
        )
        st.info(t("facts.demo_loaded", language, name=demo_name))
    domains = list(UseDomain)
    influence_options = list(TriState)
    sync_predefined_task_input(facts, language)
    with st.form("assessment_facts"):
        domain = st.selectbox(
            t("facts.domain.question", language),
            options=domains,
            index=domains.index(facts.use_context.domain),
            format_func=lambda value: domain_label(value, language),
            key="assessment_fact_domain",
        )
        task_options = {}
        if "assessment_fact_task" not in st.session_state:
            task_options["value"] = task_input_value(facts, language)
        task = st.text_area(
            t("facts.task.question", language),
            placeholder=t("facts.task.placeholder", language),
            key="assessment_fact_task",
            **task_options,
        )
        normalization_preview = normalize_legal_input(task)
        confirm_ambiguous_task = False
        if normalization_preview.status is NormalizationStatus.MATCHED:
            st.caption(
                t(
                    "normalization.recognized",
                    language,
                    mappings=", ".join(normalization_preview.mapping_ids),
                )
            )
        elif normalization_preview.status is NormalizationStatus.AMBIGUOUS:
            st.warning(t("normalization.ambiguous", language))
            confirm_ambiguous_task = st.checkbox(
                t("normalization.confirm_original", language),
                key="assessment_confirm_ambiguous_task",
            )
        materially_influences = st.selectbox(
            t("facts.influence.question", language),
            options=influence_options,
            index=influence_options.index(
                facts.use_context.materially_influences_decision
            ),
            format_func=lambda value: tri_state_label(value, language),
            key="assessment_fact_material_influence",
        )
        facts_submitted = st.form_submit_button(
            t("facts.save", language),
            type="primary",
        )

    if facts_submitted:
        facts.use_context.domain = domain
        facts.use_context.materially_influences_decision = materially_influences
        protected_paths = (
            frozenset(("use_context.materially_influences_decision",))
            if materially_influences is not TriState.UNKNOWN
            else frozenset()
        )
        apply_normalized_input(
            facts,
            normalization_preview,
            ambiguous_text_confirmed=confirm_ambiguous_task,
            protected_fact_paths=protected_paths,
        )
        st.session_state.assessment_input_normalization = {
            **normalization_preview.to_dict(),
            "ambiguous_text_confirmed": confirm_ambiguous_task,
        }
        bundle.case_service.update_facts(case_id, facts)
        st.session_state.assessment_report = None
        st.session_state.selected_finding_id = None
        if (
            normalization_preview.status is NormalizationStatus.AMBIGUOUS
            and not confirm_ambiguous_task
        ):
            st.warning(t("normalization.saved_unknown", language))
        else:
            st.success(t("facts.saved", language))


def render_assessment_action(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    language: str,
) -> None:
    """Execute the existing assessment workflow on user request."""

    render_section_header(
        t("assessment.title", language),
        eyebrow=t("assessment.eyebrow", language),
        description=t("assessment.copy", language),
    )
    if st.button(
        t("assessment.run", language),
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(t("assessment.running", language)):
            st.session_state.assessment_report = bundle.workflow.run(case_id)
        st.session_state.selected_finding_id = None
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


def render_decision_path(finding, facts, language: str) -> None:
    """Render a finding trace as human-readable decision stages."""

    st.markdown(f'### {t("decision.title", language)}')
    if not finding.trace:
        st.caption(t("decision.no_trace", language))
        return
    for index, entry in enumerate(finding.trace, start=1):
        label = (
            fact_label(entry.fact_refs[0], language)
            if entry.fact_refs
            else t("decision.condition", language, number=index)
        )
        state_label, state_tone = reasoning_state(entry.result, language)
        value = (
            fact_value(facts, entry.fact_refs[0], language)
            if entry.fact_refs
            else readable_result(entry.result, language)
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
            f'<p>{escape(trace_description(entry, language))}</p>'
            "</div></div>",
            unsafe_allow_html=True,
        )


def render_result_finding_card(
    finding,
    bound_evidence: list,
    facts,
    schema_version: str,
    language: str,
) -> None:
    """Present one finding in user-first compliance review order."""

    with st.container():
        status_column, framework_column, count_column = st.columns([2, 2, 3])
        with status_column:
            render_status_badge(finding.status, language)
        with framework_column:
            render_framework_badge(
                finding.framework,
                tone="neutral",
                language=language,
            )
        count_column.markdown(
            '<div class="ui-finding-evidence-count">'
            f'{count_text("finding.evidence_count", len(bound_evidence), language)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div class="ui-finding-kicker">{t("finding.kicker", language)}</div>'
            f'<h2 class="ui-finding-title ui-finding-title--hero">'
            f"{escape(finding_title(finding, language))}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="ui-finding-summary">{escape(finding_summary(finding, language))}</p>',
            unsafe_allow_html=True,
        )

        if finding.requires_legal_review:
            st.warning(
                t("finding.review_warning", language)
            )

        render_decision_path(finding, facts, language)

        st.markdown(f'### {t("legal.title", language)}')
        if finding.legal_basis:
            for basis in finding.legal_basis:
                st.markdown(
                    '<div class="ui-legal-basis-row">'
                    f'<span class="ui-legal-basis-citation">{escape(basis.citation)}</span>'
                    f'<span class="ui-legal-basis-instrument">{escape(instrument_label(basis.instrument, language))}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption(t("legal.none", language))

        st.markdown(f'### {t("evidence.summary.title", language)}')
        if not bound_evidence:
            st.info(t("evidence.none", language))
        else:
            citation_groups = group_evidence_by_citation(bound_evidence)
            st.markdown(
                '<div class="ui-evidence-overview">'
                f'<strong>{count_text("evidence.supporting", len(bound_evidence), language)}</strong>'
                f'<span>{count_text("evidence.citations", len(citation_groups), language)}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            for citation, records in citation_groups:
                st.markdown(
                    '<div class="ui-evidence-citation-summary">'
                    f"<strong>{escape(citation)}</strong>"
                    f'<span>{count_text("evidence.excerpts", len(records), language)}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            if st.button(
                t("evidence.view_trace", language),
                key=f"open-evidence-{finding.finding_id}",
            ):
                st.session_state.selected_finding_id = finding.finding_id
                navigate(VIEW_EVIDENCE)

        with st.expander(t("navigation.technical_details", language), expanded=False):
            metadata_columns = st.columns(3)
            metadata_columns[0].caption(t("technical.rule_id", language))
            metadata_columns[0].code(
                finding.rule_id or t("value.not_recorded", language)
            )
            metadata_columns[1].caption(t("technical.rule_version", language))
            metadata_columns[1].code(
                finding.rule_version or t("value.not_recorded", language)
            )
            metadata_columns[2].caption(t("technical.issue_code", language))
            metadata_columns[2].code(finding.issue_code)
            st.caption(t("technical.facts_schema", language))
            st.code(schema_version, language=None)
            if finding.fact_refs:
                st.caption(t("technical.raw_facts", language))
                for fact_path in finding.fact_refs:
                    st.code(fact_path, language=None)
            if finding.reason_codes:
                st.caption(t("technical.raw_reasons", language))
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


def render_audit_evidence_card(
    evidence,
    *,
    record_number: int,
    language: str,
) -> None:
    """Render atomic Evidence as an audit-ready record."""

    label = t("evidence.official_item", language, number=record_number)
    with st.expander(label, expanded=False):
        st.markdown(
            '<div class="ui-evidence-trace-meta">'
            f'<span>{t("evidence.authority", language)} · '
            f'{t(f"authority.{evidence.authority_level.value}", language)}</span>'
            f'<span>{t("evidence.version", language)} · '
            f'{escape(evidence.document_version or t("evidence.not_recorded", language))}</span>'
            f'<span>{t("evidence.source", language)} · '
            f'{escape(instrument_label(evidence.legal_source, language))}</span>'
            f'<span>{t("evidence.citation", language)} · '
            f'{escape(evidence.citation)}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f'**{t("evidence.original", language)}**')
        st.write(evidence.excerpt)
        st.caption(t("evidence.stable_id", language))
        st.code(evidence.evidence_id, language=None)


def render_report(report: AssessmentReport, facts, language: str) -> None:
    """Present structured assessment results without evidence duplication."""

    render_section_header(
        t("report.title", language),
        eyebrow=t("report.eyebrow", language),
        description=t("report.copy", language),
    )
    framework_names = ", ".join(
        framework_label(framework, language)
        for framework in report.assessed_frameworks
    ) or t("report.no_framework", language)
    st.markdown(
        '<div class="ui-report-context">'
        f'<span>{count_text("report.findings", len(report.findings), language)}</span>'
        f'<span>{count_text("report.evidence", len(report.evidence), language)}</span>'
        f'<span>{count_text("report.gaps", len(report.missing_information), language)}</span>'
        f'<span>{framework_names}</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    primary_findings, _ = ordered_findings_for_presentation(report.findings)
    primary_findings = prioritize_selected_finding(
        primary_findings,
        st.session_state.get("selected_finding_id"),
    )
    primary_frameworks = {
        finding.framework for finding in primary_findings
    }
    primary_missing_information = [
        item
        for item in report.missing_information
        if item.framework in primary_frameworks
    ]
    render_section_header(t("report.findings.title", language))
    if not primary_findings:
        st.info(t("report.no_finding", language))
    for finding in primary_findings:
        render_result_finding_card(
            finding,
            evidence_for_finding(report, finding.finding_id),
            facts,
            facts.schema_version,
            language,
        )

    render_section_header(t("report.missing.title", language))
    if primary_missing_information:
        for item in primary_missing_information:
            render_missing_item(item, language, scope="primary")
    else:
        st.write(t("report.missing.none", language))

    primary_recommendations = scoped_recommendations(
        report,
        primary_findings,
        primary_missing_information,
        language,
    )
    if primary_recommendations:
        st.subheader(t("report.recommendations", language))
        for recommendation in primary_recommendations:
            render_recommendation(recommendation, scope="primary")

    other_frameworks = [
        framework
        for framework in report.assessed_frameworks
        if framework not in primary_frameworks
    ]
    if other_frameworks:
        with st.expander(t("framework_screens.title", language), expanded=False):
            st.caption(t("framework_screens.copy", language))
            for framework in other_frameworks:
                framework_findings = [
                    finding
                    for finding in report.findings
                    if finding.framework is framework
                ]
                framework_missing = [
                    item
                    for item in report.missing_information
                    if item.framework is framework
                ]
                framework_failures = [
                    failure
                    for failure in report.execution_failures
                    if failure.framework is framework
                ]
                status_key = framework_screen_status_key(
                    framework_findings,
                    framework_missing,
                    framework_failures,
                )
                st.markdown(f"#### {framework_label(framework, language)}")
                st.caption(t(status_key, language))
                for finding in framework_findings:
                    st.markdown(f"**{finding_title(finding, language)}**")
                    st.write(finding_summary(finding, language))
                if framework_missing:
                    st.markdown(f"**{t('framework_screen.missing', language)}**")
                    for item in framework_missing:
                        render_missing_item(item, language, scope="framework")
                framework_recommendations = scoped_recommendations(
                    report,
                    framework_findings,
                    framework_missing,
                    language,
                )
                if framework_recommendations:
                    st.markdown(f"**{t('report.recommendations', language)}**")
                    for recommendation in framework_recommendations:
                        render_recommendation(
                            recommendation,
                            scope="framework",
                        )

                st.caption(t("navigation.technical_details", language))
                for finding in framework_findings:
                    st.code(finding.rule_id, language=None)
                    st.code(finding.issue_code, language=None)
                    for reason_code in finding.reason_codes:
                        st.code(reason_code, language=None)
                for item in framework_missing:
                    st.code(item.fact_path, language=None)
                for failure in framework_failures:
                    st.code(failure.rule_id, language=None)
                st.divider()

    with st.expander(t("report.technical", language), expanded=False):
        st.caption(t("technical.report_id", language))
        st.code(report.report_id, language=None)
        st.caption(t(
            "technical.report_meta",
            language,
            version=report.report_version,
            engine=report.engine_version,
            generated=report.generated_at.isoformat(),
        ))
        if report.missing_information:
            st.caption(t("technical.raw_missing", language))
            for item in report.missing_information:
                st.code(item.fact_path, language=None)
        if report.recommendations:
            st.caption(t("technical.raw_recommendations", language))
            for recommendation in report.recommendations:
                st.code(recommendation, language=None)

def render_compliance_chain(
    finding,
    bound_evidence: list,
    facts,
    language: str,
) -> None:
    """Render one finding as a centered, sequential compliance trace."""

    with st.container():
        header_column, status_column = st.columns([4, 1])
        header_column.caption(t("evidence.trace.selected", language).upper())
        header_column.markdown(f"## {finding_title(finding, language)}")
        header_column.write(finding_summary(finding, language))
        with status_column:
            render_status_badge(finding.status, language)

    render_trace_connector()
    render_trace_stage(
        "01",
        t("trace.facts.title", language),
        t("trace.facts.copy", language),
    )
    with st.container():
        if finding.fact_refs:
            for fact_path in finding.fact_refs:
                st.markdown(
                    '<div class="ui-trace-fact-row">'
                    '<span class="ui-trace-fact-dot"></span>'
                    '<span class="ui-trace-fact-content">'
                    f"<strong>{escape(fact_label(fact_path, language))}</strong>"
                    f"<span>{escape(fact_value(facts, fact_path, language))}</span>"
                    "</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption(t("value.not_recorded", language))

    render_trace_connector()
    render_trace_stage(
        "02",
        t("trace.rule.title", language),
        t("trace.rule.copy", language),
    )
    with st.container():
        st.markdown(f"### {rule_label(finding.rule_id, language)}")
        if finding.trace:
            render_decision_path(finding, facts, language)
        else:
            st.caption(t("trace.rule.none", language))
        with st.expander(t("trace.rule.technical", language), expanded=False):
            st.caption(t("trace.rule.id_version", language))
            st.code(
                f'{finding.rule_id or t("value.not_recorded", language)} · '
                f'{finding.rule_version or t("value.not_recorded", language)}',
                language=None,
            )
            if finding.reason_codes:
                st.caption(t("technical.raw_reasons", language))
                for reason_code in finding.reason_codes:
                    st.code(reason_code, language=None)

    render_trace_connector()
    render_trace_stage(
        "03",
        t("trace.legal.title", language),
        t("trace.legal.copy", language),
    )
    with st.container():
        if finding.legal_basis:
            for basis in finding.legal_basis:
                st.markdown(
                    '<div class="ui-legal-basis-row">'
                    f'<span class="ui-legal-basis-citation">{escape(basis.citation)}</span>'
                    f'<span class="ui-legal-basis-instrument">{escape(instrument_label(basis.instrument, language))}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption(t("trace.legal.none", language))

    render_trace_connector()
    render_trace_stage(
        "04",
        t("trace.evidence.title", language),
        t("trace.evidence.copy", language),
    )
    if not bound_evidence:
        st.info(t("trace.evidence.none", language))
    for citation, records in group_evidence_by_citation(bound_evidence):
        st.markdown(
            '<section class="ui-evidence-group">'
            '<div class="ui-evidence-group__heading">'
            f"<strong>{escape(citation)}</strong>"
            f'<span>{count_text("evidence.excerpts", len(records), language)}</span>'
            "</div></section>",
            unsafe_allow_html=True,
        )
        for record_number, evidence in enumerate(records, start=1):
            render_audit_evidence_card(
                evidence,
                record_number=record_number,
                language=language,
            )

    with st.expander(t("evidence.technical", language), expanded=False):
        st.caption(t("technical.raw_facts", language))
        for fact_path in finding.fact_refs:
            st.code(fact_path, language=None)
        st.caption(t("evidence.all_ids", language))
        for evidence in bound_evidence:
            st.code(evidence.evidence_id, language=None)
            st.caption(t("evidence.raw_source", language))
            st.code(evidence.legal_source, language=None)


def render_evidence_workspace(
    report: AssessmentReport,
    facts,
    language: str,
) -> None:
    """Render finding-specific evidence relationships and stable identities."""

    render_section_header(
        t("evidence.trace.title", language),
        eyebrow=t("evidence.trace.eyebrow", language),
        description=t("evidence.trace.copy", language),
    )
    if not report.findings:
        st.info(t("evidence.trace.no_finding", language))
        return

    primary_findings, screened_out_findings = ordered_findings_for_presentation(
        report.findings
    )
    finding_by_id = {
        finding.finding_id: finding
        for finding in (*primary_findings, *screened_out_findings)
    }
    options = list(finding_by_id)
    selected_id = st.session_state.get("selected_finding_id")
    if selected_id not in finding_by_id:
        selected_id = options[0]
    selected_id = st.selectbox(
        t("evidence.trace.select", language),
        options=options,
        index=options.index(selected_id),
        format_func=lambda finding_id: finding_title(
            finding_by_id[finding_id],
            language,
        ),
    )
    st.session_state.selected_finding_id = selected_id
    finding = finding_by_id[selected_id]

    bound_evidence = evidence_for_finding(report, selected_id)
    _, chain_column, _ = st.columns([1, 6, 1])
    with chain_column:
        render_compliance_chain(finding, bound_evidence, facts, language)


def main() -> None:
    page_language = st.session_state.get("ui_language", DEFAULT_LANGUAGE)
    st.set_page_config(
        page_title=t("app.page_title", page_language),
        page_icon="⚖️",
        layout="wide",
    )
    apply_enterprise_styles()
    initialize_ui_state()

    try:
        bundle = get_workflow_bundle()
    except (OSError, ValueError) as exc:
        language = st.session_state.get("ui_language", DEFAULT_LANGUAGE)
        st.error(t("app.initialization_error", language, error=exc))
        st.stop()

    case_id = st.session_state.get("assessment_case_id")
    report = st.session_state.get("assessment_report")
    report = clear_mismatched_report(bundle, case_id, report)
    language = st.session_state.get("ui_language", DEFAULT_LANGUAGE)
    render_sidebar(bundle, case_id, report, language)
    language = st.session_state.get("ui_language", DEFAULT_LANGUAGE)
    view = st.session_state.get("assessment_view", VIEW_LANDING)

    if view == VIEW_LANDING:
        render_case_creation(bundle, language)
        return

    if case_id is None:
        st.session_state.assessment_view = VIEW_LANDING
        st.rerun()

    render_case_context(bundle, case_id, language)
    if view == VIEW_WORKSPACE:
        render_fact_collection(bundle, case_id, language)
        render_assessment_action(bundle, case_id, language)
        return

    if report is None:
        st.info(t("assessment.required", language))
        if st.button(t("assessment.return", language), type="primary"):
            navigate(VIEW_WORKSPACE)
        return

    facts = bundle.case_service.get_case(case_id).current_facts
    if view == VIEW_RESULTS:
        render_report(report, facts, language)
        return

    if view == VIEW_EVIDENCE:
        render_evidence_workspace(report, facts, language)
        return

    st.session_state.assessment_view = VIEW_LANDING
    st.rerun()


if __name__ == "__main__":
    main()
