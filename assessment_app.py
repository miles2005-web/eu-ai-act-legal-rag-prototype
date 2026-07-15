"""Minimal Streamlit interface for the structured assessment workflow."""

from __future__ import annotations

import json
from copy import deepcopy
from enum import Enum
from html import escape
from pathlib import Path

import streamlit as st

from scripts.run_demo_assessment import (
    build_assessment_facts,
    load_fixture,
)
from src.assessment.demo import AssessmentWorkflowBundle, create_assessment_workflow
from src.assessment.evidence import normalized_excerpt_hash
from src.assessment.facts import AffectedPerson, UseDomain
from src.assessment.findings import FindingStatus
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import TriState
from src.assessment.questionnaire import (
    FactProvenance,
    QuestionnaireRoute,
    QuestionResponseState,
    build_default_questionnaire_router,
    calculate_invalidations,
)
from src.assessment.questionnaire.models import AnswerType
from src.assessment.questionnaire.definitions import (
    AI_ACT_EMPLOYMENT_RULE_ID,
    AI_ACT_PRODUCT_SAFETY_RULE_ID,
    EU_DATA_ACT_RULE_ID,
    GDPR_ARTICLE22_RULE_ID,
)
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
from src.ui.questionnaire import (
    IMPLEMENTED_MODULE_IDS,
    ROUTING_HINT_IDS,
    UNANSWERED_WIDGET_VALUE,
    QuestionnaireAnswer,
    apply_question_answers,
    authorized_rule_ids_for_modules,
    clear_fact_paths,
    confirmed_module_gaps,
    confirmed_missing_fact_paths,
    current_answer,
    hints_from_normalization,
    localized_text_key,
    merge_provenance,
    module_definition,
    modules_for_question,
    question_definition,
    question_id_for_fact_path,
    question_option_label,
    remove_provenance,
    required_facts_complete,
    resolve_fact,
)


PROJECT_ROOT = Path(__file__).resolve().parent
INDUSTRIAL_FIXTURE_PATH = (
    PROJECT_ROOT / "tests" / "fixtures" / "industrial_ai_case.json"
)
VIEW_LANDING = "landing"
VIEW_WORKSPACE = "workspace"
VIEW_RESULTS = "results"
VIEW_EVIDENCE = "evidence"
RUN_MODE_COMPLETE = "complete"
RUN_MODE_WITH_GAPS = "with_gaps"
CASE_FORM_STATE_KEYS = (
    "assessment_fact_system_name",
    "assessment_fact_system_purpose",
    "assessment_fact_domain",
    "assessment_fact_task",
    "assessment_fact_material_influence",
    "assessment_fact_affected_persons",
    "assessment_fact_personal_data",
    "assessment_fact_connected_product",
    "assessment_fact_related_service",
    "assessment_confirm_ambiguous_task",
    "assessment_input_normalization",
    "assessment_routing_hints_widget",
    "assessment_manual_modules_widget",
)
QUESTIONNAIRE_STATE_KEYS = (
    "assessment_confirmed_modules",
    "assessment_declined_modules",
    "assessment_routing_hints",
    "assessment_fact_provenance",
    "assessment_questionnaire_route",
    "assessment_pending_widget_resets",
    "assessment_fact_save_notice",
    "assessment_edit_question_ids",
    "assessment_run_mode",
    "assessment_run_route",
    "assessment_focus_module_id",
    "assessment_focus_question_id",
)
QUESTION_WIDGET_STATE_KEYS = {
    "INTAKE-USE-TASK": "assessment_fact_task",
    "INTAKE-DECISION-IMPACT": "assessment_fact_material_influence",
    "INTAKE-PERSONAL-DATA": "assessment_fact_personal_data",
    "GDPR-AUTOMATED-DECISION": (
        "assessment_question::GDPR-AUTOMATED-DECISION"
    ),
    "INTAKE-CONNECTED-PRODUCT": "assessment_fact_connected_product",
    "INTAKE-RELATED-SERVICE": "assessment_fact_related_service",
    "DATA-ACT-DATA-GENERATED": (
        "assessment_question::DATA-ACT-DATA-GENERATED"
    ),
}
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
    st.session_state.setdefault("assessment_confirmed_modules", [])
    st.session_state.setdefault("assessment_declined_modules", [])
    st.session_state.setdefault("assessment_routing_hints", [])
    st.session_state.setdefault("assessment_fact_provenance", [])
    st.session_state.setdefault("assessment_questionnaire_route", None)
    st.session_state.setdefault("assessment_pending_widget_resets", [])
    st.session_state.setdefault("assessment_fact_save_notice", None)
    st.session_state.setdefault("assessment_edit_question_ids", [])
    st.session_state.setdefault("assessment_run_mode", None)
    st.session_state.setdefault("assessment_run_route", None)
    st.session_state.setdefault("assessment_focus_module_id", None)
    st.session_state.setdefault("assessment_focus_question_id", None)

    # Streamlit removes widget-owned keys when their widget is not rendered.
    # Keep the manual-module selection alive while a report/evidence view is
    # open so returning to the workspace cannot expose a stale widget tree.
    if st.session_state.get("assessment_view") != VIEW_WORKSPACE:
        st.session_state.assessment_manual_modules_widget = list(
            st.session_state.get("assessment_manual_modules_widget", [])
        )
        for key in tuple(st.session_state):
            if str(key).startswith("assessment_question::"):
                st.session_state[key] = st.session_state[key]


def navigate(view: str) -> None:
    """Navigate between workspace views without changing domain state."""

    st.session_state.assessment_view = view
    st.rerun()


def reset_case_dependent_state(*, view: str) -> AssessmentWorkflowBundle:
    """Replace all case-owned state while preserving presentation preferences."""

    for key in CASE_FORM_STATE_KEYS:
        st.session_state.pop(key, None)
    for key in QUESTIONNAIRE_STATE_KEYS:
        st.session_state.pop(key, None)
    for key in tuple(st.session_state):
        if str(key).startswith("assessment_question::"):
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
    st.session_state.assessment_confirmed_modules = []
    st.session_state.assessment_declined_modules = []
    st.session_state.assessment_routing_hints = []
    st.session_state.assessment_fact_provenance = []
    st.session_state.assessment_questionnaire_route = None
    st.session_state.assessment_pending_widget_resets = []
    st.session_state.assessment_fact_save_notice = None
    st.session_state.assessment_edit_question_ids = []
    st.session_state.assessment_run_mode = None
    st.session_state.assessment_run_route = None
    st.session_state.assessment_focus_module_id = None
    st.session_state.assessment_focus_question_id = None
    return bundle


def replace_workflow_bundle_preserving_case(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    facts,
) -> AssessmentWorkflowBundle:
    """Clear run-owned state while preserving the active case identity."""

    assessment_case = bundle.case_service.get_case(case_id)
    replacement = create_assessment_workflow()
    replacement.case_service.create_case(
        assessment_case.name,
        description=assessment_case.description,
        facts=facts,
        case_id=assessment_case.case_id,
    )
    st.session_state.assessment_workflow_bundle = replacement
    st.session_state.assessment_report = None
    st.session_state.selected_finding_id = None
    st.session_state.assessment_run_mode = None
    st.session_state.assessment_run_route = None
    if st.session_state.get("assessment_view") in (VIEW_RESULTS, VIEW_EVIDENCE):
        st.session_state.assessment_view = VIEW_WORKSPACE
    return replacement


def clear_assessment_output() -> None:
    """Clear all presentation state derived from an older fact route."""

    st.session_state.assessment_report = None
    st.session_state.selected_finding_id = None
    st.session_state.assessment_run_mode = None
    st.session_state.assessment_run_route = None
    if st.session_state.get("assessment_view") in (VIEW_RESULTS, VIEW_EVIDENCE):
        st.session_state.assessment_view = VIEW_WORKSPACE


def questionnaire_router():
    """Return the deterministic Phase 1 router used by every UI scenario."""

    return build_default_questionnaire_router()


def current_questionnaire_route(facts) -> QuestionnaireRoute:
    """Compute and persist the current route using canonical state only."""

    route = questionnaire_router().route(
        facts,
        confirmed_modules=st.session_state.get(
            "assessment_confirmed_modules", []
        ),
        confirmed_routing_hints=st.session_state.get(
            "assessment_routing_hints", []
        ),
        fact_provenance=_provenance(),
    )
    st.session_state.assessment_questionnaire_route = route
    return route


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
    if assessment_run.case_id != case_id:
        return False
    authorized_rule_ids = authorized_rule_ids_for_modules(
        st.session_state.get("assessment_confirmed_modules", [])
    )
    return assessment_run.input_fingerprint == bundle.workflow.input_fingerprint(
        case_id,
        rule_ids=authorized_rule_ids,
    )


def authorized_findings(
    report: AssessmentReport,
    route: QuestionnaireRoute | None,
) -> list:
    """Return Findings produced by modules explicitly authorized for the run."""

    if route is None:
        return list(report.findings)
    confirmed = frozenset(route.confirmed_modules)
    return [finding for finding in report.findings if finding.rule_id in confirmed]


def is_incomplete_assessment_result(
    report: AssessmentReport | None,
    route: QuestionnaireRoute | None,
) -> bool:
    """Identify a deliberate gap run that produced no authorized Finding."""

    return bool(
        report is not None
        and st.session_state.get("assessment_run_mode") == RUN_MODE_WITH_GAPS
        and not authorized_findings(report, route)
    )


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
    st.session_state.assessment_run_mode = None
    st.session_state.assessment_run_route = None
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


def _preloaded_provenance(facts, module_id: str) -> list[FactProvenance]:
    """Create the same dependency metadata for fixture facts as UI answers."""

    records: list[FactProvenance] = []
    definition = module_definition(module_id)
    for question_id in definition.question_ids:
        metadata = question_definition(question_id)
        value = resolve_fact(facts, metadata.fact_path)
        if value is None or value is TriState.UNKNOWN or value is UseDomain.UNKNOWN:
            continue
        records.append(
            FactProvenance(
                fact_path=metadata.fact_path,
                question_id=question_id,
                module_id=module_id,
                explicitly_confirmed=True,
                depends_on=tuple(
                    dependency.fact_path for dependency in metadata.dependencies
                ),
            )
        )
    return records


def load_recruitment_demo() -> None:
    """Create a case from the existing pure-data recruitment demo fixture."""

    payload = load_fixture()
    bundle = reset_case_dependent_state(view=VIEW_WORKSPACE)
    facts = build_assessment_facts(payload["facts"])
    assessment_case = bundle.case_service.create_case(
        payload["scenario"]["name"],
        description=payload["scenario"]["description"],
        facts=facts,
    )
    st.session_state.assessment_case_id = assessment_case.case_id
    st.session_state.demo_loaded = True
    st.session_state.demo_scenario = "recruitment"
    st.session_state.demo_scenario_id = payload["scenario_id"]
    st.session_state.assessment_confirmed_modules = [
        AI_ACT_EMPLOYMENT_RULE_ID
    ]
    st.session_state.assessment_routing_hints = [
        "employment.recruitment",
        "employment.candidate_ranking",
    ]
    st.session_state.assessment_fact_provenance = _preloaded_provenance(
        facts,
        AI_ACT_EMPLOYMENT_RULE_ID,
    )
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
    st.session_state.assessment_confirmed_modules = [EU_DATA_ACT_RULE_ID]
    st.session_state.assessment_routing_hints = [
        "data_act.industrial_connected_equipment"
    ]
    st.session_state.assessment_fact_provenance = _preloaded_provenance(
        facts,
        EU_DATA_ACT_RULE_ID,
    )
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
        run_route = st.session_state.get("assessment_run_route")
        incomplete_result = is_incomplete_assessment_result(report, run_route)
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
            disabled=report is None or incomplete_result,
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
            route = current_questionnaire_route(facts)
            facts_complete = required_facts_complete(route)
        assessment_complete = bool(
            case_id is not None
            and report is not None
            and report_belongs_to_case(bundle, report, case_id)
        )

        statuses = (
            ("case", t("progress.case", language), case_complete),
            ("facts", t("progress.facts", language), facts_complete),
            (
                "assessment",
                t("progress.assessment", language),
                assessment_complete,
            )
        )
        for step_id, label, complete in statuses:
            state = t(
                "progress.complete" if complete else "progress.pending",
                language,
            )
            icon = "✓" if complete else "○"
            st.markdown(
                '<div class="ui-progress-row" '
                f'data-progress-step="{step_id}" '
                f'data-progress-state="{"complete" if complete else "pending"}">'
                f'<span aria-hidden="true">{icon}</span>'
                f'<span>{label}</span><span class="ui-progress-state">{state}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
        if assessment_complete and incomplete_result:
            st.caption(t("progress.assessment.incomplete", language))
        st.progress(
            sum(complete for _, _, complete in statuses) / len(statuses)
        )

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


def _question_label(question_id: str, language: str) -> str:
    return t(localized_text_key(question_id, language), language)


def _question_help(question_id: str, language: str) -> str | None:
    value = t(localized_text_key(question_id, language, help_text=True), language)
    return None if value.endswith((".help.en", ".help.zh_cn")) else value


def _hint_label(hint_id: str, language: str) -> str:
    return t_or(f"routing_hint.{hint_id}", hint_id, language)


def _module_label(module_id: str, language: str) -> str:
    return t_or(module_definition(module_id).display_module_key, module_id, language)


def _module_framework_label(module_id: str, language: str) -> str:
    return framework_label(module_definition(module_id).framework, language)


def _module_boundary_note(module_id: str, language: str) -> str | None:
    key = module_definition(module_id).boundary_note_key
    return t(key, language) if key else None


def _provenance() -> list[FactProvenance]:
    records = st.session_state.get("assessment_fact_provenance", [])
    return [record for record in records if isinstance(record, FactProvenance)]


def _store_fact_update(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    previous_facts,
    updated_facts,
    provenance_updates: list[FactProvenance],
) -> AssessmentWorkflowBundle:
    """Apply dependency invalidation and persist one canonical fact update."""

    existing_provenance = _provenance()
    invalidation = calculate_invalidations(
        previous_facts,
        updated_facts,
        existing_provenance,
    )
    clear_fact_paths(updated_facts, invalidation.stale_fact_paths)
    stale_paths = frozenset(invalidation.stale_fact_paths)
    provenance_updates = [
        record
        for record in provenance_updates
        if record.fact_path not in stale_paths
    ]
    provenance = remove_provenance(
        existing_provenance,
        invalidation.removed_provenance_fact_paths,
    )
    provenance = merge_provenance(provenance, provenance_updates)
    provenance = [
        record
        for record in provenance
        if record.response_state is QuestionResponseState.EXPLICIT_UNKNOWN
        or (
            (value := resolve_fact(updated_facts, record.fact_path)) is not None
            and value is not TriState.UNKNOWN
            and value is not UseDomain.UNKNOWN
        )
    ]
    st.session_state.assessment_fact_provenance = provenance
    invalidated_modules = frozenset(invalidation.invalidated_module_ids)
    st.session_state.assessment_confirmed_modules = [
        module_id
        for module_id in st.session_state.get("assessment_confirmed_modules", [])
        if module_id not in invalidated_modules
    ]
    clear_assessment_output()
    bundle = replace_workflow_bundle_preserving_case(
        bundle,
        case_id,
        updated_facts,
    )
    if invalidation.changed_upstream_fact_paths:
        pending_resets = list(
            st.session_state.get("assessment_pending_widget_resets", [])
        )
        for question_id in invalidation.invalidated_question_ids:
            widget_key = QUESTION_WIDGET_STATE_KEYS.get(
                question_id,
                f"assessment_question::{question_id}",
            )
            if widget_key not in pending_resets:
                pending_resets.append(widget_key)
        st.session_state.assessment_pending_widget_resets = pending_resets
    invalidated_questions = frozenset(invalidation.invalidated_question_ids)
    st.session_state.assessment_edit_question_ids = [
        question_id
        for question_id in st.session_state.get(
            "assessment_edit_question_ids", []
        )
        if question_id not in invalidated_questions
    ]
    current_questionnaire_route(updated_facts)
    return bundle


def _render_universal_intake(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    facts,
    language: str,
) -> None:
    """Render the small regulation-neutral intake using authored questions."""

    for widget_key in st.session_state.get(
        "assessment_pending_widget_resets", []
    ):
        st.session_state.pop(widget_key, None)
    st.session_state.assessment_pending_widget_resets = []
    notice = st.session_state.get("assessment_fact_save_notice")
    if notice:
        if notice == "normalization.saved_unknown":
            st.info(t(notice, language))
        else:
            st.success(t(notice, language))
        st.session_state.assessment_fact_save_notice = None
    sync_predefined_task_input(facts, language)
    domains = list(UseDomain)
    tri_states = list(TriState)
    affected_people = list(AffectedPerson)
    if "assessment_routing_hints_widget" not in st.session_state:
        st.session_state.assessment_routing_hints_widget = list(
            st.session_state.get("assessment_routing_hints", [])
        )

    with st.form("assessment_facts"):
        system_name = st.text_input(
            _question_label("INTAKE-SYSTEM-NAME", language),
            value=facts.system.name or "",
            help=_question_help("INTAKE-SYSTEM-NAME", language),
            key="assessment_fact_system_name",
        )
        system_purpose = st.text_input(
            _question_label("INTAKE-SYSTEM-PURPOSE", language),
            value=facts.system.intended_purpose or "",
            help=_question_help("INTAKE-SYSTEM-PURPOSE", language),
            key="assessment_fact_system_purpose",
        )
        domain = st.selectbox(
            _question_label("INTAKE-USE-DOMAIN", language),
            options=domains,
            index=domains.index(facts.use_context.domain),
            format_func=lambda value: domain_label(value, language),
            help=_question_help("INTAKE-USE-DOMAIN", language),
            key="assessment_fact_domain",
        )
        task_options = {}
        if "assessment_fact_task" not in st.session_state:
            task_options["value"] = task_input_value(facts, language)
        task = st.text_area(
            _question_label("INTAKE-USE-TASK", language),
            placeholder=t("facts.task.placeholder", language),
            help=_question_help("INTAKE-USE-TASK", language),
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
            st.info(t("normalization.ambiguous", language))
        materially_influences = st.selectbox(
            _question_label("INTAKE-DECISION-IMPACT", language),
            options=tri_states,
            index=tri_states.index(facts.use_context.materially_influences_decision),
            format_func=lambda value: tri_state_label(value, language),
            help=_question_help("INTAKE-DECISION-IMPACT", language),
            key="assessment_fact_material_influence",
        )
        affected_persons = st.multiselect(
            _question_label("INTAKE-AFFECTED-PERSONS", language),
            options=affected_people,
            default=facts.use_context.affected_persons or [],
            format_func=lambda value: t_or(
                f"affected_person.{value.value}", value.value, language
            ),
            help=_question_help("INTAKE-AFFECTED-PERSONS", language),
            key="assessment_fact_affected_persons",
        )
        personal_data = st.selectbox(
            _question_label("INTAKE-PERSONAL-DATA", language),
            options=tri_states,
            index=tri_states.index(facts.data_protection.personal_data_processed),
            format_func=lambda value: tri_state_label(value, language),
            help=_question_help("INTAKE-PERSONAL-DATA", language),
            key="assessment_fact_personal_data",
        )
        connected_product = st.selectbox(
            _question_label("INTAKE-CONNECTED-PRODUCT", language),
            options=tri_states,
            index=tri_states.index(facts.data_act.connected_product),
            format_func=lambda value: tri_state_label(value, language),
            help=_question_help("INTAKE-CONNECTED-PRODUCT", language),
            key="assessment_fact_connected_product",
        )
        related_service = st.selectbox(
            _question_label("INTAKE-RELATED-SERVICE", language),
            options=tri_states,
            index=tri_states.index(facts.data_act.related_service),
            format_func=lambda value: tri_state_label(value, language),
            help=_question_help("INTAKE-RELATED-SERVICE", language),
            key="assessment_fact_related_service",
        )
        routing_hints = st.multiselect(
            t("questionnaire.hints.label", language),
            options=list(ROUTING_HINT_IDS),
            format_func=lambda value: _hint_label(value, language),
            help=t("questionnaire.hints.help", language),
            key="assessment_routing_hints_widget",
        )
        facts_submitted = st.form_submit_button(
            t("facts.save", language),
            type="primary",
        )

    if not facts_submitted:
        return

    previous_facts = deepcopy(facts)
    updated_facts = deepcopy(facts)
    answers = [
        QuestionnaireAnswer("INTAKE-SYSTEM-NAME", system_name, system_name),
        QuestionnaireAnswer(
            "INTAKE-SYSTEM-PURPOSE", system_purpose, system_purpose
        ),
        QuestionnaireAnswer("INTAKE-USE-DOMAIN", domain),
        QuestionnaireAnswer(
            "INTAKE-AFFECTED-PERSONS", affected_persons
        ),
        QuestionnaireAnswer(
            "INTAKE-DECISION-IMPACT", materially_influences
        ),
        QuestionnaireAnswer("INTAKE-PERSONAL-DATA", personal_data),
        QuestionnaireAnswer("INTAKE-CONNECTED-PRODUCT", connected_product),
        QuestionnaireAnswer("INTAKE-RELATED-SERVICE", related_service),
    ]
    provenance_updates = apply_question_answers(updated_facts, answers)
    protected_paths = frozenset(
        path
        for path, value in (
            ("use_context.materially_influences_decision", materially_influences),
            ("data_protection.personal_data_processed", personal_data),
            ("data_act.connected_product", connected_product),
            ("data_act.related_service", related_service),
        )
        if value is not TriState.UNKNOWN
    )
    apply_normalized_input(
        updated_facts,
        normalization_preview,
        ambiguous_text_confirmed=confirm_ambiguous_task,
        protected_fact_paths=protected_paths,
    )
    normalized_answers = []
    for fact_path in normalization_preview.fact_updates:
        if fact_path in protected_paths:
            continue
        normalized_answers.append(
            QuestionnaireAnswer(
                question_id_for_fact_path(fact_path),
                resolve_fact(updated_facts, fact_path),
                task,
            )
        )
    if normalized_answers:
        provenance_updates.extend(
            apply_question_answers(updated_facts, normalized_answers)
        )
    if updated_facts.use_context.task is not None:
        provenance_updates.extend(
            apply_question_answers(
                updated_facts,
                [
                    QuestionnaireAnswer(
                        "INTAKE-USE-TASK",
                        updated_facts.use_context.task,
                        task,
                    )
                ],
            )
        )
    st.session_state.assessment_input_normalization = {
        **normalization_preview.to_dict(),
        "ambiguous_text_confirmed": confirm_ambiguous_task,
    }
    normalized_hints = hints_from_normalization(normalization_preview.mapping_ids)
    st.session_state.assessment_routing_hints = list(
        dict.fromkeys([*routing_hints, *normalized_hints])
    )
    _store_fact_update(
        bundle,
        case_id,
        previous_facts,
        updated_facts,
        provenance_updates,
    )
    if (
        normalization_preview.status is NormalizationStatus.AMBIGUOUS
        and not confirm_ambiguous_task
    ):
        st.session_state.assessment_fact_save_notice = (
            "normalization.saved_unknown"
        )
    else:
        st.session_state.assessment_fact_save_notice = "facts.saved"
    st.rerun()


def _confirm_module(module_id: str) -> None:
    confirmed = list(st.session_state.get("assessment_confirmed_modules", []))
    if module_id not in confirmed:
        confirmed.append(module_id)
    st.session_state.assessment_confirmed_modules = confirmed
    st.session_state.assessment_declined_modules = [
        item
        for item in st.session_state.get("assessment_declined_modules", [])
        if item != module_id
    ]
    clear_assessment_output()


def _decline_module(module_id: str) -> None:
    st.session_state.assessment_confirmed_modules = [
        item
        for item in st.session_state.get("assessment_confirmed_modules", [])
        if item != module_id
    ]
    declined = list(st.session_state.get("assessment_declined_modules", []))
    if module_id not in declined:
        declined.append(module_id)
    st.session_state.assessment_declined_modules = declined
    st.session_state.assessment_fact_provenance = [
        record
        for record in _provenance()
        if record.module_id != module_id
    ]
    module_question_ids = frozenset(module_definition(module_id).question_ids)
    st.session_state.assessment_edit_question_ids = [
        question_id
        for question_id in st.session_state.get(
            "assessment_edit_question_ids", []
        )
        if question_id not in module_question_ids
    ]
    clear_assessment_output()


def _render_question_widget(
    question,
    facts,
    language: str,
    *,
    key_prefix: str,
    explicitly_answered: bool = False,
):
    definition = question_definition(question.question_id)
    label = _question_label(question.question_id, language)
    help_text = _question_help(question.question_id, language)
    current = current_answer(facts, question)
    key = f"{key_prefix}::{question.question_id}"
    if question.answer_type is AnswerType.TRI_STATE:
        authored_options = [option.value for option in question.options]
        options = [UNANSWERED_WIDGET_VALUE, *authored_options]
        value = (
            current
            if current in authored_options
            and (
                current != TriState.UNKNOWN.value
                or explicitly_answered
            )
            else UNANSWERED_WIDGET_VALUE
        )
        return st.selectbox(
            label,
            options=options,
            index=options.index(value),
            format_func=lambda option: (
                t("value.not_answered", language)
                if option == UNANSWERED_WIDGET_VALUE
                else t(f"value.{option}", language)
            ),
            help=help_text,
            key=key,
        )
    if question.answer_type is AnswerType.SINGLE_CHOICE:
        authored_options = [option.value for option in question.options]
        options = [UNANSWERED_WIDGET_VALUE, *authored_options]
        value = (
            current
            if current in authored_options
            else (
                authored_options[0]
                if explicitly_answered
                else UNANSWERED_WIDGET_VALUE
            )
        )
        return st.selectbox(
            label,
            options=options,
            index=options.index(value),
            format_func=lambda option: (
                t("value.not_answered", language)
                if option == UNANSWERED_WIDGET_VALUE
                else question_option_label(
                    question.question_id,
                    option,
                    language,
                )
                or t_or(
                    next(
                        item.label
                        for item in question.options
                        if item.value == option
                    ),
                    option,
                    language,
                )
            ),
            help=help_text,
            key=key,
        )
    if question.answer_type is AnswerType.MULTIPLE_CHOICE:
        options = [option.value for option in question.options]
        return st.multiselect(
            label,
            options=options,
            default=current or [],
            format_func=lambda option: t_or(
                next(item.label for item in question.options if item.value == option),
                option,
                language,
            ),
            help=help_text,
            key=key,
        )
    if question.answer_type is AnswerType.TEXT:
        return st.text_input(
            label,
            value=current or "",
            help=help_text,
            key=key,
        )
    raise ValueError(
        f"unsupported questionnaire answer type {definition.answer_type.value!r}"
    )


def _render_routed_follow_ups(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    facts,
    route: QuestionnaireRoute,
    language: str,
) -> None:
    follow_ups = [
        question
        for question in route.next_questions
        if not question_definition(question.question_id).universal
    ]
    recorded_unknown_ids = list(route.recorded_unknown_question_ids)
    editing_ids = [
        question_id
        for question_id in st.session_state.get(
            "assessment_edit_question_ids", []
        )
        if question_id in recorded_unknown_ids
    ]
    follow_up_ids = {question.question_id for question in follow_ups}
    for question_id in editing_ids:
        if question_id not in follow_up_ids:
            follow_ups.append(question_definition(question_id).as_question())
            follow_up_ids.add(question_id)
    if recorded_unknown_ids:
        for question_id in recorded_unknown_ids:
            label_column, action_column = st.columns([4, 1])
            label_column.markdown(
                f"**{escape(_question_label(question_id, language))}**  "
                f"\n{escape(t('questionnaire.response.recorded_unknown', language))}"
            )
            if action_column.button(
                t("questionnaire.response.edit", language),
                key=f"edit_answer::{question_id}",
                disabled=question_id in editing_ids,
                use_container_width=True,
            ):
                st.session_state.assessment_edit_question_ids = [
                    *editing_ids,
                    question_id,
                ]
                clear_assessment_output()
                st.rerun()
    if not follow_ups:
        if recorded_unknown_ids:
            st.info(t("questionnaire.followups.unresolved_unknown", language))
        else:
            st.caption(t("questionnaire.followups.none", language))
        return
    render_section_header(
        t("questionnaire.followups.title", language),
        description=t("questionnaire.followups.copy", language),
    )
    answers: dict[str, object] = {}
    with st.form("assessment_follow_up_questions"):
        for question in follow_ups:
            answers[question.question_id] = _render_question_widget(
                question,
                facts,
                language,
                key_prefix="assessment_question",
                explicitly_answered=(
                    question.question_id in recorded_unknown_ids
                ),
            )
        submitted = st.form_submit_button(
            t("questionnaire.followups.save", language),
            type="primary",
        )
    if not submitted:
        return
    previous_facts = deepcopy(facts)
    updated_facts = deepcopy(facts)
    provenance_updates: list[FactProvenance] = []
    for question in follow_ups:
        answer_value = answers[question.question_id]
        if answer_value == UNANSWERED_WIDGET_VALUE:
            continue
        if (
            question.answer_type is AnswerType.TEXT
            and (not isinstance(answer_value, str) or not answer_value.strip())
        ):
            continue
        owners = [
            module_id
            for module_id in route.confirmed_modules
            if module_id in modules_for_question(question.question_id)
        ]
        provenance_updates.extend(
            apply_question_answers(
                updated_facts,
                [QuestionnaireAnswer(question.question_id, answer_value)],
                module_id=owners[0] if len(owners) == 1 else None,
                record_explicit_unknown=True,
            )
        )
    if not provenance_updates:
        st.info(t("questionnaire.followups.no_changes", language))
        return
    _store_fact_update(
        bundle,
        case_id,
        previous_facts,
        updated_facts,
        provenance_updates,
    )
    st.session_state.assessment_fact_save_notice = (
        "questionnaire.followups.saved"
    )
    submitted_question_ids = {
        record.question_id for record in provenance_updates
    }
    st.session_state.assessment_edit_question_ids = [
        question_id
        for question_id in editing_ids
        if question_id not in submitted_question_ids
    ]
    st.rerun()


def _render_module_routing(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    facts,
    language: str,
) -> None:
    route = current_questionnaire_route(facts)
    declined = frozenset(st.session_state.get("assessment_declined_modules", []))
    suggestions = [
        module_id for module_id in route.suggested_modules if module_id not in declined
    ]

    render_section_header(
        t("questionnaire.modules.title", language),
        eyebrow=t("questionnaire.modules.eyebrow", language),
        description=t("questionnaire.modules.copy", language),
    )
    st.markdown(f"### {t('questionnaire.suggested.title', language)}")
    if not suggestions:
        st.caption(t("questionnaire.suggested.none", language))
    for module_id in suggestions:
        st.markdown(
            f"**{escape(_module_label(module_id, language))}**  "
            f"\n{escape(_module_framework_label(module_id, language))}"
        )
        if boundary_note := _module_boundary_note(module_id, language):
            st.caption(boundary_note)
        reasons = route.routing_reasons.get(module_id, [])
        with st.expander(t("questionnaire.why_suggested", language), expanded=False):
            for reason in reasons:
                st.write(t_or(f"routing_reason.{reason}", reason, language))
        confirm_column, decline_column = st.columns(2)
        if confirm_column.button(
            t("questionnaire.confirm", language),
            key=f"confirm_module::{module_id}",
            type="primary",
            use_container_width=True,
        ):
            _confirm_module(module_id)
            st.rerun()
        if decline_column.button(
            t("questionnaire.decline", language),
            key=f"decline_module::{module_id}",
            use_container_width=True,
        ):
            _decline_module(module_id)
            st.rerun()

    st.markdown(f"### {t('questionnaire.confirmed.title', language)}")
    if not route.confirmed_modules:
        st.caption(t("questionnaire.confirmed.none", language))
    for module_id in route.confirmed_modules:
        label_column, action_column = st.columns([4, 1])
        label_column.markdown(
            f"**{escape(_module_label(module_id, language))}**  "
            f"\n{escape(_module_framework_label(module_id, language))}"
        )
        if boundary_note := _module_boundary_note(module_id, language):
            label_column.caption(boundary_note)
        if action_column.button(
            t("questionnaire.remove", language),
            key=f"remove_module::{module_id}",
            use_container_width=True,
        ):
            _decline_module(module_id)
            st.rerun()

    with st.expander(t("questionnaire.manual.title", language), expanded=False):
        manual = st.multiselect(
            t("questionnaire.manual.label", language),
            options=list(IMPLEMENTED_MODULE_IDS),
            format_func=lambda value: _module_label(value, language),
            key="assessment_manual_modules_widget",
        )
        if st.button(
            t("questionnaire.manual.confirm", language),
            key="confirm_manual_modules",
            disabled=not manual,
        ):
            for module_id in manual:
                _confirm_module(module_id)
            st.rerun()

    st.markdown(f"### {t('questionnaire.unsupported.title', language)}")
    if not route.unsupported_modules:
        st.caption(t("questionnaire.unsupported.none", language))
    for unsupported in route.unsupported_modules:
        message_key = (
            unsupported.message_keys.zh_cn_label_key
            if language == "zh-CN"
            else unsupported.message_keys.en_label_key
        )
        help_key = (
            unsupported.message_keys.zh_cn_help_key
            if language == "zh-CN"
            else unsupported.message_keys.en_help_key
        )
        st.warning(
            f"**{t_or(unsupported.display_module_key, unsupported.path_id, language)}**\n\n"
            f"{t(message_key, language)} {t(help_key, language)}"
        )

    with st.expander(t("questionnaire.screened.title", language), expanded=False):
        if not route.screened_out_modules:
            st.caption(t("questionnaire.screened.none", language))
        for module_id in route.screened_out_modules:
            st.write(_module_label(module_id, language))

    _render_routed_follow_ups(bundle, case_id, facts, route, language)


def render_fact_collection(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    language: str,
) -> None:
    """Collect universal facts and route only relevant legal modules."""

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
    _render_universal_intake(bundle, case_id, facts, language)
    latest_bundle = get_workflow_bundle()
    latest_facts = latest_bundle.case_service.get_case(case_id).current_facts
    _render_module_routing(latest_bundle, case_id, latest_facts, language)


def render_assessment_action(
    bundle: AssessmentWorkflowBundle,
    case_id: str,
    language: str,
) -> None:
    """Execute only explicitly confirmed modules through the existing workflow."""

    render_section_header(
        t("assessment.title", language),
        eyebrow=t("assessment.eyebrow", language),
        description=t("assessment.copy", language),
    )
    confirmed_modules = list(
        st.session_state.get("assessment_confirmed_modules", [])
    )
    facts = bundle.case_service.get_case(case_id).current_facts
    route = current_questionnaire_route(facts)
    unresolved_fact_paths = confirmed_missing_fact_paths(route)
    has_confirmed_modules = bool(confirmed_modules)
    has_information_gaps = bool(unresolved_fact_paths)
    if not confirmed_modules:
        st.info(t("assessment.confirm_module_first", language))
    run_normal = st.button(
        t("assessment.run", language),
        type="primary",
        use_container_width=True,
        disabled=not has_confirmed_modules or has_information_gaps,
    )
    run_with_gaps = False
    if has_confirmed_modules and has_information_gaps:
        run_with_gaps = st.button(
            t("assessment.run_with_gaps", language),
            type="secondary",
            use_container_width=True,
        )
    if run_normal or run_with_gaps:
        authorized_rule_ids = authorized_rule_ids_for_modules(
            route.confirmed_modules,
        )
        with st.spinner(t("assessment.running", language)):
            st.session_state.assessment_report = bundle.workflow.run(
                case_id,
                rule_ids=authorized_rule_ids,
            )
        st.session_state.assessment_run_mode = (
            RUN_MODE_WITH_GAPS if run_with_gaps else RUN_MODE_COMPLETE
        )
        st.session_state.assessment_run_route = deepcopy(route)
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
            st.info(
                t(
                    "evidence.pending_binding"
                    if finding.legal_basis
                    else "evidence.none",
                    language,
                )
            )
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


def scoped_report_missing_information(
    report: AssessmentReport,
    route: QuestionnaireRoute,
) -> list:
    """Limit presentation gaps to current requirements of confirmed modules."""

    required_by_rule = {
        module_id: frozenset(route.missing_fact_paths.get(module_id, []))
        for module_id in route.confirmed_modules
    }
    return [
        item
        for item in report.missing_information
        if item.rule_id in required_by_rule
        and item.fact_path in required_by_rule[item.rule_id]
    ]


def render_incomplete_assessment(
    report: AssessmentReport,
    facts,
    route: QuestionnaireRoute,
    language: str,
) -> None:
    """Present a deliberate gap run without implying a legal conclusion."""

    module_gaps = tuple(
        gap
        for gap in confirmed_module_gaps(route, facts)
        if gap.unresolved or gap.blocked
    )
    unresolved_count = sum(gap.unresolved_count for gap in module_gaps)
    frameworks = list(
        dict.fromkeys(
            module_definition(gap.module_id).framework for gap in module_gaps
        )
    )
    framework_names = ", ".join(
        framework_label(framework, language) for framework in frameworks
    ) or t("report.no_framework", language)

    st.warning(t("assessment.status.with_gaps", language))
    render_section_header(
        t("incomplete.title", language),
        eyebrow=t("incomplete.eyebrow", language),
        description=t("incomplete.summary", language),
    )
    st.markdown(
        '<div class="ui-report-context" data-assessment-state="incomplete">'
        f'<span>{count_text("report.gaps", unresolved_count, language)}</span>'
        f'<span>{framework_names}</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    for gap in module_gaps:
        st.markdown(f"### {_module_label(gap.module_id, language)}")
        if boundary_note := _module_boundary_note(gap.module_id, language):
            st.caption(boundary_note)
        st.markdown(f"#### {t('incomplete.unresolved.title', language)}")
        for item in gap.unresolved:
            st.markdown(f"**{escape(_question_label(item.question_id, language))}**")
            st.caption(
                t(
                    "incomplete.recorded_answer"
                    if item.recorded_unknown
                    else "incomplete.awaiting_answer",
                    language,
                )
            )
        if gap.blocked:
            st.markdown(f"#### {t('incomplete.blocked.title', language)}")
            st.caption(t("incomplete.blocked.copy", language))
            for item in gap.blocked:
                st.markdown(f"- {escape(_question_label(item.question_id, language))}")

    st.info(t("incomplete.evidence", language))

    first_gap = next(
        (
            (gap.module_id, item.question_id)
            for gap in module_gaps
            for item in gap.unresolved
        ),
        (AI_ACT_PRODUCT_SAFETY_RULE_ID, None),
    )
    edit_column, return_column, later_column = st.columns(3)
    if edit_column.button(
        t("incomplete.edit", language),
        type="primary",
        use_container_width=True,
    ):
        module_id, question_id = first_gap
        if question_id is not None:
            editing = list(
                st.session_state.get("assessment_edit_question_ids", [])
            )
            if question_id not in editing:
                editing.append(question_id)
            st.session_state.assessment_edit_question_ids = editing
        st.session_state.assessment_focus_module_id = module_id
        st.session_state.assessment_focus_question_id = question_id
        navigate(VIEW_WORKSPACE)
    if return_column.button(
        t("incomplete.return", language),
        use_container_width=True,
    ):
        module_id, question_id = first_gap
        st.session_state.assessment_focus_module_id = module_id
        st.session_state.assessment_focus_question_id = question_id
        navigate(VIEW_WORKSPACE)
    if later_column.button(
        t("incomplete.later", language),
        use_container_width=True,
    ):
        navigate(VIEW_LANDING)


def render_report(
    report: AssessmentReport,
    facts,
    route: QuestionnaireRoute,
    language: str,
) -> None:
    """Present structured assessment results without evidence duplication."""

    if st.session_state.get("assessment_run_mode") == RUN_MODE_WITH_GAPS:
        st.warning(t("assessment.status.with_gaps", language))
    else:
        st.success(t("assessment.status.generated", language))
    render_section_header(
        t("report.title", language),
        eyebrow=t("report.eyebrow", language),
        description=t("report.copy", language),
    )
    report_findings = authorized_findings(report, route)
    scoped_missing = scoped_report_missing_information(report, route)
    confirmed_frameworks = list(
        dict.fromkeys(
            module_definition(module_id).framework
            for module_id in route.confirmed_modules
        )
    )
    authorized_finding_ids = {
        finding.finding_id for finding in report_findings
    }
    authorized_evidence_ids = {
        evidence_id
        for binding in report.evidence_bindings
        if binding.finding_id in authorized_finding_ids
        for evidence_id in binding.evidence_refs
    }
    framework_names = ", ".join(
        framework_label(framework, language)
        for framework in confirmed_frameworks
    ) or t("report.no_framework", language)
    st.markdown(
        '<div class="ui-report-context">'
        f'<span>{count_text("report.findings", len(report_findings), language)}</span>'
        f'<span>{count_text("report.evidence", len(authorized_evidence_ids), language)}</span>'
        f'<span>{count_text("report.gaps", len(scoped_missing), language)}</span>'
        f'<span>{framework_names}</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    primary_findings, _ = ordered_findings_for_presentation(report_findings)
    primary_findings = prioritize_selected_finding(
        primary_findings,
        st.session_state.get("selected_finding_id"),
    )
    primary_finding_ids = {
        finding.finding_id for finding in primary_findings
    }
    primary_rule_ids = {finding.rule_id for finding in primary_findings}
    primary_missing_information = [
        item
        for item in scoped_missing
        if item.rule_id in primary_rule_ids
    ]
    render_section_header(t("report.findings.title", language))
    if not report_findings:
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

    secondary_findings = [
        finding
        for finding in report_findings
        if finding.finding_id not in primary_finding_ids
    ]
    secondary_missing_information = [
        item
        for item in scoped_missing
        if item.rule_id not in primary_rule_ids
    ]
    confirmed_rule_ids = frozenset(route.confirmed_modules)
    secondary_failures = [
        failure
        for failure in report.execution_failures
        if failure.rule_id in confirmed_rule_ids
        and failure.rule_id not in primary_rule_ids
    ]
    other_frameworks = [
        framework
        for framework in confirmed_frameworks
        if any(
            item.framework is framework
            for item in (
                *secondary_findings,
                *secondary_missing_information,
                *secondary_failures,
            )
        )
    ]
    if other_frameworks:
        with st.expander(t("framework_screens.title", language), expanded=False):
            st.caption(t("framework_screens.copy", language))
            for framework in other_frameworks:
                framework_findings = [
                    finding
                    for finding in secondary_findings
                    if finding.framework is framework
                ]
                framework_missing = [
                    item
                    for item in secondary_missing_information
                    if item.framework is framework
                ]
                framework_failures = [
                    failure
                    for failure in secondary_failures
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
        if scoped_missing:
            st.caption(t("technical.raw_missing", language))
            for item in scoped_missing:
                st.code(item.fact_path, language=None)
        if report.recommendations:
            st.caption(t("technical.raw_recommendations", language))
            for recommendation in report.recommendations:
                st.code(recommendation, language=None)


def _fact_input_source(facts, fact_path: str, language: str) -> str:
    """Describe where a displayed fact entered the current case record."""

    normalization = st.session_state.get("assessment_input_normalization") or {}
    normalized_paths = set((normalization.get("fact_updates") or {}).keys())
    if (
        normalization.get("status") == NormalizationStatus.MATCHED.value
        and (
            fact_path in normalized_paths
            or (
                fact_path == "use_context.task"
                and normalization.get("canonical_task")
            )
        )
    ):
        return t("trace.fact_source.normalization", language)
    metadata = facts.fact_metadata.get(fact_path)
    if (
        st.session_state.get("demo_loaded")
        and metadata is not None
        and metadata.recorded_at is not None
    ):
        return t("trace.fact_source.demo_fixture", language)
    record = next(
        (
            item
            for item in _provenance()
            if item.fact_path == fact_path
        ),
        None,
    )
    if record is not None and record.module_id:
        return t("trace.fact_source.dynamic_questionnaire", language)
    if record is not None and record.explicitly_confirmed:
        return t("trace.fact_source.user_confirmed", language)
    if metadata is not None and metadata.question_id:
        return t("trace.fact_source.dynamic_questionnaire", language)
    if st.session_state.get("demo_loaded"):
        return t("trace.fact_source.demo_fixture", language)
    return t("trace.fact_source.case_record", language)


def _canonical_fact_value(facts, fact_path: str) -> str:
    value = resolve_fact(facts, fact_path)
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, list):
        return ", ".join(
            str(item.value if isinstance(item, Enum) else item)
            for item in value
        )
    return "null" if value is None else str(value)


def _rule_trace_summary(finding, language: str) -> str:
    key = f"trace.rule.explanation.{finding.rule_id}.{finding.status.value}"
    return t_or(key, finding_summary(finding, language), language)


def _render_condition_fact_mapping(finding, facts, language: str) -> None:
    """Render detailed predicates only through progressive disclosure."""

    with st.expander(t("trace.rule.mapping", language), expanded=False):
        if not finding.trace:
            st.caption(t("trace.rule.none", language))
            return
        for entry in finding.trace:
            fact_path = entry.fact_refs[0] if entry.fact_refs else None
            state_label, state_tone = reasoning_state(entry.result, language)
            fact_name = (
                fact_label(fact_path, language)
                if fact_path
                else t("value.not_recorded", language)
            )
            value = (
                fact_value(facts, fact_path, language)
                if fact_path
                else readable_result(entry.result, language)
            )
            st.markdown(
                '<div class="ui-condition-map-row">'
                '<div class="ui-condition-map-condition">'
                f"{escape(trace_description(entry, language))}</div>"
                '<div class="ui-condition-map-fact">'
                f"<strong>{escape(fact_name)}</strong> · {escape(value)}</div>"
                f'<span class="ui-state ui-state--{state_tone}">'
                f"{escape(state_label)}</span></div>",
                unsafe_allow_html=True,
            )


def _render_trace_technical_details(finding, facts, language: str) -> None:
    """Keep raw rule and provenance identities outside the primary trace."""

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
        provenance_by_path = {
            item.fact_path: item for item in _provenance()
        }
        for fact_path in finding.fact_refs:
            st.caption(fact_label(fact_path, language))
            st.code(fact_path, language=None)
            st.code(_canonical_fact_value(facts, fact_path), language=None)
            provenance = provenance_by_path.get(fact_path)
            metadata = facts.fact_metadata.get(fact_path)
            question_ids = []
            if provenance is not None:
                question_ids.append(provenance.question_id)
            if (
                metadata is not None
                and metadata.question_id
                and metadata.question_id not in question_ids
            ):
                question_ids.append(metadata.question_id)
            for question_id in question_ids:
                st.code(question_id, language=None)
            if provenance is not None and provenance.module_id:
                st.code(provenance.module_id, language=None)


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
                    '<span class="ui-trace-fact-value">'
                    f"{escape(fact_value(facts, fact_path, language))}</span>"
                    '<small class="ui-trace-fact-source">'
                    f"{escape(_fact_input_source(facts, fact_path, language))}</small>"
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
        rule_name = t_or(
            f"trace.rule.name.{finding.rule_id}",
            rule_label(finding.rule_id, language),
            language,
        )
        st.markdown(f"### {rule_name}")
        if finding.trace:
            matched_count = sum(
                reasoning_state(entry.result, language)[1] == "matched"
                for entry in finding.trace
            )
            st.markdown(
                '<div class="ui-rule-application-summary">'
                f'<strong>{escape(t("trace.rule.conditions", language, matched=matched_count, total=len(finding.trace)))}</strong>'
                '<span class="ui-rule-application-result">'
                f'{escape(t("trace.rule.overall_result", language))}: '
                f'{escape(status_label(finding.status, language))}</span>'
                f'<p>{escape(_rule_trace_summary(finding, language))}</p>'
                "</div>",
                unsafe_allow_html=True,
            )
            _render_condition_fact_mapping(finding, facts, language)
        else:
            st.caption(t("trace.rule.none", language))
        _render_trace_technical_details(finding, facts, language)

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
        st.info(
            t(
                "trace.evidence.pending_binding"
                if finding.legal_basis
                else "trace.evidence.none",
                language,
            )
        )
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
            st.caption(t("evidence.excerpt_hash", language))
            st.code(normalized_excerpt_hash(evidence.excerpt), language=None)
            st.caption(t("evidence.raw_source", language))
            st.code(evidence.legal_source, language=None)
            st.caption(t("evidence.source_version", language))
            st.code(
                evidence.document_version
                or t("evidence.not_recorded", language),
                language=None,
            )


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
    run_route = st.session_state.get("assessment_run_route")
    if not isinstance(run_route, QuestionnaireRoute):
        run_route = current_questionnaire_route(facts)
    incomplete_result = is_incomplete_assessment_result(report, run_route)
    if view == VIEW_RESULTS:
        if incomplete_result:
            render_incomplete_assessment(report, facts, run_route, language)
        else:
            render_report(report, facts, run_route, language)
        return

    if view == VIEW_EVIDENCE:
        if incomplete_result:
            render_incomplete_assessment(report, facts, run_route, language)
        else:
            render_evidence_workspace(report, facts, language)
        return

    st.session_state.assessment_view = VIEW_LANDING
    st.rerun()


if __name__ == "__main__":
    main()
