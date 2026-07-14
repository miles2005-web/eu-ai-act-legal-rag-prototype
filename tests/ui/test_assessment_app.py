"""Streamlit smoke tests for the enterprise assessment workspace."""

from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from src.assessment.demo import create_assessment_workflow
from src.assessment.facts import UseDomain
from src.assessment.findings import FindingStatus
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import TriState
from src.ui.styles import ENTERPRISE_STYLES


APP_PATH = Path(__file__).resolve().parents[2] / "assessment_app.py"


class AssessmentAppSmokeTests(unittest.TestCase):
    @staticmethod
    def _progress_state(app: AppTest, step_id: str) -> str:
        markup = "\n".join(item.value for item in app.markdown)
        marker = f'data-progress-step="{step_id}" data-progress-state="'
        start = markup.index(marker) + len(marker)
        return markup[start : markup.index('"', start)]

    @staticmethod
    def _open_recruitment_report(app: AppTest) -> AppTest:
        next(
            button
            for button in app.button
            if button.label == "Open recruitment demo"
        ).click()
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "Run assessment"
        ).click()
        app.run(timeout=10)
        return app

    def test_english_is_default_and_language_options_are_accessible(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

        self.assertEqual(app.session_state["ui_language"], "en")
        self.assertEqual(len(app.radio), 1)
        self.assertEqual(app.radio[0].options, ["EN", "中文"])
        self.assertIn("Assessment", [button.label for button in app.button])

    def test_chinese_landing_and_assessment_form_are_localized(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)

        labels = [button.label for button in app.button]
        self.assertIn("评估", labels)
        self.assertIn("演示案例", labels)
        self.assertIn("打开招聘 AI 演示", labels)
        landing_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("欧盟数字监管", landing_markup)

        next(
            button for button in app.button if button.label == "打开招聘 AI 演示"
        ).click()
        app.run(timeout=10)

        self.assertIn("运行评估", [button.label for button in app.button])
        self.assertIn(
            "该系统用于什么场景？",
            [select.label for select in app.selectbox],
        )
        self.assertIn(
            "该系统执行什么任务？",
            [area.label for area in app.text_area],
        )

    def test_sidebar_uses_lightweight_workspace_navigation(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

        labels = [button.label for button in app.button]
        self.assertIn("Assessment", labels)
        self.assertIn("Demo cases", labels)
        self.assertIn("Findings", labels)
        self.assertIn("Evidence trace", labels)
        self.assertNotIn("Evidence engine", labels)
        self.assertIn(
            '[data-testid="stSidebar"] .stButton > button',
            ENTERPRISE_STYLES,
        )
        self.assertIn("background: transparent", ENTERPRISE_STYLES)
        self.assertIn("border: 0", ENTERPRISE_STYLES)

    def test_landing_renders_both_demo_entry_points(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        button_labels = [button.label for button in app.button]
        self.assertIn("Open recruitment demo", button_labels)
        self.assertIn("Open industrial demo", button_labels)

    def test_industrial_demo_opens_assessment_workspace(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        next(
            button
            for button in app.button
            if button.label == "Open industrial demo"
        ).click()

        app.run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["assessment_view"], "workspace")
        self.assertEqual(app.session_state["demo_scenario"], "industrial")
        self.assertIn("Run assessment", [button.label for button in app.button])
        workspace_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("ui-system-summary", workspace_markup)
        self.assertIn("ForgeMonitor AI", workspace_markup)
        self.assertIn("Technical details", [item.label for item in app.expander])

    def test_predefined_demo_content_localizes_without_mutating_facts(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        next(
            button
            for button in app.button
            if button.label == "Open recruitment demo"
        ).click()
        app.run(timeout=10)
        case_id = app.session_state["assessment_case_id"]
        bundle = app.session_state["assessment_workflow_bundle"]
        assessment_case = bundle.case_service.get_case(case_id)
        original_case = assessment_case.to_dict()

        english_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn(
            "Recruitment AI candidate screening and ranking",
            english_markup,
        )
        self.assertIn(
            "Support recruitment teams by screening and ranking job candidates",
            english_markup,
        )

        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)

        chinese_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("招聘 AI 候选人筛选与排序", chinese_markup)
        self.assertIn("一家虚构的欧盟公司使用第三方 AI 系统", chinese_markup)
        self.assertIn("为招聘团队提供支持", chinese_markup)
        self.assertIn(
            "招聘筛选与候选人排序，用于面试遴选",
            [area.value for area in app.text_area],
        )
        self.assertEqual(
            bundle.case_service.get_case(case_id).to_dict(),
            original_case,
        )

    def test_user_created_chinese_case_text_is_not_translated(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)
        case_name = "上海智能招聘合规评估"
        description = "用户自行输入的中文案例说明"
        app.text_input[0].set_value(case_name)
        app.text_area[0].set_value(description)
        next(
            button for button in app.button if button.label == "创建案例"
        ).click()
        app.run(timeout=10)

        bundle = app.session_state["assessment_workflow_bundle"]
        assessment_case = bundle.case_service.get_case(
            app.session_state["assessment_case_id"]
        )
        self.assertEqual(assessment_case.name, case_name)
        self.assertEqual(assessment_case.description, description)
        markup = "\n".join(item.value for item in app.markdown)
        self.assertIn(case_name, markup)
        self.assertIn(description, markup)

    def test_chinese_controls_store_canonical_enums_and_normalized_task(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "打开招聘 AI 演示"
        ).click()
        app.run(timeout=10)
        app.selectbox[0].set_value(UseDomain.EMPLOYMENT)
        app.selectbox[1].set_value(TriState.YES)
        original_text = (
            "招聘筛选 候选人排序 处理个人数据 "
            "完全自动化决定"
        )
        app.text_area[0].set_value(original_text)
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "保存事实"
        ).click()
        app.run(timeout=10)

        bundle = app.session_state["assessment_workflow_bundle"]
        facts = bundle.case_service.get_case(
            app.session_state["assessment_case_id"]
        ).current_facts
        self.assertIs(facts.use_context.domain, UseDomain.EMPLOYMENT)
        self.assertIs(
            facts.use_context.materially_influences_decision,
            TriState.YES,
        )
        self.assertEqual(
            facts.use_context.task,
            "recruitment screening of candidates; candidate ranking",
        )
        self.assertIs(
            facts.data_protection.personal_data_processed,
            TriState.YES,
        )
        self.assertIs(
            facts.data_protection.automated_individual_decision,
            TriState.YES,
        )
        metadata = app.session_state["assessment_input_normalization"]
        self.assertEqual(metadata["original_text"], original_text)
        self.assertEqual(metadata["status"], "matched")

        next(
            button for button in app.button if button.label == "确认模块"
        ).click()
        app.run(timeout=10)

        next(
            button for button in app.button if button.label == "运行评估"
        ).click()
        app.run(timeout=10)
        self.assertEqual(
            [finding.rule_id for finding in app.session_state["assessment_report"].findings],
            ["AI_ACT_HIGH_RISK_EMPLOYMENT", "GDPR_ARTICLE22_RELEVANCE"],
        )

    def test_ambiguous_chinese_task_requires_confirmation_and_stays_unknown(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        next(
            button
            for button in app.button
            if button.label == "Open recruitment demo"
        ).click()
        app.run(timeout=10)
        original_text = "用来帮助人事团队的智能工具"
        app.text_area[0].set_value(original_text)
        app.run(timeout=10)

        self.assertIn(
            "Confirm the original text as the task description without legal classification",
            [checkbox.label for checkbox in app.checkbox],
        )
        next(
            button for button in app.button if button.label == "Save facts"
        ).click()
        app.run(timeout=10)

        bundle = app.session_state["assessment_workflow_bundle"]
        facts = bundle.case_service.get_case(
            app.session_state["assessment_case_id"]
        ).current_facts
        self.assertIsNone(facts.use_context.task)
        metadata = app.session_state["assessment_input_normalization"]
        self.assertEqual(metadata["original_text"], original_text)
        self.assertEqual(metadata["status"], "ambiguous")
        self.assertFalse(metadata["ambiguous_text_confirmed"])

    def test_custom_loan_routes_gdpr_and_exposes_unsupported_ai_act_path(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("Automated loan decision")
        app.text_area[0].set_value("Consumer lending relevance assessment")
        next(button for button in app.button if button.label == "Create case").click()
        app.run(timeout=10)

        app.selectbox[0].set_value(UseDomain.ESSENTIAL_SERVICES)
        app.text_area[0].set_value(
            "personal financial and credit analysis; automated loan "
            "approval/rejection; legal or significant economic effect"
        )
        next(button for button in app.button if button.label == "Save facts").click()
        app.run(timeout=10)

        route = app.session_state["assessment_questionnaire_route"]
        self.assertEqual(route.suggested_modules, ["GDPR_ARTICLE22_RELEVANCE"])
        self.assertEqual(
            [item.path_id for item in route.unsupported_modules],
            ["AI_ACT_ESSENTIAL_SERVICES_CREDIT_UNSUPPORTED"],
        )
        self.assertNotIn(
            "employment",
            " ".join(select.label.casefold() for select in app.selectbox),
        )
        warnings = "\n".join(item.value for item in app.warning)
        self.assertIn("AI Act credit and essential-services assessment", warnings)
        self.assertIn("not yet implemented", warnings)
        self.assertTrue(
            next(
                button for button in app.button if button.label == "Run assessment"
            ).disabled
        )
        self.assertEqual(self._progress_state(app, "facts"), "pending")

        next(
            button for button in app.button if button.label == "Confirm module"
        ).click()
        app.run(timeout=10)
        self.assertEqual(self._progress_state(app, "facts"), "pending")
        self.assertIn(
            "Is an automated decision about an individual involved?",
            [select.label for select in app.selectbox],
        )
        follow_up = next(
            select
            for select in app.selectbox
            if select.label == "Is an automated decision about an individual involved?"
        )
        follow_up.set_value("yes")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)
        self.assertEqual(self._progress_state(app, "facts"), "complete")
        next(button for button in app.button if button.label == "Run assessment").click()
        app.run(timeout=10)

        report = app.session_state["assessment_report"]
        self.assertEqual(
            [finding.rule_id for finding in report.findings],
            ["GDPR_ARTICLE22_RELEVANCE"],
        )
        self.assertEqual(
            report.findings[0].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(self._progress_state(app, "facts"), "complete")
        self.assertEqual(self._progress_state(app, "assessment"), "complete")
        self.assertTrue(
            any(
                evidence.citation == "Article 22(1)"
                for evidence in report.evidence
            )
        )
        next(
            button
            for button in app.button
            if button.label == "View full evidence trace"
        ).click()
        app.run(timeout=10)
        self.assertEqual(app.session_state["assessment_view"], "evidence")
        self.assertEqual(self._progress_state(app, "facts"), "complete")
        self.assertEqual(self._progress_state(app, "assessment"), "complete")
        trace_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("GDPR Article 22 relevance test", trace_markup)
        self.assertIn("Conditions satisfied: 3 of 3", trace_markup)
        self.assertIn("Deterministic normalization", trace_markup)
        self.assertIn("Dynamic questionnaire", trace_markup)
        self.assertIn("ui-condition-map-row", trace_markup)
        mapping_expander = next(
            item
            for item in app.expander
            if item.label == "View condition-to-fact mapping"
        )
        self.assertFalse(mapping_expander.proto.expanded)
        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)
        self.assertEqual(self._progress_state(app, "facts"), "complete")
        self.assertEqual(self._progress_state(app, "assessment"), "complete")

    def test_evidence_chain_does_not_reuse_primary_decision_path_renderer(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        chain_source = source.split("def render_compliance_chain(", 1)[1].split(
            "def render_evidence_workspace(", 1
        )[0]

        self.assertNotIn("render_decision_path(", chain_source)
        self.assertIn("_render_condition_fact_mapping(", chain_source)
        self.assertIn("ui-rule-application-summary", chain_source)

    def test_incomplete_confirmed_module_distinguishes_facts_from_run_completion(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("Incomplete GDPR case")
        app.text_area[0].set_value("Progress-state regression")
        next(button for button in app.button if button.label == "Create case").click()
        app.run(timeout=10)

        manual_modules = next(
            item
            for item in app.multiselect
            if item.label == "Modules available for screening"
        )
        manual_modules.set_value(["GDPR_ARTICLE22_RELEVANCE"])
        next(
            button
            for button in app.button
            if button.label == "Confirm selected modules"
        ).click()
        app.run(timeout=10)

        route = app.session_state["assessment_questionnaire_route"]
        self.assertIn(
            "data_protection.personal_data_processed",
            route.missing_fact_paths["GDPR_ARTICLE22_RELEVANCE"],
        )
        self.assertEqual(self._progress_state(app, "facts"), "pending")
        next(button for button in app.button if button.label == "Run assessment").click()
        app.run(timeout=10)

        self.assertIsNotNone(app.session_state["assessment_report"])
        self.assertEqual(self._progress_state(app, "facts"), "pending")
        self.assertEqual(self._progress_state(app, "assessment"), "complete")

    def test_domain_change_invalidates_employment_answers_and_stale_report(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_recruitment_report(app)
        original_bundle = app.session_state["assessment_workflow_bundle"]
        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)
        next(button for button in app.button if button.label == "评估").click()
        app.run(timeout=10)

        app.selectbox[0].set_value(UseDomain.ESSENTIAL_SERVICES)
        next(button for button in app.button if button.label == "保存事实").click()
        app.run(timeout=10)

        bundle = app.session_state["assessment_workflow_bundle"]
        facts = bundle.case_service.get_case(
            app.session_state["assessment_case_id"]
        ).current_facts
        self.assertIs(facts.use_context.domain, UseDomain.ESSENTIAL_SERVICES)
        self.assertIsNone(facts.use_context.task)
        self.assertIs(
            facts.use_context.materially_influences_decision,
            TriState.UNKNOWN,
        )
        self.assertNotIn(
            "AI_ACT_HIGH_RISK_EMPLOYMENT",
            app.session_state["assessment_confirmed_modules"],
        )
        self.assertIsNone(app.session_state["assessment_report"])
        self.assertIsNone(app.session_state["selected_finding_id"])
        self.assertEqual(app.session_state["ui_language"], "zh-CN")
        self.assertIsNot(bundle, original_bundle)

    def test_unsupported_judicial_case_does_not_activate_formal_module(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("Judicial support system")
        app.text_area[0].set_value("Preliminary route screening")
        next(button for button in app.button if button.label == "Create case").click()
        app.run(timeout=10)

        app.selectbox[0].set_value(UseDomain.JUSTICE_DEMOCRATIC_PROCESSES)
        app.text_area[0].set_value("Assist judicial decision preparation")
        next(button for button in app.button if button.label == "Save facts").click()
        app.run(timeout=10)

        route = app.session_state["assessment_questionnaire_route"]
        self.assertEqual(route.suggested_modules, [])
        self.assertEqual(
            [item.path_id for item in route.unsupported_modules],
            ["AI_ACT_JUDICIAL_ROUTE_UNSUPPORTED"],
        )
        self.assertEqual(app.session_state["assessment_confirmed_modules"], [])
        self.assertNotIn(
            "employment",
            " ".join(select.label.casefold() for select in app.selectbox),
        )
        run_button = next(
            button for button in app.button if button.label == "Run assessment"
        )
        self.assertTrue(run_button.disabled)

    def test_connected_product_change_invalidates_only_data_act_route_state(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        next(
            button for button in app.button if button.label == "Open industrial demo"
        ).click()
        app.run(timeout=10)
        next(button for button in app.button if button.label == "Run assessment").click()
        app.run(timeout=10)
        self.assertIsNotNone(app.session_state["assessment_report"])
        next(button for button in app.button if button.label == "Assessment").click()
        app.run(timeout=10)

        connected = next(
            select
            for select in app.selectbox
            if select.label == "Is a connected product involved?"
        )
        connected.set_value(TriState.NO)
        next(button for button in app.button if button.label == "Save facts").click()
        app.run(timeout=10)

        bundle = app.session_state["assessment_workflow_bundle"]
        facts = bundle.case_service.get_case(
            app.session_state["assessment_case_id"]
        ).current_facts
        self.assertIs(facts.data_act.connected_product, TriState.NO)
        self.assertIs(facts.data_act.related_service, TriState.UNKNOWN)
        self.assertIs(facts.data_act.data_generated, TriState.UNKNOWN)
        self.assertNotIn(
            "EU_DATA_ACT_RELEVANCE",
            app.session_state["assessment_confirmed_modules"],
        )
        self.assertIsNone(app.session_state["assessment_report"])

    def test_recruitment_flow_opens_results_and_evidence_trace(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_recruitment_report(app)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["assessment_view"], "results")
        report = app.session_state["assessment_report"]
        self.assertEqual(report.findings[0].status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertEqual(report.findings[0].rule_id, "AI_ACT_HIGH_RISK_EMPLOYMENT")
        self.assertNotIn(
            "EU_DATA_ACT",
            {evidence.legal_source for evidence in report.evidence},
        )
        result_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("ui-finding-title--hero", result_markup)
        self.assertIn("Decision path", result_markup)
        self.assertIn("Employment context", result_markup)
        self.assertIn("Material influence on decisions", result_markup)
        self.assertIn("Evidence summary", result_markup)
        self.assertIn("Technical details", [item.label for item in app.expander])
        next(
            button
            for button in app.button
            if button.label == "View full evidence trace"
        ).click()

        app.run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["assessment_view"], "evidence")
        trace_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("Facts", trace_markup)
        self.assertIn("Rule evaluation", trace_markup)
        self.assertIn("Legal basis", trace_markup)
        self.assertIn("Source evidence", trace_markup)
        self.assertIn("Employment context", trace_markup)
        self.assertIn("Material influence on decisions", trace_markup)
        self.assertIn("supporting excerpts", trace_markup)
        code_values = [code.value for code in app.code]
        expected_evidence_ids = {
            evidence.evidence_id for evidence in report.evidence
        }
        self.assertTrue(expected_evidence_ids.issubset(set(code_values)))

    def test_language_switch_preserves_case_report_and_legal_evidence(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_recruitment_report(app)

        original_case_id = app.session_state["assessment_case_id"]
        original_bundle = app.session_state["assessment_workflow_bundle"]
        original_report = app.session_state["assessment_report"]
        self.assertEqual(len(original_report.evidence), 7)
        original_report_id = original_report.report_id
        original_finding = original_report.findings[0]
        original_facts = original_bundle.case_service.get_case(
            original_case_id
        ).current_facts.to_dict()
        app.session_state["selected_finding_id"] = original_finding.finding_id
        original_citations = [
            basis.citation for basis in original_finding.legal_basis
        ]
        original_excerpts = [evidence.excerpt for evidence in original_report.evidence]
        original_evidence_ids = [
            evidence.evidence_id for evidence in original_report.evidence
        ]
        original_modules = list(
            app.session_state["assessment_confirmed_modules"]
        )
        original_provenance = [
            item.to_dict()
            for item in app.session_state["assessment_fact_provenance"]
        ]
        original_route = app.session_state["assessment_questionnaire_route"].to_dict()

        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)

        self.assertEqual(app.session_state["ui_language"], "zh-CN")
        self.assertEqual(app.session_state["assessment_view"], "results")
        self.assertEqual(app.session_state["assessment_case_id"], original_case_id)
        self.assertIs(
            app.session_state["assessment_workflow_bundle"],
            original_bundle,
        )
        self.assertEqual(
            app.session_state["assessment_workflow_bundle"]
            .case_service.get_case(original_case_id)
            .current_facts.to_dict(),
            original_facts,
        )
        self.assertEqual(
            app.session_state["selected_finding_id"],
            original_finding.finding_id,
        )
        self.assertEqual(
            app.session_state["assessment_confirmed_modules"],
            original_modules,
        )
        self.assertEqual(
            [
                item.to_dict()
                for item in app.session_state["assessment_fact_provenance"]
            ],
            original_provenance,
        )
        self.assertEqual(
            app.session_state["assessment_questionnaire_route"].to_dict(),
            original_route,
        )
        translated_report = app.session_state["assessment_report"]
        self.assertEqual(translated_report.report_id, original_report_id)
        self.assertEqual(
            translated_report.findings[0].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(
            [basis.citation for basis in translated_report.findings[0].legal_basis],
            original_citations,
        )
        self.assertEqual(
            [evidence.excerpt for evidence in translated_report.evidence],
            original_excerpts,
        )
        self.assertEqual(
            [evidence.evidence_id for evidence in translated_report.evidence],
            original_evidence_ids,
        )

        result_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("可能属于就业相关高风险 AI 系统", result_markup)
        self.assertIn("判断路径", result_markup)
        self.assertIn("《欧盟人工智能法案》", result_markup)
        self.assertTrue(any("法律审查" in item.value for item in app.markdown))
        self.assertNotIn(
            "Obtain legal review",
            "\n".join(item.value for item in app.markdown),
        )
        self.assertIn("查看完整证据链", [button.label for button in app.button])

        next(
            button for button in app.button if button.label == "查看完整证据链"
        ).click()
        app.run(timeout=10)

        self.assertEqual(app.session_state["assessment_view"], "evidence")
        trace_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("事实", trace_markup)
        self.assertIn("规则判断", trace_markup)
        self.assertIn("法律依据", trace_markup)
        self.assertIn("来源证据", trace_markup)
        self.assertIn("官方原文（英语）", trace_markup)
        self.assertIn("Article 6", trace_markup)
        self.assertIn("Annex III point 4(a)", trace_markup)
        self.assertEqual(
            [evidence.excerpt for evidence in app.session_state["assessment_report"].evidence],
            original_excerpts,
        )
        self.assertEqual(
            [evidence.evidence_id for evidence in app.session_state["assessment_report"].evidence],
            original_evidence_ids,
        )
        self.assertTrue(set(original_evidence_ids).issubset({code.value for code in app.code}))

    def test_report_scopes_missing_information_and_humanizes_recommendations(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_recruitment_report(app)
        report = app.session_state["assessment_report"]
        report_snapshot = report.to_dict()

        english_markup = "\n".join(item.value for item in app.markdown)
        self.assertNotIn("ui-primary-missing", english_markup)
        self.assertIn("Personal data processing", english_markup)
        self.assertIn("Connected product", english_markup)
        self.assertIn("Other framework screens", [item.label for item in app.expander])
        self.assertNotIn(
            "AIA_HIGH_RISK_EMPLOYMENT_PRELIMINARY",
            "\n".join(
                item.value
                for item in app.markdown
                if "recommendation" in item.value
            ),
        )

        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)

        chinese_markup = "\n".join(item.value for item in app.markdown)
        primary_markup = "\n".join(
            item.value
            for item in app.markdown
            if "ui-primary-" in item.value
        )
        framework_markup = "\n".join(
            item.value
            for item in app.markdown
            if "ui-framework-" in item.value
        )
        self.assertNotIn("ui-primary-missing", primary_markup)
        self.assertIn("就业领域高风险初步分类", primary_markup)
        self.assertNotIn("data_act.", primary_markup)
        self.assertNotIn("data_protection.", primary_markup)
        self.assertNotIn("AIA_HIGH_RISK_EMPLOYMENT_PRELIMINARY", primary_markup)
        self.assertIn("个人数据处理", framework_markup)
        self.assertIn("自动化个人决策", framework_markup)
        self.assertIn("互联产品", framework_markup)
        self.assertIn("相关服务", framework_markup)
        self.assertIn("产品或相关服务生成数据", framework_markup)
        self.assertEqual(
            framework_markup.count('class="ui-framework-missing"'),
            len(report.missing_information),
        )
        self.assertIn("其他框架筛查", [item.label for item in app.expander])
        raw_code = {item.value for item in app.code}
        self.assertTrue(
            {item.fact_path for item in report.missing_information}.issubset(raw_code)
        )
        self.assertIn("AIA_HIGH_RISK_EMPLOYMENT_PRELIMINARY", raw_code)
        self.assertNotIn("data_act.connected_product", chinese_markup)
        self.assertEqual(app.session_state["assessment_report"].to_dict(), report_snapshot)

    def test_demo_switch_isolates_state_and_industrial_primary_finding(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_recruitment_report(app)
        recruitment_case_id = app.session_state["assessment_case_id"]
        recruitment_bundle = app.session_state["assessment_workflow_bundle"]
        recruitment_report = app.session_state["assessment_report"]
        app.session_state["selected_finding_id"] = (
            recruitment_report.findings[0].finding_id
        )

        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "演示案例"
        ).click()
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "打开工业 AI 演示"
        ).click()
        app.run(timeout=10)

        industrial_case_id = app.session_state["assessment_case_id"]
        industrial_bundle = app.session_state["assessment_workflow_bundle"]
        self.assertEqual(app.session_state["ui_language"], "zh-CN")
        self.assertNotEqual(industrial_case_id, recruitment_case_id)
        self.assertIsNot(industrial_bundle, recruitment_bundle)
        self.assertIsNone(app.session_state["assessment_report"])
        self.assertIsNone(app.session_state["selected_finding_id"])
        self.assertEqual(app.session_state["demo_scenario"], "industrial")
        industrial_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("工业 AI 互联机械监测", industrial_markup)
        self.assertIn("一家欧洲制造商运行一套连接生产机械的 AI 工业监测系统", industrial_markup)
        self.assertIn("利用设备运行数据监测互联生产设备", industrial_markup)
        self.assertEqual(
            app.session_state["assessment_fact_task"],
            "通过相关服务监测联网机械并处理产品运行数据",
        )
        self.assertEqual(
            industrial_bundle.case_service.get_case(
                industrial_case_id
            ).current_facts.use_context.task,
            (
                "Industrial equipment monitoring, anomaly detection, and "
                "preventive maintenance recommendation"
            ),
        )

        next(
            button for button in app.button if button.label == "运行评估"
        ).click()
        app.run(timeout=10)

        report = app.session_state["assessment_report"]
        run = industrial_bundle.workflow.get_run(
            report.assessment_run_reference
        )
        self.assertEqual(run.case_id, industrial_case_id)
        self.assertNotEqual(report.report_id, recruitment_report.report_id)
        findings_by_rule = {
            finding.rule_id: finding for finding in report.findings
        }
        self.assertEqual(
            set(findings_by_rule),
            {"EU_DATA_ACT_RELEVANCE"},
        )
        data_act_finding = findings_by_rule["EU_DATA_ACT_RELEVANCE"]
        self.assertEqual(
            data_act_finding.framework,
            RegulatoryFramework.EU_DATA_ACT,
        )
        self.assertEqual(
            data_act_finding.status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(
            data_act_finding.reason_codes,
            ["CONNECTED_PRODUCT", "RELATED_SERVICE", "DATA_GENERATED"],
        )
        evidence_by_id = {
            evidence.evidence_id: evidence for evidence in report.evidence
        }
        findings_by_id = {
            finding.finding_id: finding for finding in report.findings
        }
        for binding in report.evidence_bindings:
            finding = findings_by_id[binding.finding_id]
            expected_sources = {
                basis.instrument for basis in finding.legal_basis
            }
            actual_sources = {
                evidence_by_id[evidence_id].legal_source
                for evidence_id in binding.evidence_refs
            }
            self.assertTrue(actual_sources.issubset(expected_sources))

        data_binding = next(
            binding
            for binding in report.evidence_bindings
            if binding.finding_id == data_act_finding.finding_id
        )
        self.assertGreater(len(data_binding.evidence_refs), 0)
        self.assertEqual(
            {
                evidence_by_id[evidence_id].legal_source
                for evidence_id in data_binding.evidence_refs
            },
            {"EU_DATA_ACT"},
        )
        result_markup = "\n".join(item.value for item in app.markdown)
        self.assertEqual(
            result_markup.count(
                'class="ui-finding-title ui-finding-title--hero"'
            ),
            1,
        )
        self.assertIn("可能涉及《欧盟数据法案》", result_markup)
        self.assertIn("其他框架筛查", [item.label for item in app.expander])
        self.assertNotIn("ui-primary-missing", result_markup)
        self.assertNotIn(
            "use_context.domain",
            "\n".join(
                item.value
                for item in app.markdown
                if "ui-primary-" in item.value
            ),
        )

    def test_report_from_another_bundle_is_rejected_before_rendering(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_recruitment_report(app)
        stale_report = app.session_state["assessment_report"]

        replacement_bundle = create_assessment_workflow()
        replacement_case = replacement_bundle.case_service.create_case(
            "Replacement case"
        )
        app.session_state["assessment_workflow_bundle"] = replacement_bundle
        app.session_state["assessment_case_id"] = replacement_case.case_id
        app.session_state["assessment_report"] = stale_report
        app.session_state["selected_finding_id"] = (
            stale_report.findings[0].finding_id
        )
        app.session_state["assessment_view"] = "results"

        app.run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state["assessment_report"])
        self.assertIsNone(app.session_state["selected_finding_id"])
        self.assertEqual(app.session_state["assessment_view"], "workspace")
        markup = "\n".join(item.value for item in app.markdown)
        self.assertNotIn(
            "Employment-related high-risk classification potentially applies",
            markup,
        )


if __name__ == "__main__":
    unittest.main()
