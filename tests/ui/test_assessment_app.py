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
from src.assessment.questionnaire import QuestionResponseState
from src.assessment.questionnaire.definitions import (
    AI_ACT_PRODUCT_SAFETY_RULE_ID,
    EU_DATA_ACT_RULE_ID,
)
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

    @staticmethod
    def _open_industrial_multiframework_report(app: AppTest) -> AppTest:
        next(
            button
            for button in app.button
            if button.label == "Open multi-framework demo"
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
        self.assertIn("Open multi-framework demo", button_labels)

    def test_multiframework_demo_shows_two_independent_framework_findings(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_industrial_multiframework_report(app)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(
            app.session_state["assessment_confirmed_modules"],
            [AI_ACT_PRODUCT_SAFETY_RULE_ID, EU_DATA_ACT_RULE_ID],
        )
        report = app.session_state["assessment_report"]
        self.assertEqual(
            report.authorized_rule_ids,
            [AI_ACT_PRODUCT_SAFETY_RULE_ID, EU_DATA_ACT_RULE_ID],
        )
        self.assertEqual(
            [item.rule_id for item in report.findings],
            [AI_ACT_PRODUCT_SAFETY_RULE_ID, EU_DATA_ACT_RULE_ID],
        )
        self.assertEqual(len(report.evidence), 7)
        self.assertEqual(report.missing_information, [])
        self.assertEqual(report.execution_failures, [])
        self.assertIn(
            "Two independent regulatory screens produced substantive findings.",
            [item.value for item in app.info],
        )
        framework_summaries = "\n".join(
            item.value
            for item in app.markdown
            if '<div class="ui-framework-finding-summary"' in item.value
        )
        self.assertIn('data-framework="EU_AI_ACT"', framework_summaries)
        self.assertIn('data-framework="EU_DATA_ACT"', framework_summaries)
        self.assertEqual(
            framework_summaries.count(
                '<div class="ui-framework-finding-summary"'
            ),
            2,
        )
        recommendations = "\n".join(
            item.value
            for item in app.markdown
            if "ui-primary-recommendation" in item.value
        )
        self.assertIn("Annex I product legislation", recommendations)
        self.assertIn("data-holder relationships", recommendations)
        self.assertNotIn("GDPR", framework_summaries)
        self.assertNotIn("employment", framework_summaries.casefold())
        primary_markup = "\n".join(item.value for item in app.markdown)
        self.assertNotIn(
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
            primary_markup,
        )

    def test_multiframework_evidence_selection_keeps_sources_separate(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_industrial_multiframework_report(app)
        report = app.session_state["assessment_report"]
        findings = {item.rule_id: item for item in report.findings}

        trace_buttons = [
            button
            for button in app.button
            if button.label == "View full evidence trace"
        ]
        self.assertEqual(len(trace_buttons), 2)
        trace_buttons[1].click()
        app.run(timeout=10)

        self.assertEqual(
            app.session_state["selected_finding_id"],
            findings[EU_DATA_ACT_RULE_ID].finding_id,
        )
        data_markup = "\n".join(item.value for item in app.markdown)
        data_trace_markup = data_markup.split(
            '<section class="ui-section-header"><h2 class="ui-section-title">Findings</h2>'
        )[0]
        self.assertIn("Data Act relevance potentially applies", data_trace_markup)
        self.assertIn("Article 2(5)", data_trace_markup)
        self.assertNotIn("Article 3(14)", data_trace_markup)

        app.selectbox[0].set_value(
            findings[AI_ACT_PRODUCT_SAFETY_RULE_ID].finding_id
        )
        app.run(timeout=10)

        self.assertEqual(
            app.session_state["selected_finding_id"],
            findings[AI_ACT_PRODUCT_SAFETY_RULE_ID].finding_id,
        )
        ai_markup = "\n".join(item.value for item in app.markdown)
        ai_trace_markup = ai_markup.split(
            '<section class="ui-section-header"><h2 class="ui-section-title">Findings</h2>'
        )[0]
        self.assertIn("Article 3(14)", ai_trace_markup)
        self.assertIn("Article 6(1)(a)", ai_trace_markup)
        self.assertNotIn("Article 2(5)", ai_trace_markup)
        self.assertIn(
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC",
            {item.value for item in app.code},
        )

    def test_multiframework_language_and_scenario_switch_preserve_then_isolate_state(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_industrial_multiframework_report(app)
        report = app.session_state["assessment_report"]
        report_snapshot = report.to_dict()
        selected = report.findings[1].finding_id
        app.session_state["selected_finding_id"] = selected

        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)

        self.assertEqual(app.session_state["selected_finding_id"], selected)
        self.assertEqual(
            app.session_state["assessment_report"].to_dict(), report_snapshot
        )
        self.assertEqual(
            app.session_state["assessment_confirmed_modules"],
            [AI_ACT_PRODUCT_SAFETY_RULE_ID, EU_DATA_ACT_RULE_ID],
        )
        self.assertIn(
            "两项相互独立的监管筛查形成了实质性结论。",
            [item.value for item in app.info],
        )

        next(
            button for button in app.button if button.label == "演示案例"
        ).click()
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "打开工业 AI 演示"
        ).click()
        app.run(timeout=10)

        self.assertEqual(app.session_state["ui_language"], "zh-CN")
        self.assertEqual(app.session_state["demo_scenario"], "industrial")
        self.assertEqual(
            app.session_state["assessment_confirmed_modules"],
            [EU_DATA_ACT_RULE_ID],
        )
        self.assertIsNone(app.session_state["assessment_report"])
        self.assertIsNone(app.session_state["selected_finding_id"])
        self.assertIsNone(app.session_state["assessment_run_route"])

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

    def test_product_safety_module_uses_confirmed_bilingual_questionnaire(self) -> None:
        instrument_id = "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("Machinery safety controller")
        app.text_area[0].set_value("Article 6(1) product-safety screening")
        next(button for button in app.button if button.label == "Create case").click()
        app.run(timeout=10)

        app.selectbox[0].set_value(UseDomain.PRODUCT_SAFETY)
        next(button for button in app.button if button.label == "Save facts").click()
        app.run(timeout=10)

        route = app.session_state["assessment_questionnaire_route"]
        self.assertIn(AI_ACT_PRODUCT_SAFETY_RULE_ID, route.suggested_modules)
        self.assertNotIn(
            AI_ACT_PRODUCT_SAFETY_RULE_ID,
            app.session_state["assessment_confirmed_modules"],
        )
        self.assertTrue(
            next(
                button
                for button in app.button
                if button.label == "Run assessment"
            ).disabled
        )
        boundary_text = "\n".join(
            [item.value for item in app.markdown]
            + [item.value for item in app.caption]
        )
        self.assertIn("only the Article 6(1)", boundary_text)

        next(
            button for button in app.button if button.label == "Confirm module"
        ).click()
        app.run(timeout=10)
        next(
            select
            for select in app.selectbox
            if select.label.startswith("Is the AI system itself")
        ).set_value("no")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)

        next(
            select
            for select in app.selectbox
            if select.label.startswith("Is the AI system intended")
        ).set_value("yes")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)

        next(
            item
            for item in app.text_input
            if item.label == "What type of product is involved?"
        ).set_value("industrial machinery")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)

        instrument_select = next(
            select
            for select in app.selectbox
            if select.label.startswith("Which Annex I")
        )
        self.assertTrue(
            any("Directive 2006/42/EC" in option for option in instrument_select.options)
        )
        self.assertNotIn(instrument_id, instrument_select.options)

        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)
        self.assertIn(
            AI_ACT_PRODUCT_SAFETY_RULE_ID,
            app.session_state["assessment_confirmed_modules"],
        )
        chinese_instrument_select = next(
            select for select in app.selectbox if select.label.startswith("哪一项附件 I")
        )
        self.assertTrue(
            any("机械指令" in option for option in chinese_instrument_select.options)
        )
        self.assertNotIn(instrument_id, chinese_instrument_select.options)
        chinese_instrument_select.set_value(instrument_id)
        next(
            button for button in app.button if button.label == "保存补充回答"
        ).click()
        app.run(timeout=10)
        facts = app.session_state["assessment_workflow_bundle"].case_service.get_case(
            app.session_state["assessment_case_id"]
        ).current_facts
        self.assertEqual(facts.product_regulation.annex_i_instrument, instrument_id)

        next(
            select for select in app.selectbox if "所选附件 I 法规" in select.label
        ).set_value("yes")
        next(
            button for button in app.button if button.label == "保存补充回答"
        ).click()
        app.run(timeout=10)
        next(
            select for select in app.selectbox if "独立第三方" in select.label
        ).set_value("yes")
        next(
            button for button in app.button if button.label == "保存补充回答"
        ).click()
        app.run(timeout=10)

        route = app.session_state["assessment_questionnaire_route"]
        self.assertEqual(route.missing_fact_paths[AI_ACT_PRODUCT_SAFETY_RULE_ID], [])
        next(button for button in app.button if button.label == "运行评估").click()
        app.run(timeout=10)
        report = app.session_state["assessment_report"]
        finding = next(
            item
            for item in report.findings
            if item.rule_id == AI_ACT_PRODUCT_SAFETY_RULE_ID
        )
        self.assertIs(finding.status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertEqual(
            [item.citation for item in report.evidence],
            [
                "Article 3(14)",
                "Article 6(1)(a)",
                "Article 6(1)(b)",
                "Annex I, Section A, point 1",
            ],
        )
        self.assertEqual(len(report.evidence_bindings), 1)
        self.assertFalse(
            any("尚未为该规则绑定原子官方原文证据" in item.value for item in app.info)
        )
        next(button for button in app.button if button.label == "证据链").click()
        app.run(timeout=10)
        self.assertEqual(app.session_state["assessment_view"], "evidence")
        self.assertFalse(
            any("尚未为该规则绑定原子官方原文证据" in item.value for item in app.info)
        )
        self.assertEqual(
            len(
                [
                    item
                    for item in app.expander
                    if item.label.startswith("官方原文")
                ]
            ),
            4,
        )
        code_values = {item.value for item in app.code}
        self.assertTrue(
            {item.evidence_id for item in report.evidence}.issubset(code_values)
        )

    def test_explicit_unknown_is_recorded_without_repeating_question(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("Unresolved product relationship")
        app.text_area[0].set_value("Explicit Unknown response regression")
        next(button for button in app.button if button.label == "Create case").click()
        app.run(timeout=10)
        app.selectbox[0].set_value(UseDomain.PRODUCT_SAFETY)
        next(button for button in app.button if button.label == "Save facts").click()
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "Confirm module"
        ).click()
        app.run(timeout=10)

        product_question = next(
            select
            for select in app.selectbox
            if select.label.startswith("Is the AI system itself")
        )
        self.assertEqual(product_question.value, "__unanswered__")
        product_question.set_value("unknown")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)

        route = app.session_state["assessment_questionnaire_route"]
        self.assertEqual(
            route.recorded_unknown_question_ids,
            ["AI-ACT-6-1-AI-IS-PRODUCT"],
        )
        self.assertIn(
            "product_regulation.ai_is_product",
            route.missing_fact_paths["AI_ACT_HIGH_RISK_PRODUCT_SAFETY"],
        )
        self.assertEqual(self._progress_state(app, "facts"), "pending")
        self.assertNotIn(
            "Is the AI system itself a product placed on the market or put into service?",
            [select.label for select in app.selectbox],
        )
        self.assertIn(
            "Is the AI system intended to perform a safety function as part of a product?",
            [select.label for select in app.selectbox],
        )
        self.assertIn(
            "Recorded: Unknown",
            "\n".join(item.value for item in app.markdown),
        )
        response_record = next(
            record
            for record in app.session_state["assessment_fact_provenance"]
            if record.question_id == "AI-ACT-6-1-AI-IS-PRODUCT"
        )
        self.assertIs(
            response_record.response_state,
            QuestionResponseState.EXPLICIT_UNKNOWN,
        )

        app.radio[0].set_value("zh-CN")
        app.run(timeout=10)
        self.assertIn(
            "已记录：未知",
            "\n".join(item.value for item in app.markdown),
        )
        self.assertEqual(
            app.session_state["assessment_questionnaire_route"]
            .recorded_unknown_question_ids,
            ["AI-ACT-6-1-AI-IS-PRODUCT"],
        )
        app.radio[0].set_value("en")
        app.run(timeout=10)

        run_button = next(
            button for button in app.button if button.label == "Run assessment"
        )
        self.assertTrue(run_button.disabled)
        next(
            button
            for button in app.button
            if button.label == "Run with information gaps"
        ).click()
        app.run(timeout=10)
        report = app.session_state["assessment_report"]
        self.assertEqual(report.findings, [])
        self.assertTrue(
            any(
                item.rule_id == "AI_ACT_HIGH_RISK_PRODUCT_SAFETY"
                and item.fact_path == "product_regulation.ai_is_product"
                for item in report.missing_information
            )
        )
        self.assertTrue(
            any("Assessment incomplete" in item.value for item in app.markdown)
        )
        self.assertFalse(
            any("Report generated" in item.value for item in app.success)
        )

        next(button for button in app.button if button.label == "Assessment").click()
        app.run(timeout=10)
        next(button for button in app.button if button.label == "Edit answer").click()
        app.run(timeout=10)
        edited_question = next(
            select
            for select in app.selectbox
            if select.label.startswith("Is the AI system itself")
        )
        self.assertEqual(edited_question.value, "unknown")
        edited_question.set_value("yes")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)
        route = app.session_state["assessment_questionnaire_route"]
        self.assertEqual(route.recorded_unknown_question_ids, [])
        facts = app.session_state["assessment_workflow_bundle"].case_service.get_case(
            app.session_state["assessment_case_id"]
        ).current_facts
        self.assertIs(facts.product_regulation.ai_is_product, TriState.YES)
        self.assertIn(
            "What type of product is involved?",
            [item.label for item in app.text_input],
        )

    def test_removing_module_discards_explicit_unknown_response_state(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("Removable product module")
        app.text_area[0].set_value("Module response-state invalidation")
        next(button for button in app.button if button.label == "Create case").click()
        app.run(timeout=10)
        app.selectbox[0].set_value(UseDomain.PRODUCT_SAFETY)
        next(button for button in app.button if button.label == "Save facts").click()
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "Confirm module"
        ).click()
        app.run(timeout=10)
        next(
            select
            for select in app.selectbox
            if select.label.startswith("Is the AI system itself")
        ).set_value("unknown")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)
        self.assertTrue(
            any(
                record.response_state is QuestionResponseState.EXPLICIT_UNKNOWN
                for record in app.session_state["assessment_fact_provenance"]
            )
        )

        next(button for button in app.button if button.label == "Remove").click()
        app.run(timeout=10)

        self.assertNotIn(
            "AI_ACT_HIGH_RISK_PRODUCT_SAFETY",
            app.session_state["assessment_confirmed_modules"],
        )
        self.assertFalse(
            any(
                record.module_id == "AI_ACT_HIGH_RISK_PRODUCT_SAFETY"
                and record.response_state
                is QuestionResponseState.EXPLICIT_UNKNOWN
                for record in app.session_state["assessment_fact_provenance"]
            )
        )
        self.assertIsNone(app.session_state["assessment_report"])

    def test_explicit_unknown_annex_instrument_blocks_dependent_questions(self) -> None:
        instrument_id = "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("Unknown product legislation")
        app.text_area[0].set_value("Annex I dependency regression")
        next(button for button in app.button if button.label == "Create case").click()
        app.run(timeout=10)
        app.selectbox[0].set_value(UseDomain.PRODUCT_SAFETY)
        next(button for button in app.button if button.label == "Save facts").click()
        app.run(timeout=10)
        next(
            button for button in app.button if button.label == "Confirm module"
        ).click()
        app.run(timeout=10)
        next(
            select
            for select in app.selectbox
            if select.label.startswith("Is the AI system itself")
        ).set_value("yes")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)
        next(
            item
            for item in app.text_input
            if item.label == "What type of product is involved?"
        ).set_value("industrial machinery")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)
        next(
            select
            for select in app.selectbox
            if select.label.startswith("Which Annex I")
        ).set_value("ANNEX_I_INSTRUMENT_UNKNOWN")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)

        route = app.session_state["assessment_questionnaire_route"]
        self.assertEqual(
            route.recorded_unknown_question_ids,
            ["AI-ACT-6-1-ANNEX-I-INSTRUMENT"],
        )
        labels = [select.label for select in app.selectbox]
        self.assertNotIn(
            "Which Annex I product legislation may cover the product?",
            labels,
        )
        self.assertFalse(any("confirmed as applying" in label for label in labels))
        self.assertFalse(any("independent third party" in label for label in labels))
        self.assertTrue(
            any("module remains unresolved" in item.value for item in app.info)
        )
        self.assertEqual(self._progress_state(app, "facts"), "pending")

        primary_run = next(
            button for button in app.button if button.label == "Run assessment"
        )
        self.assertTrue(primary_run.disabled)
        gap_run = next(
            button
            for button in app.button
            if button.label == "Run with information gaps"
        )
        self.assertFalse(gap_run.disabled)
        gap_run.click()
        app.run(timeout=10)

        report = app.session_state["assessment_report"]
        self.assertEqual(report.findings, [])
        self.assertEqual(len(report.missing_information), 2)
        self.assertEqual(
            {item.rule_id for item in report.missing_information},
            {AI_ACT_PRODUCT_SAFETY_RULE_ID},
        )
        self.assertEqual(
            {item.fact_path for item in report.missing_information},
            {
                "product_regulation.annex_i_instrument",
                "product_regulation.annex_i_instrument_confirmed",
            },
        )
        self.assertEqual(
            report.authorized_rule_ids,
            [AI_ACT_PRODUCT_SAFETY_RULE_ID],
        )
        self.assertEqual(
            report.assessed_frameworks,
            [RegulatoryFramework.EU_AI_ACT],
        )
        self.assertEqual(app.session_state["assessment_run_mode"], "with_gaps")
        incomplete_markup = next(
            item.value
            for item in app.markdown
            if 'data-assessment-state="incomplete"' in item.value
        )
        self.assertIn("1 information gap", incomplete_markup)
        self.assertIn("EU AI Act", incomplete_markup)
        self.assertNotIn("GDPR", incomplete_markup)
        self.assertNotIn("EU Data Act", incomplete_markup)
        page_markup = "\n".join(item.value for item in app.markdown)
        self.assertIn("Assessment incomplete", page_markup)
        self.assertIn(
            "Has the selected Annex I legislation been confirmed as applying",
            page_markup,
        )
        self.assertIn(
            "Must an independent third party assess conformity",
            page_markup,
        )
        self.assertFalse(
            any("Report generated" in item.value for item in app.success)
        )
        self.assertFalse(
            any("atomic official source Evidence" in item.value for item in app.info)
        )
        evidence_button = next(
            button for button in app.button if button.label == "Evidence trace"
        )
        self.assertTrue(evidence_button.disabled)
        self.assertEqual(self._progress_state(app, "facts"), "pending")
        self.assertEqual(self._progress_state(app, "assessment"), "complete")
        self.assertTrue(
            {
                "Edit unresolved answer",
                "Return to Article 6(1) questions",
                "Continue assessment later",
            }.issubset({button.label for button in app.button})
        )

        next(
            button
            for button in app.button
            if button.label == "Edit unresolved answer"
        ).click()
        app.run(timeout=10)
        self.assertEqual(app.session_state["assessment_view"], "workspace")
        instrument_select = next(
            select
            for select in app.selectbox
            if select.label.startswith("Which Annex I")
        )
        self.assertEqual(instrument_select.value, "ANNEX_I_INSTRUMENT_UNKNOWN")
        instrument_select.set_value(instrument_id)
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)

        self.assertIsNone(app.session_state["assessment_report"])
        self.assertIsNone(app.session_state["assessment_run_mode"])
        self.assertIsNone(app.session_state["assessment_run_route"])
        self.assertTrue(
            any("confirmed as applying" in select.label for select in app.selectbox)
        )
        next(
            select
            for select in app.selectbox
            if "confirmed as applying" in select.label
        ).set_value("yes")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)
        next(
            select
            for select in app.selectbox
            if "independent third party" in select.label
        ).set_value("yes")
        next(
            button
            for button in app.button
            if button.label == "Save follow-up answers"
        ).click()
        app.run(timeout=10)

        primary_run = next(
            button for button in app.button if button.label == "Run assessment"
        )
        self.assertFalse(primary_run.disabled)
        self.assertFalse(
            any(
                button.label == "Run with information gaps"
                for button in app.button
            )
        )
        primary_run.click()
        app.run(timeout=10)
        finding = next(
            item
            for item in app.session_state["assessment_report"].findings
            if item.rule_id == AI_ACT_PRODUCT_SAFETY_RULE_ID
        )
        self.assertIs(finding.status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertTrue(
            any("Report generated" in item.value for item in app.success)
        )

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

    def test_unmatched_chinese_task_is_informational_context_only(self) -> None:
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

        self.assertNotIn(
            "Confirm the original text as the task description without legal classification",
            [checkbox.label for checkbox in app.checkbox],
        )
        self.assertTrue(
            any(
                "No controlled scenario was automatically identified" in item.value
                for item in app.info
            )
        )
        self.assertFalse(
            any("controlled expression" in item.value for item in app.warning)
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

    def test_structured_product_route_uses_info_for_unmatched_free_text(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_input[0].set_value("Factory safety assistant")
        app.text_area[0].set_value("Structured routing message regression")
        next(button for button in app.button if button.label == "Create case").click()
        app.run(timeout=10)

        app.selectbox[0].set_value(UseDomain.PRODUCT_SAFETY)
        original_text = "这是一个用于生产现场的辅助系统"
        app.text_area[0].set_value(original_text)
        next(button for button in app.button if button.label == "Save facts").click()
        app.run(timeout=10)

        route = app.session_state["assessment_questionnaire_route"]
        self.assertIn("AI_ACT_HIGH_RISK_PRODUCT_SAFETY", route.suggested_modules)
        self.assertEqual(
            app.session_state["assessment_input_normalization"]["original_text"],
            original_text,
        )
        facts = app.session_state["assessment_workflow_bundle"].case_service.get_case(
            app.session_state["assessment_case_id"]
        ).current_facts
        self.assertIsNone(facts.use_context.task)
        self.assertIn(
            original_text,
            "\n".join(item.value for item in app.markdown),
        )
        self.assertTrue(
            any(
                "No controlled scenario was automatically identified" in item.value
                for item in app.info
            )
        )
        self.assertFalse(
            any("controlled expression" in item.value for item in app.warning)
        )

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
            report.authorized_rule_ids,
            ["GDPR_ARTICLE22_RELEVANCE"],
        )
        self.assertEqual(report.missing_information, [])
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
        self.assertTrue(
            next(
                button
                for button in app.button
                if button.label == "Run assessment"
            ).disabled
        )
        next(
            button
            for button in app.button
            if button.label == "Run with information gaps"
        ).click()
        app.run(timeout=10)

        self.assertIsNotNone(app.session_state["assessment_report"])
        self.assertEqual(app.session_state["assessment_run_mode"], "with_gaps")
        self.assertEqual(self._progress_state(app, "facts"), "pending")
        self.assertEqual(self._progress_state(app, "assessment"), "complete")
        self.assertTrue(
            any("Assessment incomplete" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any(
                "Run completed without a substantive finding" in item.value
                for item in app.caption
            )
        )

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

    def test_module_removal_invalidates_report_and_execution_fingerprint(self) -> None:
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        self._open_recruitment_report(app)
        bundle = app.session_state["assessment_workflow_bundle"]
        report = app.session_state["assessment_report"]
        previous_run = bundle.workflow.get_run(report.assessment_run_reference)

        next(button for button in app.button if button.label == "Assessment").click()
        app.run(timeout=10)
        next(button for button in app.button if button.label == "Remove").click()
        app.run(timeout=10)

        self.assertEqual(app.session_state["assessment_confirmed_modules"], [])
        self.assertIsNone(app.session_state["assessment_report"])
        self.assertNotEqual(
            previous_run.input_fingerprint,
            bundle.workflow.input_fingerprint(
                app.session_state["assessment_case_id"],
                rule_ids=(),
            ),
        )

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
        self.assertEqual(
            [finding.rule_id for finding in report.findings],
            ["AI_ACT_HIGH_RISK_EMPLOYMENT"],
        )
        self.assertEqual(
            report.authorized_rule_ids,
            ["AI_ACT_HIGH_RISK_EMPLOYMENT"],
        )
        self.assertEqual(report.missing_information, [])
        self.assertEqual(len(report.evidence), 7)
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
        run_fingerprint = app.session_state[
            "assessment_workflow_bundle"
        ].workflow.get_run(report.assessment_run_reference).input_fingerprint

        english_markup = "\n".join(item.value for item in app.markdown)
        self.assertNotIn("ui-primary-missing", english_markup)
        self.assertNotIn("Personal data processing", english_markup)
        self.assertNotIn("Connected product", english_markup)
        self.assertNotIn(
            "Other framework screens",
            [item.label for item in app.expander],
        )
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
        self.assertNotIn("ui-primary-missing", primary_markup)
        self.assertIn("就业领域高风险初步分类", primary_markup)
        self.assertNotIn("data_act.", primary_markup)
        self.assertNotIn("data_protection.", primary_markup)
        self.assertNotIn("AIA_HIGH_RISK_EMPLOYMENT_PRELIMINARY", primary_markup)
        self.assertNotIn("其他框架筛查", [item.label for item in app.expander])
        raw_code = {item.value for item in app.code}
        self.assertTrue(
            {item.fact_path for item in report.missing_information}.isdisjoint(raw_code)
        )
        self.assertIn("AIA_HIGH_RISK_EMPLOYMENT_PRELIMINARY", raw_code)
        self.assertNotIn("data_act.connected_product", chinese_markup)
        self.assertEqual(app.session_state["assessment_report"].to_dict(), report_snapshot)
        self.assertEqual(
            app.session_state["assessment_workflow_bundle"].workflow.get_run(
                report.assessment_run_reference
            ).input_fingerprint,
            run_fingerprint,
        )

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
        self.assertEqual(
            report.authorized_rule_ids,
            ["EU_DATA_ACT_RELEVANCE"],
        )
        self.assertEqual(report.missing_information, [])
        self.assertEqual(len(report.evidence), 3)
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
        self.assertNotIn("其他框架筛查", [item.label for item in app.expander])
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
