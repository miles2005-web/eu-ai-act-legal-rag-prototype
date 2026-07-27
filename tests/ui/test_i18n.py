"""Tests for the centralized Streamlit presentation localization layer."""

from __future__ import annotations

import unittest

from src.ui.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_LABELS,
    SUPPORTED_LANGUAGES,
    count_text,
    normalize_language,
    t,
)


class I18nTests(unittest.TestCase):
    def test_english_is_default_and_supported_languages_are_stable(self) -> None:
        self.assertEqual(DEFAULT_LANGUAGE, "en")
        self.assertEqual(SUPPORTED_LANGUAGES, ("en", "zh-CN"))
        self.assertEqual(LANGUAGE_LABELS, {"en": "EN", "zh-CN": "中文"})

    def test_unknown_language_falls_back_to_english(self) -> None:
        self.assertEqual(normalize_language("fr"), "en")
        self.assertEqual(t("navigation.assessment", "fr"), "Assessment")

    def test_missing_chinese_translation_falls_back_to_english(self) -> None:
        self.assertEqual(t("test.english_only", "zh-CN"), "English fallback")

    def test_routed_questionnaire_copy_is_bilingual_and_neutral(self) -> None:
        self.assertEqual(
            t("question.decision_impact.label.en", "en"),
            "Does the output materially influence a decision or operational outcome?",
        )
        self.assertEqual(
            t("question.decision_impact.label.zh_cn", "zh-CN"),
            "系统输出是否会实质影响个人决定或运营结果？",
        )
        self.assertNotIn(
            "employment",
            t("question.decision_impact.label.en", "en").casefold(),
        )
        self.assertEqual(
            t("module.gdpr.article22", "zh-CN"),
            "GDPR 第 22 条相关性筛查",
        )

    def test_zero_finding_copy_does_not_imply_zero_legal_risk(self) -> None:
        self.assertEqual(
            t("report.no_finding", "en"),
            "No substantive assessment was produced. Additional facts or an "
            "implemented assessment module are required.",
        )
        self.assertEqual(
            t("report.no_finding", "zh-CN"),
            "尚未形成实质性评估结论。需要补充事实，或当前场景尚需相应的评估模块支持。",
        )

    def test_incomplete_assessment_copy_is_bilingual_and_non_conclusive(self) -> None:
        self.assertEqual(t("incomplete.title", "en"), "Assessment incomplete")
        self.assertEqual(t("incomplete.title", "zh-CN"), "评估尚未完成")
        self.assertEqual(
            t("assessment.run_with_gaps", "en"),
            "Run with information gaps",
        )
        self.assertEqual(
            t("assessment.run_with_gaps", "zh-CN"),
            "带信息缺口运行",
        )
        for language in SUPPORTED_LANGUAGES:
            copy = t("incomplete.summary", language).casefold()
            self.assertNotIn("compliant", copy)
            self.assertNotIn("does not apply", copy)

    def test_evidence_trace_hierarchy_copy_is_bilingual(self) -> None:
        self.assertEqual(
            t("trace.rule.mapping", "en"),
            "View condition-to-fact mapping",
        )
        self.assertEqual(
            t("trace.rule.mapping", "zh-CN"),
            "查看规则条件与事实映射",
        )
        self.assertEqual(
            t("trace.rule.conditions", "en", matched=3, total=3),
            "Conditions satisfied: 3 of 3",
        )
        self.assertEqual(
            t("trace.rule.conditions", "zh-CN", matched=3, total=3),
            "已满足条件：3/3",
        )

    def test_article_6_1_questions_and_boundary_are_bilingual(self) -> None:
        self.assertIn(
            "itself a product",
            t("question.ai_is_product.label.en", "en"),
        )
        self.assertIn(
            "本身是否作为产品",
            t("question.ai_is_product.label.zh_cn", "zh-CN"),
        )
        self.assertIn(
            "Selecting an instrument is not confirmation",
            t("question.annex_i_instrument_confirmed.help.en", "en"),
        )
        self.assertIn(
            "选择法规并不等于确认适用",
            t(
                "question.annex_i_instrument_confirmed.help.zh_cn",
                "zh-CN",
            ),
        )
        self.assertIn(
            "only the Article 6(1)",
            t("module.ai_act.product_safety.boundary", "en"),
        )
        self.assertIn(
            "仅评估《人工智能法案》第6条第1款",
            t("module.ai_act.product_safety.boundary", "zh-CN"),
        )

    def test_context_unknown_and_pending_evidence_copy_is_bilingual(self) -> None:
        self.assertIn(
            "saved as case context",
            t("normalization.ambiguous", "en"),
        )
        self.assertIn(
            "描述已作为案例信息保存",
            t("normalization.ambiguous", "zh-CN"),
        )
        self.assertEqual(
            t("questionnaire.response.recorded_unknown", "en"),
            "Recorded: Unknown",
        )
        self.assertEqual(
            t("questionnaire.response.recorded_unknown", "zh-CN"),
            "已记录：未知",
        )
        self.assertIn(
            "atomic official source Evidence",
            t("evidence.pending_binding", "en"),
        )
        self.assertIn(
            "原子官方原文证据",
            t("evidence.pending_binding", "zh-CN"),
        )

    def test_unknown_key_is_returned_safely(self) -> None:
        self.assertEqual(t("unknown.translation.key", "zh-CN"), "unknown.translation.key")

    def test_english_counts_use_natural_singular_plural_and_zero(self) -> None:
        self.assertEqual(count_text("report.findings", 1), "1 finding")
        self.assertEqual(count_text("report.findings", 2), "2 findings")
        self.assertEqual(count_text("report.evidence", 1), "1 evidence record")
        self.assertEqual(count_text("report.evidence", 7), "7 evidence records")
        self.assertEqual(count_text("report.gaps", 0), "No information gaps")

    def test_chinese_counts_and_domain_labels_are_localized(self) -> None:
        self.assertEqual(count_text("report.findings", 1, "zh-CN"), "1 项结论")
        self.assertEqual(count_text("report.evidence", 7, "zh-CN"), "7 条证据记录")
        self.assertEqual(count_text("report.gaps", 0, "zh-CN"), "无信息缺口")
        self.assertEqual(t("framework.EU_AI_ACT", "zh-CN"), "《欧盟人工智能法案》")
        self.assertEqual(t("status.potentially_applies", "zh-CN"), "可能适用")
        self.assertEqual(t("fact.use_context.domain", "zh-CN"), "就业场景")

    def test_predefined_scenario_copy_is_localized_by_stable_id(self) -> None:
        self.assertEqual(
            t(
                "scenario.recruitment-ai-ranking-candidates.case_name",
                "zh-CN",
            ),
            "招聘 AI 候选人筛选与排序",
        )
        self.assertEqual(
            t(
                "scenario.industrial-ai-connected-machinery-data-access.case_name",
                "zh-CN",
            ),
            "工业 AI 互联机械监测",
        )
        self.assertIn(
            "外部维护服务商请求访问",
            t(
                "scenario.industrial-ai-connected-machinery-data-access.description",
                "zh-CN",
            ),
        )
        self.assertEqual(
            t("demo.industrial_multi_framework.title", "en"),
            "Industrial Robot Safety and Data Access",
        )
        self.assertEqual(
            t("demo.industrial_multi_framework.title", "zh-CN"),
            "工业机器人安全与数据访问联合评估",
        )
        self.assertEqual(
            t("report.multiple_findings.summary", "en"),
            "Two independent regulatory screens produced substantive findings.",
        )
        self.assertEqual(
            t("report.multiple_findings.summary", "zh-CN"),
            "两项相互独立的监管筛查形成了实质性结论。",
        )
        self.assertEqual(
            t("missing_reason.path_not_found", "zh-CN"),
            "未找到事实字段",
        )

    def test_count_rejects_boolean_and_non_integer_values(self) -> None:
        with self.assertRaises(TypeError):
            count_text("report.findings", True)
        with self.assertRaises(TypeError):
            count_text("report.findings", 1.5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
