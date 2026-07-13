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
