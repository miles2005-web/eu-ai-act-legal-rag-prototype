"""Tests for deterministic bilingual legal fact normalization."""

from __future__ import annotations

import unittest

from src.assessment import AssessmentFacts, TriState
from src.assessment.engine import AssessmentEngine
from src.assessment.facts import UseDomain
from src.assessment.rules import (
    AIActHighRiskEmploymentRule,
    EUDataActRelevanceRule,
    GDPRArticle22RelevanceRule,
    RuleRegistry,
)
from src.ui.normalization import (
    NormalizationStatus,
    apply_normalized_input,
    normalize_legal_input,
)


class LegalInputNormalizationTests(unittest.TestCase):
    @staticmethod
    def _engine() -> AssessmentEngine:
        return AssessmentEngine(
            RuleRegistry(
                [
                    AIActHighRiskEmploymentRule(),
                    GDPRArticle22RelevanceRule(),
                    EUDataActRelevanceRule(),
                ]
            )
        )

    def test_supported_recruitment_phrases_map_to_canonical_task(self) -> None:
        result = normalize_legal_input("招聘筛选，候选人排序")

        self.assertEqual(result.status, NormalizationStatus.MATCHED)
        self.assertEqual(
            result.canonical_task,
            "recruitment screening of candidates; candidate ranking",
        )
        self.assertEqual(
            result.mapping_ids,
            [
                "employment.recruitment_screening.v1",
                "employment.candidate_ranking.v1",
            ],
        )

    def test_supported_chinese_and_english_recruitment_inputs_are_equivalent(self) -> None:
        outcomes = []
        for text in (
            "招聘筛选 候选人排序 处理个人数据 完全自动化决定",
            (
                "recruitment screening candidate ranking processes personal "
                "data solely automated decision"
            ),
        ):
            facts = AssessmentFacts()
            facts.use_context.domain = UseDomain.EMPLOYMENT
            facts.use_context.materially_influences_decision = TriState.YES
            result = normalize_legal_input(text)
            apply_normalized_input(
                facts,
                result,
                protected_fact_paths=frozenset(
                    ("use_context.materially_influences_decision",)
                ),
            )
            assessment = self._engine().run(facts)
            outcomes.append(
                [
                    (finding.rule_id, finding.status.value, finding.reason_codes)
                    for finding in assessment.findings
                ]
            )

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(
            [rule_id for rule_id, _, _ in outcomes[0]],
            ["AI_ACT_HIGH_RISK_EMPLOYMENT", "GDPR_ARTICLE22_RELEVANCE"],
        )

    def test_supported_chinese_and_english_data_act_inputs_are_equivalent(self) -> None:
        outcomes = []
        for text in (
            "联网机械 相关服务 产品运行数据",
            "connected machinery related service product operational data",
        ):
            facts = AssessmentFacts()
            facts.use_context.domain = UseDomain.PRODUCT_SAFETY
            apply_normalized_input(facts, normalize_legal_input(text))
            assessment = self._engine().run(facts)
            outcomes.append(
                [
                    (finding.rule_id, finding.status.value, finding.reason_codes)
                    for finding in assessment.findings
                ]
            )

        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(
            outcomes[0],
            [
                (
                    "EU_DATA_ACT_RELEVANCE",
                    "potentially_applies",
                    ["CONNECTED_PRODUCT", "RELATED_SERVICE", "DATA_GENERATED"],
                )
            ],
        )

    def test_ambiguous_chinese_remains_unknown_without_confirmation(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.materially_influences_decision = TriState.YES
        result = normalize_legal_input("用来帮助人事团队的智能工具")

        apply_normalized_input(facts, result)
        assessment = self._engine().run(facts)

        self.assertEqual(result.status, NormalizationStatus.AMBIGUOUS)
        self.assertEqual(result.original_text, "用来帮助人事团队的智能工具")
        self.assertIsNone(facts.use_context.task)
        self.assertEqual(assessment.findings, [])
        self.assertIn(
            "use_context.task",
            {
                missing.fact_path
                for requirement in assessment.missing_fact_requirements
                for missing in requirement.missing_facts
            },
        )

    def test_negated_supported_phrase_does_not_create_positive_fact(self) -> None:
        facts = AssessmentFacts()
        result = normalize_legal_input("不处理个人数据")

        apply_normalized_input(facts, result)

        self.assertEqual(result.status, NormalizationStatus.AMBIGUOUS)
        self.assertEqual(
            facts.data_protection.personal_data_processed,
            TriState.UNKNOWN,
        )

    def test_original_text_and_mapping_ids_are_serializable(self) -> None:
        result = normalize_legal_input("联网设备和设备生成数据")

        serialized = result.to_dict()

        self.assertEqual(serialized["original_text"], "联网设备和设备生成数据")
        self.assertEqual(serialized["status"], "matched")
        self.assertEqual(
            serialized["fact_updates"],
            {
                "data_act.connected_product": "yes",
                "data_act.data_generated": "yes",
            },
        )

    def test_controlled_loan_phrases_are_auditable_without_silent_decision_fact(self) -> None:
        result = normalize_legal_input(
            "personal financial and credit analysis; automated loan "
            "approval/rejection; legal or significant economic effect"
        )

        self.assertEqual(result.status, NormalizationStatus.MATCHED)
        self.assertEqual(result.canonical_task, "automated consumer credit decision")
        self.assertIn("decision.credit.v1", result.mapping_ids)
        self.assertEqual(
            result.fact_updates["data_protection.personal_data_processed"],
            TriState.YES,
        )
        self.assertEqual(
            result.fact_updates["use_context.materially_influences_decision"],
            TriState.YES,
        )
        self.assertNotIn(
            "data_protection.automated_individual_decision",
            result.fact_updates,
        )


if __name__ == "__main__":
    unittest.main()
