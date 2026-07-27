"""Tests for deterministic bilingual legal fact normalization."""

from __future__ import annotations

import unittest

from src.assessment import AssessmentFacts, TriState
from src.assessment.engine import AssessmentEngine
from src.assessment.demo import create_assessment_workflow
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
from src.ui.questionnaire import hints_from_normalization


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

    def test_arbitrary_chinese_is_retained_without_controlled_suggestion(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.materially_influences_decision = TriState.YES
        result = normalize_legal_input("用来帮助人事团队的智能工具")

        apply_normalized_input(facts, result)
        assessment = self._engine().run(facts)

        self.assertEqual(result.status, NormalizationStatus.AMBIGUOUS)
        self.assertEqual(result.original_text, "用来帮助人事团队的智能工具")
        self.assertEqual(hints_from_normalization(result.mapping_ids), ())
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

    def test_product_safety_mapping_suggests_without_setting_legal_facts(
        self,
    ) -> None:
        result = normalize_legal_input(
            "AI safety component requiring third-party conformity assessment"
        )
        facts = AssessmentFacts()

        apply_normalized_input(facts, result)
        hints = hints_from_normalization(result.mapping_ids)

        self.assertEqual(result.status, NormalizationStatus.MATCHED)
        self.assertIn("ai_act.product_safety_component", hints)
        self.assertIn("ai_act.conformity_assessment", hints)
        self.assertIsNone(result.canonical_task)
        self.assertEqual(result.fact_updates, {})
        self.assertIs(
            facts.product_regulation.ai_is_product,
            TriState.UNKNOWN,
        )
        self.assertIs(
            facts.product_regulation.ai_is_safety_component,
            TriState.UNKNOWN,
        )
        self.assertIs(
            facts.product_regulation.annex_i_instrument_confirmed,
            TriState.UNKNOWN,
        )

    def test_english_and_chinese_product_safety_hints_are_equivalent(self) -> None:
        english = normalize_legal_input(
            "AI safety component and third-party conformity assessment"
        )
        chinese = normalize_legal_input("AI 安全部件与第三方合格评定")

        self.assertEqual(
            hints_from_normalization(english.mapping_ids),
            hints_from_normalization(chinese.mapping_ids),
        )
        self.assertEqual(english.fact_updates, chinese.fact_updates)

    def test_natural_chinese_product_safety_description_uses_phrase_patterns(
        self,
    ) -> None:
        text = (
            "监测工业机器人的运行状态，识别可能导致人员受伤或设备损坏的异常，"
            "并触发安全控制措施"
        )
        result = normalize_legal_input(text)
        facts = AssessmentFacts()

        apply_normalized_input(facts, result)
        hints = hints_from_normalization(result.mapping_ids)

        self.assertEqual(result.status, NormalizationStatus.MATCHED)
        self.assertIn("ai_act.product_safety_context", hints)
        self.assertIsNone(result.canonical_task)
        self.assertIsNone(facts.use_context.task)
        self.assertEqual(result.fact_updates, {})
        self.assertIs(facts.product_regulation.ai_is_product, TriState.UNKNOWN)
        self.assertIs(
            facts.product_regulation.ai_is_safety_component,
            TriState.UNKNOWN,
        )
        self.assertIs(
            facts.product_regulation.annex_i_instrument_confirmed,
            TriState.UNKNOWN,
        )
        self.assertIs(
            facts.product_regulation.third_party_conformity_required,
            TriState.UNKNOWN,
        )

    def test_phrase_matching_tolerates_punctuation_and_sentence_order(self) -> None:
        result = normalize_legal_input(
            "Before deployment, an independent third party reviews the "
            "protective-control functions of an industrial robot."
        )

        self.assertEqual(result.status, NormalizationStatus.MATCHED)
        self.assertEqual(
            set(hints_from_normalization(result.mapping_ids)),
            {
                "ai_act.product_safety_context",
                "ai_act.conformity_assessment",
            },
        )
        self.assertEqual(result.fact_updates, {})

    def test_product_safety_text_normalization_does_not_change_retrieval(self) -> None:
        bundle = create_assessment_workflow()
        before = bundle.evidence_retriever.retrieve(
            "EU_AI_ACT",
            "Article 6",
            limit=5,
        )

        facts = AssessmentFacts()
        result = normalize_legal_input(
            "Industrial robot protective control and emergency stop"
        )
        apply_normalized_input(facts, result)
        after = bundle.evidence_retriever.retrieve(
            "EU_AI_ACT",
            "Article 6",
            limit=5,
        )

        self.assertEqual(
            [evidence.evidence_id for evidence in after],
            [evidence.evidence_id for evidence in before],
        )


if __name__ == "__main__":
    unittest.main()
