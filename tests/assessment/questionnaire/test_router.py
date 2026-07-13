"""Tests for deterministic rule-driven questionnaire routing."""

from __future__ import annotations

import json
import unittest

from src.assessment import AssessmentFacts, TriState
from src.assessment.facts import AffectedPerson, UseDomain
from src.assessment.questionnaire import (
    QUESTION_DEFINITIONS,
    RULE_QUESTIONNAIRE_DEFINITIONS,
    UNSUPPORTED_PATH_DEFINITIONS,
    QuestionnaireRoute,
    build_default_questionnaire_router,
    build_question_registry,
    build_rule_questionnaire_registry,
)
from src.assessment.questionnaire.definitions import (
    AI_ACT_EMPLOYMENT_RULE_ID,
    EU_DATA_ACT_RULE_ID,
    GDPR_ARTICLE22_RULE_ID,
    HINT_CANDIDATE_RANKING,
    HINT_CREDIT_DECISION,
    HINT_RECRUITMENT,
)
from src.assessment.rules import (
    AIActHighRiskEmploymentRule,
    EUDataActRelevanceRule,
    GDPRArticle22RelevanceRule,
    RuleRegistry,
)


class QuestionnaireRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = build_default_questionnaire_router()

    @staticmethod
    def _complete_intake(facts: AssessmentFacts) -> None:
        facts.system.name = "Example system"
        facts.system.intended_purpose = "Assess a defined operational use case"
        facts.use_context.affected_persons = [AffectedPerson.CONSUMER]
        facts.use_context.human_review_before_effect = TriState.NO
        facts.data_act.connected_product = TriState.NO
        facts.data_act.related_service = TriState.NO

    def test_registry_is_complete_and_matches_actual_rule_requirements(self) -> None:
        rules = RuleRegistry(
            [
                AIActHighRiskEmploymentRule(),
                GDPRArticle22RelevanceRule(),
                EUDataActRelevanceRule(),
            ]
        )
        questions = build_question_registry()
        registry = build_rule_questionnaire_registry(rules, questions)

        self.assertEqual(
            [definition.rule_id for definition in registry],
            [
                AI_ACT_EMPLOYMENT_RULE_ID,
                GDPR_ARTICLE22_RULE_ID,
                EU_DATA_ACT_RULE_ID,
            ],
        )
        for definition in registry:
            with self.subTest(rule_id=definition.rule_id):
                rule = rules.get(definition.rule_id)
                self.assertEqual(
                    definition.required_fact_paths,
                    rule.required_fact_paths,
                )
                mapped_paths = {
                    questions.get(question_id).fact_path
                    for question_id in definition.question_ids
                }
                self.assertTrue(
                    set(definition.required_fact_paths).issubset(mapped_paths)
                )

    def test_all_questions_and_unsupported_messages_have_bilingual_keys(self) -> None:
        localized_items = [
            *(question.text_keys for question in QUESTION_DEFINITIONS),
            *(
                definition.confirmation_text_keys
                for definition in RULE_QUESTIONNAIRE_DEFINITIONS
            ),
            *(item.message_keys for item in UNSUPPORTED_PATH_DEFINITIONS),
        ]
        for item in localized_items:
            with self.subTest(en_key=item.en_label_key):
                self.assertTrue(item.en_label_key.endswith(".en"))
                self.assertTrue(item.en_help_key.endswith(".en"))
                self.assertTrue(item.zh_cn_label_key.endswith(".zh_cn"))
                self.assertTrue(item.zh_cn_help_key.endswith(".zh_cn"))
        self.assertEqual(len(RULE_QUESTIONNAIRE_DEFINITIONS), 3)

    def test_recruitment_suggests_ai_act_and_independent_gdpr_only(self) -> None:
        facts = AssessmentFacts()
        self._complete_intake(facts)
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = (
            "recruitment screening of candidates; candidate ranking"
        )
        facts.use_context.materially_influences_decision = TriState.YES
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.YES
        before = facts.to_dict()

        route = self.router.route(
            facts,
            confirmed_routing_hints=(
                HINT_RECRUITMENT,
                HINT_CANDIDATE_RANKING,
            ),
        )

        self.assertEqual(
            route.suggested_modules,
            [AI_ACT_EMPLOYMENT_RULE_ID, GDPR_ARTICLE22_RULE_ID],
        )
        self.assertIn(EU_DATA_ACT_RULE_ID, route.screened_out_modules)
        self.assertNotIn(EU_DATA_ACT_RULE_ID, route.missing_fact_paths)
        self.assertEqual(facts.to_dict(), before)
        self.assertIsInstance(route, QuestionnaireRoute)
        self.assertFalse(hasattr(route, "findings"))

    def test_industrial_connected_machinery_suggests_data_act_only(self) -> None:
        facts = AssessmentFacts()
        self._complete_intake(facts)
        facts.use_context.domain = UseDomain.PRODUCT_SAFETY
        facts.use_context.task = "Industrial connected machinery monitoring"
        facts.use_context.materially_influences_decision = TriState.NO
        facts.data_protection.personal_data_processed = TriState.NO
        facts.data_protection.automated_individual_decision = TriState.NO
        facts.data_act.connected_product = TriState.YES
        facts.data_act.related_service = TriState.YES
        facts.data_act.data_generated = TriState.YES

        route = self.router.route(facts)

        self.assertEqual(route.suggested_modules, [EU_DATA_ACT_RULE_ID])
        self.assertIn(AI_ACT_EMPLOYMENT_RULE_ID, route.screened_out_modules)
        self.assertFalse(route.unsupported_modules)

    def test_personal_loan_routes_gdpr_and_reports_credit_path_unsupported(self) -> None:
        facts = AssessmentFacts()
        self._complete_intake(facts)
        facts.use_context.domain = UseDomain.ESSENTIAL_SERVICES
        facts.use_context.task = "Confirmed automated consumer credit decision"
        facts.use_context.materially_influences_decision = TriState.YES
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.YES

        route = self.router.route(
            facts,
            confirmed_routing_hints=(HINT_CREDIT_DECISION,),
        )

        self.assertEqual(route.suggested_modules, [GDPR_ARTICLE22_RULE_ID])
        self.assertIn(AI_ACT_EMPLOYMENT_RULE_ID, route.screened_out_modules)
        self.assertEqual(
            [item.path_id for item in route.unsupported_modules],
            ["AI_ACT_ESSENTIAL_SERVICES_CREDIT_UNSUPPORTED"],
        )
        self.assertNotIn(
            "INTAKE-DECISION-IMPACT",
            [
                question.question_id
                for question in route.next_questions
                if "EMPLOYMENT" in question.question_id
            ],
        )
        self.assertTrue(
            all(
                "employment" not in question.text.casefold()
                for question in route.next_questions
            )
        )

    def test_judicial_scenario_is_unsupported_without_employment_questions(self) -> None:
        facts = AssessmentFacts()
        self._complete_intake(facts)
        facts.use_context.domain = UseDomain.JUSTICE_DEMOCRATIC_PROCESSES
        facts.use_context.task = "Assist judicial decision preparation"
        facts.use_context.materially_influences_decision = TriState.UNKNOWN
        facts.data_protection.personal_data_processed = TriState.NO
        facts.data_protection.automated_individual_decision = TriState.NO

        route = self.router.route(facts)

        self.assertEqual(route.suggested_modules, [])
        self.assertEqual(
            [item.path_id for item in route.unsupported_modules],
            ["AI_ACT_JUDICIAL_ROUTE_UNSUPPORTED"],
        )
        self.assertIn(AI_ACT_EMPLOYMENT_RULE_ID, route.screened_out_modules)
        self.assertFalse(hasattr(route, "findings"))

    def test_ambiguous_chinese_task_does_not_silently_activate_module(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "用来帮助团队处理人员事务的智能系统"

        route = self.router.route(facts)

        self.assertNotIn(AI_ACT_EMPLOYMENT_RULE_ID, route.suggested_modules)
        self.assertIn(AI_ACT_EMPLOYMENT_RULE_ID, route.screened_out_modules)
        self.assertEqual(route.module_confirmation_question_ids, [])

    def test_arbitrary_keyword_text_without_confirmed_hint_does_not_route(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "recruitment candidate ranking"

        route = self.router.route(facts)

        self.assertNotIn(AI_ACT_EMPLOYMENT_RULE_ID, route.suggested_modules)

    def test_confirmed_module_uses_existing_engine_for_missing_follow_up(self) -> None:
        facts = AssessmentFacts()
        self._complete_intake(facts)
        facts.use_context.domain = UseDomain.ESSENTIAL_SERVICES
        facts.use_context.task = "Consumer credit decision"
        facts.use_context.materially_influences_decision = TriState.YES
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.UNKNOWN

        route = self.router.route(
            facts,
            confirmed_modules=(GDPR_ARTICLE22_RULE_ID,),
            confirmed_routing_hints=(HINT_CREDIT_DECISION,),
        )

        self.assertEqual(
            route.missing_fact_paths[GDPR_ARTICLE22_RULE_ID],
            ["data_protection.automated_individual_decision"],
        )
        self.assertIn(
            "GDPR-AUTOMATED-DECISION",
            [question.question_id for question in route.next_questions],
        )
        json.dumps(route.to_dict())

    def test_question_and_module_ordering_is_stable(self) -> None:
        facts = AssessmentFacts()
        first = self.router.route(facts)
        second = self.router.route(facts)

        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            first.ordered_step_ids,
            [
                "INTAKE-SYSTEM-NAME",
                "INTAKE-SYSTEM-PURPOSE",
                "INTAKE-USE-DOMAIN",
                "INTAKE-AFFECTED-PERSONS",
                "INTAKE-HUMAN-REVIEW",
                "INTAKE-PERSONAL-DATA",
                "INTAKE-CONNECTED-PRODUCT",
            ],
        )

    def test_module_confirmation_precedes_confirmed_module_follow_up(self) -> None:
        facts = AssessmentFacts()
        self._complete_intake(facts)
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "recruitment screening of candidates"
        facts.use_context.materially_influences_decision = TriState.YES
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.UNKNOWN

        route = self.router.route(
            facts,
            confirmed_modules=(GDPR_ARTICLE22_RULE_ID,),
            confirmed_routing_hints=(HINT_RECRUITMENT,),
        )

        self.assertEqual(
            route.ordered_step_ids,
            [
                "CONFIRM-MODULE::AI_ACT_HIGH_RISK_EMPLOYMENT",
                "GDPR-AUTOMATED-DECISION",
            ],
        )


if __name__ == "__main__":
    unittest.main()
