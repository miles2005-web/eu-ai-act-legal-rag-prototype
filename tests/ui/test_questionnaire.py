"""Tests for the UI-neutral routed-questionnaire adapter."""

from __future__ import annotations

import unittest

from src.assessment import AssessmentFacts, TriState
from src.assessment.facts import AffectedPerson, UseDomain
from src.assessment.questionnaire import (
    QuestionResponseState,
    build_default_questionnaire_router,
)
from src.assessment.questionnaire.definitions import (
    AI_ACT_EMPLOYMENT_RULE_ID,
    AI_ACT_PRODUCT_SAFETY_RULE_ID,
    EU_DATA_ACT_RULE_ID,
    GDPR_ARTICLE22_RULE_ID,
    HINT_CREDIT_DECISION,
    HINT_PRODUCT_SAFETY_COMPONENT,
    RULE_QUESTIONNAIRE_DEFINITIONS,
)
from src.ui.normalization import normalize_legal_input
from src.ui.questionnaire import (
    QuestionnaireAnswer,
    apply_question_answers,
    authorized_rule_ids_for_modules,
    clear_fact_paths,
    confirmed_module_gaps,
    confirmed_missing_fact_paths,
    execution_facts_for_modules,
    hints_from_normalization,
    question_definition,
    question_option_label,
    required_facts_complete,
    universal_questions,
)


class QuestionnaireUIAdapterTests(unittest.TestCase):
    def test_confirmed_modules_map_to_rules_in_definition_order(self) -> None:
        self.assertEqual(
            authorized_rule_ids_for_modules(
                [GDPR_ARTICLE22_RULE_ID, AI_ACT_EMPLOYMENT_RULE_ID]
            ),
            (AI_ACT_EMPLOYMENT_RULE_ID, GDPR_ARTICLE22_RULE_ID),
        )

    def test_universal_intake_uses_stable_authored_question_ids(self) -> None:
        self.assertEqual(
            [question.question_id for question in universal_questions()],
            [
                "INTAKE-SYSTEM-NAME",
                "INTAKE-SYSTEM-PURPOSE",
                "INTAKE-USE-DOMAIN",
                "INTAKE-USE-TASK",
                "INTAKE-AFFECTED-PERSONS",
                "INTAKE-DECISION-IMPACT",
                "INTAKE-PERSONAL-DATA",
                "INTAKE-CONNECTED-PRODUCT",
                "INTAKE-RELATED-SERVICE",
            ],
        )

    def test_answers_write_canonical_values_and_provenance(self) -> None:
        facts = AssessmentFacts()
        provenance = apply_question_answers(
            facts,
            [
                QuestionnaireAnswer("INTAKE-USE-DOMAIN", "employment"),
                QuestionnaireAnswer(
                    "INTAKE-AFFECTED-PERSONS",
                    ["job_candidate", "worker"],
                ),
                QuestionnaireAnswer("INTAKE-DECISION-IMPACT", "yes"),
            ],
        )

        self.assertIs(facts.use_context.domain, UseDomain.EMPLOYMENT)
        self.assertEqual(
            facts.use_context.affected_persons,
            [AffectedPerson.JOB_CANDIDATE, AffectedPerson.WORKER],
        )
        self.assertIs(
            facts.use_context.materially_influences_decision,
            TriState.YES,
        )
        self.assertEqual(
            {record.fact_path for record in provenance},
            {
                "use_context.domain",
                "use_context.affected_persons",
                "use_context.materially_influences_decision",
            },
        )
        self.assertEqual(
            facts.fact_metadata["use_context.domain"].question_id,
            "INTAKE-USE-DOMAIN",
        )

    def test_explicit_unknown_has_persisted_response_state(self) -> None:
        facts = AssessmentFacts()

        provenance = apply_question_answers(
            facts,
            [
                QuestionnaireAnswer(
                    "AI-ACT-6-1-AI-IS-PRODUCT",
                    TriState.UNKNOWN,
                )
            ],
            module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
            record_explicit_unknown=True,
        )

        self.assertIs(
            facts.product_regulation.ai_is_product,
            TriState.UNKNOWN,
        )
        self.assertIs(
            provenance[0].response_state,
            QuestionResponseState.EXPLICIT_UNKNOWN,
        )
        self.assertEqual(
            provenance[0].question_id,
            "AI-ACT-6-1-AI-IS-PRODUCT",
        )

    def test_controlled_loan_input_routes_gdpr_and_unsupported_ai_act(self) -> None:
        text = (
            "personal financial and credit analysis; automated loan "
            "approval/rejection; legal or significant economic effect"
        )
        normalization = normalize_legal_input(text)
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.ESSENTIAL_SERVICES
        for path, value in normalization.fact_updates.items():
            target = facts
            segments = path.split(".")
            for segment in segments[:-1]:
                target = getattr(target, segment)
            setattr(target, segments[-1], value)
        facts.use_context.task = normalization.canonical_task
        hints = hints_from_normalization(normalization.mapping_ids)

        route = build_default_questionnaire_router().route(
            facts,
            confirmed_routing_hints=hints,
        )

        self.assertIn(HINT_CREDIT_DECISION, hints)
        self.assertEqual(route.suggested_modules, [GDPR_ARTICLE22_RULE_ID])
        self.assertEqual(
            [item.path_id for item in route.unsupported_modules],
            ["AI_ACT_ESSENTIAL_SERVICES_CREDIT_UNSUPPORTED"],
        )
        self.assertNotIn(AI_ACT_EMPLOYMENT_RULE_ID, route.suggested_modules)

    def test_english_and_chinese_loan_inputs_produce_identical_route_inputs(self) -> None:
        results = [
            normalize_legal_input(text)
            for text in (
                "personal financial and credit analysis; automated loan "
                "approval/rejection; legal or significant economic effect",
                "个人财务与信贷分析；自动贷款批准或拒绝；法律效果或重大经济影响",
            )
        ]

        self.assertEqual(results[0].canonical_task, results[1].canonical_task)
        self.assertEqual(results[0].fact_updates, results[1].fact_updates)
        self.assertEqual(
            hints_from_normalization(results[0].mapping_ids),
            hints_from_normalization(results[1].mapping_ids),
        )

    def test_execution_snapshot_includes_only_confirmed_module_inputs(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "recruitment screening of candidates"
        facts.use_context.materially_influences_decision = TriState.YES
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.YES
        facts.data_act.connected_product = TriState.YES
        facts.data_act.related_service = TriState.YES
        facts.data_act.data_generated = TriState.YES
        facts.product_regulation.ai_is_product = TriState.YES
        facts.product_regulation.annex_i_instrument = (
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
        )
        facts.product_regulation.annex_i_instrument_confirmed = TriState.YES
        facts.product_regulation.third_party_conformity_required = TriState.YES

        execution = execution_facts_for_modules(
            facts,
            [GDPR_ARTICLE22_RULE_ID],
        )

        self.assertIs(execution.use_context.domain, UseDomain.UNKNOWN)
        self.assertIsNone(execution.use_context.task)
        self.assertIs(
            execution.use_context.materially_influences_decision,
            TriState.YES,
        )
        self.assertIs(
            execution.data_protection.personal_data_processed,
            TriState.YES,
        )
        self.assertIs(execution.data_act.connected_product, TriState.UNKNOWN)
        self.assertIs(
            execution.product_regulation.ai_is_product,
            TriState.UNKNOWN,
        )
        self.assertIs(facts.use_context.domain, UseDomain.EMPLOYMENT)
        self.assertIs(facts.data_act.connected_product, TriState.YES)

    def test_clear_fact_paths_preserves_unrelated_framework_facts(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.task = "recruitment screening"
        facts.data_act.connected_product = TriState.YES

        clear_fact_paths(facts, ["use_context.task"])

        self.assertIsNone(facts.use_context.task)
        self.assertIs(facts.data_act.connected_product, TriState.YES)

    def test_all_implemented_module_ids_remain_available(self) -> None:
        facts = AssessmentFacts()
        for module_id in (
            AI_ACT_EMPLOYMENT_RULE_ID,
            AI_ACT_PRODUCT_SAFETY_RULE_ID,
            GDPR_ARTICLE22_RULE_ID,
            EU_DATA_ACT_RULE_ID,
        ):
            with self.subTest(module_id=module_id):
                execution_facts_for_modules(facts, [module_id])

    def test_annex_i_selector_stores_stable_id_without_legal_inference(
        self,
    ) -> None:
        facts = AssessmentFacts()
        instrument_id = "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"

        provenance = apply_question_answers(
            facts,
            [
                QuestionnaireAnswer(
                    "AI-ACT-6-1-ANNEX-I-INSTRUMENT",
                    instrument_id,
                )
            ],
            module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
        )

        self.assertEqual(
            facts.product_regulation.annex_i_instrument,
            instrument_id,
        )
        self.assertIs(
            facts.product_regulation.annex_i_instrument_confirmed,
            TriState.UNKNOWN,
        )
        self.assertIs(
            facts.product_regulation.third_party_conformity_required,
            TriState.UNKNOWN,
        )
        self.assertEqual(provenance[0].module_id, AI_ACT_PRODUCT_SAFETY_RULE_ID)

    def test_annex_i_labels_are_bilingual_without_changing_identity(self) -> None:
        instrument_id = "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"

        english = question_option_label(
            "AI-ACT-6-1-ANNEX-I-INSTRUMENT",
            instrument_id,
            "en",
        )
        chinese = question_option_label(
            "AI-ACT-6-1-ANNEX-I-INSTRUMENT",
            instrument_id,
            "zh-CN",
        )

        self.assertIn("Directive 2006/42/EC", english)
        self.assertIn("机械", chinese)
        self.assertIn("Annex I, Section A, point 1", english)
        self.assertIn("Annex I, Section A, point 1", chinese)
        self.assertNotIn(instrument_id, english)
        self.assertNotIn(instrument_id, chinese)

    def test_unknown_and_invalid_annex_i_selections_fail_safely(self) -> None:
        facts = AssessmentFacts()

        apply_question_answers(
            facts,
            [
                QuestionnaireAnswer(
                    "AI-ACT-6-1-ANNEX-I-INSTRUMENT",
                    "ANNEX_I_INSTRUMENT_UNKNOWN",
                )
            ],
        )
        self.assertIsNone(facts.product_regulation.annex_i_instrument)
        with self.assertRaises(KeyError):
            apply_question_answers(
                facts,
                [
                    QuestionnaireAnswer(
                        "AI-ACT-6-1-ANNEX-I-INSTRUMENT",
                        "machinery",
                    )
                ],
            )

    def test_product_route_completion_uses_conditional_requirements(self) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.YES
        facts.product_regulation.product_type = "machinery"
        facts.product_regulation.annex_i_instrument = (
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
        )
        facts.product_regulation.annex_i_instrument_confirmed = TriState.YES
        facts.product_regulation.third_party_conformity_required = TriState.YES

        route = build_default_questionnaire_router().route(
            facts,
            confirmed_modules=[AI_ACT_PRODUCT_SAFETY_RULE_ID],
            confirmed_routing_hints=[HINT_PRODUCT_SAFETY_COMPONENT],
        )

        self.assertEqual(confirmed_missing_fact_paths(route), ())
        self.assertTrue(required_facts_complete(route))
        self.assertIs(
            facts.product_regulation.ai_is_safety_component,
            TriState.UNKNOWN,
        )

    def test_questionnaire_metadata_keeps_product_type_non_conclusive(self) -> None:
        definition = question_definition("AI-ACT-6-1-PRODUCT-TYPE")

        self.assertEqual(
            definition.fact_path,
            "product_regulation.product_type",
        )
        self.assertNotIn(
            definition.fact_path,
            next(
                item.required_fact_paths
                for item in RULE_QUESTIONNAIRE_DEFINITIONS
                if item.rule_id == AI_ACT_PRODUCT_SAFETY_RULE_ID
            ),
        )

    def test_progress_uses_only_confirmed_module_missing_facts(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.ESSENTIAL_SERVICES
        facts.use_context.task = "automated consumer credit decision"
        facts.use_context.materially_influences_decision = TriState.YES
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.YES
        route = build_default_questionnaire_router().route(
            facts,
            confirmed_modules=[GDPR_ARTICLE22_RULE_ID],
            confirmed_routing_hints=[HINT_CREDIT_DECISION],
        )

        self.assertTrue(route.unsupported_modules)
        self.assertEqual(confirmed_missing_fact_paths(route), ())
        self.assertTrue(required_facts_complete(route))

    def test_incomplete_confirmed_module_reports_only_its_missing_fact(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.materially_influences_decision = TriState.YES
        facts.data_protection.personal_data_processed = TriState.YES
        route = build_default_questionnaire_router().route(
            facts,
            confirmed_modules=[GDPR_ARTICLE22_RULE_ID],
        )

        self.assertEqual(
            confirmed_missing_fact_paths(route),
            ("data_protection.automated_individual_decision",),
        )
        self.assertFalse(required_facts_complete(route))

    def test_explicit_unknown_is_counted_once_and_downstream_questions_are_blocked(
        self,
    ) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.YES
        facts.product_regulation.ai_is_safety_component = TriState.NO
        facts.product_regulation.product_type = "industrial machinery"
        provenance = apply_question_answers(
            facts,
            [
                QuestionnaireAnswer(
                    "AI-ACT-6-1-ANNEX-I-INSTRUMENT",
                    "ANNEX_I_INSTRUMENT_UNKNOWN",
                )
            ],
            module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
            record_explicit_unknown=True,
        )
        route = build_default_questionnaire_router().route(
            facts,
            confirmed_modules=[AI_ACT_PRODUCT_SAFETY_RULE_ID],
            confirmed_routing_hints=[HINT_PRODUCT_SAFETY_COMPONENT],
            fact_provenance=provenance,
        )

        gaps = confirmed_module_gaps(route, facts)

        self.assertEqual(len(gaps), 1)
        self.assertEqual(
            [item.question_id for item in gaps[0].unresolved],
            ["AI-ACT-6-1-ANNEX-I-INSTRUMENT"],
        )
        self.assertTrue(gaps[0].unresolved[0].recorded_unknown)
        self.assertEqual(gaps[0].unresolved_count, 1)
        self.assertEqual(
            [item.question_id for item in gaps[0].blocked],
            [
                "AI-ACT-6-1-ANNEX-I-CONFIRMED",
                "AI-ACT-6-1-THIRD-PARTY-CONFORMITY",
            ],
        )


if __name__ == "__main__":
    unittest.main()
