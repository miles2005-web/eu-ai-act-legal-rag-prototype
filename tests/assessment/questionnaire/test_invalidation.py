"""Tests for provenance-aware questionnaire dependency invalidation."""

from __future__ import annotations

from copy import deepcopy
import unittest

from src.assessment import AssessmentFacts, TriState
from src.assessment.facts import UseDomain
from src.assessment.questionnaire import (
    FactProvenance,
    QuestionResponseState,
    calculate_invalidations,
)
from src.assessment.questionnaire.definitions import (
    AI_ACT_EMPLOYMENT_RULE_ID,
    AI_ACT_PRODUCT_SAFETY_RULE_ID,
    EU_DATA_ACT_RULE_ID,
    GDPR_ARTICLE22_RULE_ID,
)


class QuestionnaireInvalidationTests(unittest.TestCase):
    def test_domain_change_invalidates_only_employment_dependent_answers(self) -> None:
        previous = AssessmentFacts()
        previous.use_context.domain = UseDomain.EMPLOYMENT
        previous.use_context.task = "recruitment screening of candidates"
        previous.use_context.materially_influences_decision = TriState.YES
        previous.data_protection.personal_data_processed = TriState.YES
        previous.data_act.connected_product = TriState.YES
        current = deepcopy(previous)
        current.use_context.domain = UseDomain.ESSENTIAL_SERVICES
        previous_snapshot = previous.to_dict()
        current_snapshot = current.to_dict()
        provenance = (
            FactProvenance(
                fact_path="use_context.task",
                question_id="INTAKE-USE-TASK",
                module_id=AI_ACT_EMPLOYMENT_RULE_ID,
                explicitly_confirmed=True,
                depends_on=("use_context.domain",),
            ),
            FactProvenance(
                fact_path="use_context.materially_influences_decision",
                question_id="INTAKE-DECISION-IMPACT",
                module_id=AI_ACT_EMPLOYMENT_RULE_ID,
                explicitly_confirmed=True,
                depends_on=("use_context.domain",),
            ),
            FactProvenance(
                fact_path="data_protection.personal_data_processed",
                question_id="INTAKE-PERSONAL-DATA",
                module_id=GDPR_ARTICLE22_RULE_ID,
                explicitly_confirmed=True,
            ),
            FactProvenance(
                fact_path="data_act.connected_product",
                question_id="INTAKE-CONNECTED-PRODUCT",
                module_id=EU_DATA_ACT_RULE_ID,
                explicitly_confirmed=True,
            ),
        )

        result = calculate_invalidations(previous, current, provenance)

        self.assertEqual(
            result.stale_fact_paths,
            [
                "use_context.task",
                "use_context.materially_influences_decision",
            ],
        )
        self.assertEqual(
            result.invalidated_question_ids,
            ["INTAKE-USE-TASK", "INTAKE-DECISION-IMPACT"],
        )
        self.assertEqual(
            result.invalidated_module_ids,
            [AI_ACT_EMPLOYMENT_RULE_ID],
        )
        self.assertNotIn(
            "data_protection.personal_data_processed",
            result.stale_fact_paths,
        )
        self.assertNotIn("data_act.connected_product", result.stale_fact_paths)
        self.assertEqual(previous.to_dict(), previous_snapshot)
        self.assertEqual(current.to_dict(), current_snapshot)

    def test_connected_product_change_invalidates_route_dependent_answers(self) -> None:
        previous = AssessmentFacts()
        previous.data_act.connected_product = TriState.YES
        previous.data_act.related_service = TriState.YES
        previous.data_act.data_generated = TriState.YES
        previous.data_protection.personal_data_processed = TriState.YES
        current = deepcopy(previous)
        current.data_act.connected_product = TriState.NO
        provenance = (
            FactProvenance(
                fact_path="data_act.related_service",
                question_id="INTAKE-RELATED-SERVICE",
                module_id=EU_DATA_ACT_RULE_ID,
                explicitly_confirmed=True,
                depends_on=("data_act.connected_product",),
            ),
            FactProvenance(
                fact_path="data_act.data_generated",
                question_id="DATA-ACT-DATA-GENERATED",
                module_id=EU_DATA_ACT_RULE_ID,
                explicitly_confirmed=True,
                depends_on=("data_act.connected_product",),
            ),
            FactProvenance(
                fact_path="data_protection.personal_data_processed",
                question_id="INTAKE-PERSONAL-DATA",
                module_id=GDPR_ARTICLE22_RULE_ID,
                explicitly_confirmed=True,
            ),
        )

        result = calculate_invalidations(previous, current, provenance)

        self.assertEqual(
            result.changed_upstream_fact_paths,
            ["data_act.connected_product"],
        )
        self.assertEqual(
            result.stale_fact_paths,
            ["data_act.related_service", "data_act.data_generated"],
        )
        self.assertEqual(
            result.invalidated_module_ids,
            [EU_DATA_ACT_RULE_ID],
        )
        self.assertNotIn(
            "data_protection.personal_data_processed",
            result.stale_fact_paths,
        )

    def test_both_product_relationship_branches_no_invalidate_only_downstream(
        self,
    ) -> None:
        previous = AssessmentFacts()
        previous.product_regulation.ai_is_product = TriState.YES
        previous.product_regulation.ai_is_safety_component = TriState.NO
        previous.product_regulation.product_type = "machinery"
        previous.product_regulation.annex_i_instrument = (
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
        )
        previous.product_regulation.annex_i_instrument_confirmed = TriState.YES
        previous.product_regulation.third_party_conformity_required = TriState.YES
        previous.data_protection.personal_data_processed = TriState.YES
        previous.data_act.connected_product = TriState.YES
        current = deepcopy(previous)
        current.product_regulation.ai_is_product = TriState.NO
        provenance = (
            FactProvenance(
                fact_path="product_regulation.product_type",
                question_id="AI-ACT-6-1-PRODUCT-TYPE",
                module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                depends_on=(
                    "product_regulation.ai_is_product",
                    "product_regulation.ai_is_safety_component",
                ),
            ),
            FactProvenance(
                fact_path="product_regulation.annex_i_instrument",
                question_id="AI-ACT-6-1-ANNEX-I-INSTRUMENT",
                module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                depends_on=(
                    "product_regulation.product_type",
                    "product_regulation.ai_is_product",
                    "product_regulation.ai_is_safety_component",
                ),
            ),
            FactProvenance(
                fact_path="product_regulation.annex_i_instrument_confirmed",
                question_id="AI-ACT-6-1-ANNEX-I-CONFIRMED",
                module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                depends_on=(
                    "product_regulation.product_type",
                    "product_regulation.annex_i_instrument",
                    "product_regulation.ai_is_product",
                    "product_regulation.ai_is_safety_component",
                ),
            ),
            FactProvenance(
                fact_path="product_regulation.third_party_conformity_required",
                question_id="AI-ACT-6-1-THIRD-PARTY-CONFORMITY",
                module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                depends_on=(
                    "product_regulation.product_type",
                    "product_regulation.annex_i_instrument",
                    "product_regulation.annex_i_instrument_confirmed",
                    "product_regulation.ai_is_product",
                    "product_regulation.ai_is_safety_component",
                ),
            ),
            FactProvenance(
                fact_path="data_protection.personal_data_processed",
                question_id="INTAKE-PERSONAL-DATA",
                module_id=GDPR_ARTICLE22_RULE_ID,
            ),
            FactProvenance(
                fact_path="data_act.connected_product",
                question_id="INTAKE-CONNECTED-PRODUCT",
                module_id=EU_DATA_ACT_RULE_ID,
            ),
        )

        result = calculate_invalidations(previous, current, provenance)

        self.assertEqual(
            result.stale_fact_paths,
            [
                "product_regulation.product_type",
                "product_regulation.annex_i_instrument",
                "product_regulation.annex_i_instrument_confirmed",
                "product_regulation.third_party_conformity_required",
            ],
        )
        self.assertNotIn(
            "data_protection.personal_data_processed",
            result.stale_fact_paths,
        )
        self.assertNotIn(
            "data_act.connected_product",
            result.stale_fact_paths,
        )

    def test_other_positive_relationship_branch_prevents_invalidation(self) -> None:
        previous = AssessmentFacts()
        previous.product_regulation.ai_is_product = TriState.YES
        previous.product_regulation.ai_is_safety_component = TriState.YES
        previous.product_regulation.product_type = "machinery"
        current = deepcopy(previous)
        current.product_regulation.ai_is_product = TriState.NO
        provenance = (
            FactProvenance(
                fact_path="product_regulation.product_type",
                question_id="AI-ACT-6-1-PRODUCT-TYPE",
                module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                depends_on=(
                    "product_regulation.ai_is_product",
                    "product_regulation.ai_is_safety_component",
                ),
            ),
        )

        result = calculate_invalidations(previous, current, provenance)

        self.assertEqual(result.stale_fact_paths, [])

    def test_instrument_change_invalidates_confirmation_and_conformity(
        self,
    ) -> None:
        previous = AssessmentFacts()
        previous.product_regulation.ai_is_product = TriState.YES
        previous.product_regulation.product_type = "machinery"
        previous.product_regulation.annex_i_instrument = (
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
        )
        previous.product_regulation.annex_i_instrument_confirmed = TriState.YES
        previous.product_regulation.third_party_conformity_required = TriState.YES
        current = deepcopy(previous)
        current.product_regulation.annex_i_instrument = (
            "ANNEX_I_A_02_TOY_SAFETY_DIRECTIVE_2009_48_EC"
        )
        provenance = (
            FactProvenance(
                fact_path="product_regulation.annex_i_instrument_confirmed",
                question_id="AI-ACT-6-1-ANNEX-I-CONFIRMED",
                module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                explicitly_confirmed=True,
                depends_on=("product_regulation.annex_i_instrument",),
            ),
            FactProvenance(
                fact_path="product_regulation.third_party_conformity_required",
                question_id="AI-ACT-6-1-THIRD-PARTY-CONFORMITY",
                module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                explicitly_confirmed=True,
                depends_on=("product_regulation.annex_i_instrument",),
            ),
        )

        result = calculate_invalidations(previous, current, provenance)

        self.assertEqual(
            result.stale_fact_paths,
            [
                "product_regulation.annex_i_instrument_confirmed",
                "product_regulation.third_party_conformity_required",
            ],
        )

    def test_product_type_change_respects_explicit_instrument_provenance(
        self,
    ) -> None:
        previous = AssessmentFacts()
        previous.product_regulation.product_type = "machinery"
        current = deepcopy(previous)
        current.product_regulation.product_type = "medical device"

        def calculate(explicitly_confirmed: bool):
            return calculate_invalidations(
                previous,
                current,
                (
                    FactProvenance(
                        fact_path="product_regulation.annex_i_instrument",
                        question_id="AI-ACT-6-1-ANNEX-I-INSTRUMENT",
                        module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                        explicitly_confirmed=explicitly_confirmed,
                        depends_on=("product_regulation.product_type",),
                    ),
                ),
            )

        self.assertEqual(calculate(True).stale_fact_paths, [])
        self.assertEqual(
            calculate(False).stale_fact_paths,
            ["product_regulation.annex_i_instrument"],
        )

    def test_instrument_change_invalidates_explicit_unknown_response(self) -> None:
        previous = AssessmentFacts()
        previous.product_regulation.annex_i_instrument = (
            "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
        )
        current = deepcopy(previous)
        current.product_regulation.annex_i_instrument = (
            "ANNEX_I_A_02_TOY_SAFETY_DIRECTIVE_2009_48_EC"
        )
        provenance = (
            FactProvenance(
                fact_path="product_regulation.annex_i_instrument_confirmed",
                question_id="AI-ACT-6-1-ANNEX-I-CONFIRMED",
                module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                explicitly_confirmed=True,
                depends_on=("product_regulation.annex_i_instrument",),
                response_state=QuestionResponseState.EXPLICIT_UNKNOWN,
            ),
            FactProvenance(
                fact_path="data_protection.personal_data_processed",
                question_id="INTAKE-PERSONAL-DATA",
                module_id=GDPR_ARTICLE22_RULE_ID,
                explicitly_confirmed=True,
            ),
        )

        result = calculate_invalidations(previous, current, provenance)

        self.assertEqual(
            result.stale_fact_paths,
            ["product_regulation.annex_i_instrument_confirmed"],
        )
        self.assertNotIn(
            "data_protection.personal_data_processed",
            result.stale_fact_paths,
        )

    def test_relationship_change_invalidates_dependent_explicit_unknown(self) -> None:
        previous = AssessmentFacts()
        previous.product_regulation.ai_is_product = TriState.YES
        previous.product_regulation.ai_is_safety_component = TriState.NO
        previous.product_regulation.product_type = "machinery"
        current = deepcopy(previous)
        current.product_regulation.ai_is_product = TriState.NO
        provenance = (
            FactProvenance(
                fact_path="product_regulation.annex_i_instrument",
                question_id="AI-ACT-6-1-ANNEX-I-INSTRUMENT",
                module_id=AI_ACT_PRODUCT_SAFETY_RULE_ID,
                explicitly_confirmed=True,
                depends_on=(
                    "product_regulation.product_type",
                    "product_regulation.ai_is_product",
                    "product_regulation.ai_is_safety_component",
                ),
                response_state=QuestionResponseState.EXPLICIT_UNKNOWN,
            ),
            FactProvenance(
                fact_path="data_act.connected_product",
                question_id="INTAKE-CONNECTED-PRODUCT",
                module_id=EU_DATA_ACT_RULE_ID,
                explicitly_confirmed=True,
            ),
        )

        result = calculate_invalidations(previous, current, provenance)

        self.assertEqual(
            result.stale_fact_paths,
            ["product_regulation.annex_i_instrument"],
        )
        self.assertEqual(
            result.removed_provenance_fact_paths,
            ["product_regulation.annex_i_instrument"],
        )
        self.assertNotIn("data_act.connected_product", result.stale_fact_paths)


if __name__ == "__main__":
    unittest.main()
