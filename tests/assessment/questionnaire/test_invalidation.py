"""Tests for provenance-aware questionnaire dependency invalidation."""

from __future__ import annotations

from copy import deepcopy
import unittest

from src.assessment import AssessmentFacts, TriState
from src.assessment.facts import UseDomain
from src.assessment.questionnaire import FactProvenance, calculate_invalidations
from src.assessment.questionnaire.definitions import (
    AI_ACT_EMPLOYMENT_RULE_ID,
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


if __name__ == "__main__":
    unittest.main()
