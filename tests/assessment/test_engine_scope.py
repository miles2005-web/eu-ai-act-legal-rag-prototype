"""Regression tests for deterministic authorized rule execution."""

from __future__ import annotations

import unittest

from src.assessment import AssessmentEngine, AssessmentFacts
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.rules import (
    AIActHighRiskEmploymentRule,
    EUDataActRelevanceRule,
    GDPRArticle22RelevanceRule,
    RuleRegistry,
)


class AssessmentEngineScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = AssessmentEngine(
            RuleRegistry(
                [
                    AIActHighRiskEmploymentRule(),
                    GDPRArticle22RelevanceRule(),
                    EUDataActRelevanceRule(),
                ]
            )
        )

    def test_selected_rules_execute_in_registry_order(self) -> None:
        result = self.engine.run(
            AssessmentFacts(),
            rule_ids=("EU_DATA_ACT_RELEVANCE", "GDPR_ARTICLE22_RELEVANCE"),
        )

        self.assertEqual(
            result.authorized_rule_ids,
            ["GDPR_ARTICLE22_RELEVANCE", "EU_DATA_ACT_RELEVANCE"],
        )
        self.assertEqual(
            [item.rule_id for item in result.missing_fact_requirements],
            ["GDPR_ARTICLE22_RELEVANCE", "EU_DATA_ACT_RELEVANCE"],
        )
        self.assertEqual(
            result.assessed_frameworks,
            [RegulatoryFramework.GDPR, RegulatoryFramework.EU_DATA_ACT],
        )

    def test_omitted_scope_preserves_full_engine_execution(self) -> None:
        result = self.engine.run(AssessmentFacts())

        self.assertEqual(
            result.authorized_rule_ids,
            list(self.engine.registered_rule_ids),
        )
        self.assertEqual(
            [item.rule_id for item in result.missing_fact_requirements],
            list(self.engine.registered_rule_ids),
        )

    def test_explicit_empty_scope_runs_no_rules(self) -> None:
        result = self.engine.run(AssessmentFacts(), rule_ids=())

        self.assertEqual(result.authorized_rule_ids, [])
        self.assertEqual(result.executed_rule_ids, [])
        self.assertEqual(result.missing_fact_requirements, [])
        self.assertEqual(result.assessed_frameworks, [])

    def test_invalid_scope_is_rejected_without_mutating_registry(self) -> None:
        original_ids = self.engine.registered_rule_ids

        with self.assertRaises(ValueError):
            self.engine.run(AssessmentFacts(), rule_ids=("NOT_REGISTERED",))

        self.assertEqual(self.engine.registered_rule_ids, original_ids)


if __name__ == "__main__":
    unittest.main()
