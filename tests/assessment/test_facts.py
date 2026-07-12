"""Tests for the cross-regulation assessment fact foundation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import unittest

from scripts.run_demo_assessment import build_assessment_facts, load_fixture
from src.assessment import (
    AssessmentEngine,
    FactRequirementValidator,
    FindingStatus,
    TriState,
)
from src.assessment.facts import FactMetadata, FactSource
from src.assessment.rules import AIActHighRiskEmploymentRule, RuleRegistry


class DataProtectionFactsTests(unittest.TestCase):
    def test_unknown_values_are_the_backward_compatible_default(self) -> None:
        facts = build_assessment_facts(load_fixture()["facts"])

        self.assertEqual(
            facts.data_protection.personal_data_processed,
            TriState.UNKNOWN,
        )
        self.assertEqual(
            facts.data_protection.automated_individual_decision,
            TriState.UNKNOWN,
        )
        self.assertEqual(
            facts.data_protection.special_category_data_processed,
            TriState.UNKNOWN,
        )

    def test_data_protection_facts_and_provenance_serialize(self) -> None:
        facts = build_assessment_facts(load_fixture()["facts"])
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.YES
        facts.data_protection.special_category_data_processed = TriState.NO
        recorded_at = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
        facts.fact_metadata["data_protection.personal_data_processed"] = (
            FactMetadata(
                source=FactSource.QUESTIONNAIRE,
                question_id="GDPR-PERSONAL-DATA",
                recorded_at=recorded_at,
            )
        )

        payload = facts.to_dict()

        self.assertEqual(
            payload["data_protection"],
            {
                "personal_data_processed": "yes",
                "automated_individual_decision": "yes",
                "special_category_data_processed": "no",
            },
        )
        self.assertEqual(
            payload["fact_metadata"][
                "data_protection.personal_data_processed"
            ],
            {
                "source": "questionnaire",
                "question_id": "GDPR-PERSONAL-DATA",
                "recorded_at": recorded_at.isoformat(),
            },
        )
        json.dumps(payload)

    def test_existing_fixture_still_executes_ai_act_rule(self) -> None:
        facts = build_assessment_facts(load_fixture()["facts"])
        rule = AIActHighRiskEmploymentRule()

        requirement_result = FactRequirementValidator().validate(rule, facts)
        result = AssessmentEngine(RuleRegistry([rule])).run(facts)

        self.assertTrue(requirement_result.is_satisfied)
        self.assertEqual(result.executed_rule_ids, [rule.rule_id])
        self.assertEqual(len(result.findings), 1)
        self.assertEqual(
            result.findings[0].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(result.missing_fact_requirements, [])


if __name__ == "__main__":
    unittest.main()
