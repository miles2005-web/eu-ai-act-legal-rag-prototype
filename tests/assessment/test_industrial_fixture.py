"""Tests for the industrial AI EU Data Act demonstration fixture."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
import unittest

from src.assessment import (
    AssessmentEngine,
    AssessmentFacts,
    FindingCategory,
    FindingStatus,
    RegulatoryFramework,
    TriState,
)
from src.assessment.facts import (
    AffectedPerson,
    FactMetadata,
    FactSource,
    LifecycleStatus,
    SystemOutput,
    UseDomain,
)
from src.assessment.rules import EUDataActRelevanceRule, RuleRegistry


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "industrial_ai_case.json"
)


class IndustrialAIFixtureTests(unittest.TestCase):
    @staticmethod
    def _load_fixture() -> dict[str, object]:
        with FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
            payload = json.load(fixture_file)
        if not isinstance(payload, dict):
            raise TypeError("fixture root must be an object")
        return payload

    @staticmethod
    def _build_relevant_facts(payload: dict[str, object]) -> AssessmentFacts:
        raw_facts = payload["facts"]
        if not isinstance(raw_facts, dict):
            raise TypeError("fixture facts must be an object")

        facts = AssessmentFacts(schema_version=raw_facts["schema_version"])
        system = raw_facts["system"]
        use_context = raw_facts["use_context"]
        data_act = raw_facts["data_act"]
        metadata = raw_facts["fact_metadata"]
        if not all(
            isinstance(section, dict)
            for section in (system, use_context, data_act, metadata)
        ):
            raise TypeError("fixture fact sections must be objects")

        facts.system.name = system["name"]
        facts.system.description = system["description"]
        facts.system.lifecycle_status = LifecycleStatus(
            system["lifecycle_status"]
        )
        facts.system.outputs = [
            SystemOutput(value) for value in system["outputs"]
        ]
        facts.use_context.domain = UseDomain(use_context["domain"])
        facts.use_context.task = use_context["task"]
        facts.use_context.affected_persons = [
            AffectedPerson(value) for value in use_context["affected_persons"]
        ]
        for fact_name, value in data_act.items():
            setattr(facts.data_act, fact_name, TriState(value))
        facts.fact_metadata = {
            fact_path: FactMetadata(
                source=FactSource(value["source"]),
                question_id=value["question_id"],
                recorded_at=datetime.fromisoformat(value["recorded_at"]),
            )
            for fact_path, value in metadata.items()
        }
        return facts

    def test_fixture_loads_as_json_with_assessment_fact_sections(self) -> None:
        payload = self._load_fixture()

        self.assertEqual(payload["fixture_version"], "1.0.0")
        self.assertEqual(
            payload["scenario_id"],
            "industrial-ai-connected-machinery-data-access",
        )
        expected_sections = {
            model_field.name
            for model_field in fields(AssessmentFacts)
        }
        self.assertEqual(set(payload["facts"]), expected_sections)

    def test_expected_outcome_fields_are_stable(self) -> None:
        expected = self._load_fixture()["expected_assessment"]

        self.assertEqual(
            expected,
            {
                "rule_id": "EU_DATA_ACT_RELEVANCE",
                "framework": "EU_DATA_ACT",
                "category": "DATA_GOVERNANCE",
                "status": "potentially_applies",
            },
        )

    def test_fixture_is_compatible_with_facts_and_expected_rule(self) -> None:
        payload = self._load_fixture()
        facts = self._build_relevant_facts(payload)
        result = AssessmentEngine(
            RuleRegistry([EUDataActRelevanceRule()])
        ).run(facts)

        self.assertEqual(facts.data_act.connected_product, TriState.YES)
        self.assertEqual(facts.data_act.related_service, TriState.YES)
        self.assertEqual(facts.data_act.data_generated, TriState.YES)
        self.assertEqual(facts.data_act.data_holder_identified, TriState.YES)
        self.assertEqual(
            facts.data_act.user_or_third_party_access_request,
            TriState.YES,
        )
        self.assertEqual(
            facts.fact_metadata["data_act.connected_product"].source,
            FactSource.QUESTIONNAIRE,
        )
        self.assertEqual(len(result.findings), 1)
        finding = result.findings[0]
        self.assertEqual(finding.rule_id, "EU_DATA_ACT_RELEVANCE")
        self.assertEqual(finding.framework, RegulatoryFramework.EU_DATA_ACT)
        self.assertEqual(finding.category, FindingCategory.DATA_GOVERNANCE)
        self.assertEqual(finding.status, FindingStatus.POTENTIALLY_APPLIES)


if __name__ == "__main__":
    unittest.main()
