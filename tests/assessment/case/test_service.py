"""Tests for in-memory assessment case lifecycle operations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import unittest

from src.assessment import (
    AssessmentCaseNotFoundError,
    AssessmentCaseSchemaMismatchError,
    AssessmentCaseService,
    AssessmentFacts,
    AssessmentRun,
)


class AssessmentCaseServiceTests(unittest.TestCase):
    def test_create_and_retrieve_are_isolated(self) -> None:
        facts = AssessmentFacts()
        facts.system.name = "Original system"
        service = AssessmentCaseService()

        created = service.create_case(
            "Employment assessment",
            description="Preliminary EU AI Act review",
            facts=facts,
            case_id="case-001",
        )

        facts.system.name = "Mutated input"
        created.current_facts.system.name = "Mutated returned case"
        retrieved = service.get_case("case-001")
        self.assertEqual(retrieved.case_id, "case-001")
        self.assertEqual(retrieved.current_facts.system.name, "Original system")
        self.assertEqual(retrieved.schema_version, "2.0.0")
        self.assertEqual(len(service), 1)
        json.dumps(retrieved.to_dict())

    def test_update_replaces_current_facts_but_not_run_snapshot(self) -> None:
        created_at = datetime(2026, 7, 11, 8, 0, tzinfo=timezone.utc)
        updated_at = datetime(2026, 7, 11, 9, 0, tzinfo=timezone.utc)
        timestamps = iter((created_at, updated_at))
        service = AssessmentCaseService(clock=lambda: next(timestamps))

        original_facts = AssessmentFacts()
        original_facts.system.name = "Version one"
        created = service.create_case(
            "Versioned case",
            facts=original_facts,
            case_id="case-history",
        )
        historical_run = AssessmentRun(
            case_id=created.case_id,
            facts_snapshot=created.current_facts,
        )

        replacement_facts = AssessmentFacts()
        replacement_facts.system.name = "Version two"
        updated = service.update_facts(created.case_id, replacement_facts)

        self.assertEqual(updated.current_facts.system.name, "Version two")
        self.assertEqual(updated.created_at, created_at)
        self.assertEqual(updated.updated_at, updated_at)
        self.assertEqual(
            historical_run.facts_snapshot.system.name,
            "Version one",
        )

    def test_missing_case_and_schema_mismatch_raise_domain_errors(self) -> None:
        service = AssessmentCaseService()
        with self.assertRaises(AssessmentCaseNotFoundError):
            service.get_case("missing")

        service.create_case("Schema case", case_id="schema-case")
        incompatible_facts = AssessmentFacts(schema_version="3.0.0")
        with self.assertRaises(AssessmentCaseSchemaMismatchError):
            service.update_facts("schema-case", incompatible_facts)


if __name__ == "__main__":
    unittest.main()
