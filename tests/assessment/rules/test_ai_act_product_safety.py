"""Tests for the EU AI Act Article 6(1) product-safety route."""

from __future__ import annotations

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
from src.assessment.facts import ProductRegulationFacts, UseDomain
from src.assessment.requirements import (
    FactRequirementValidator,
    MissingFactReason,
)
from src.assessment.rules import (
    AIActHighRiskEmploymentRule,
    AIActHighRiskProductSafetyRule,
    EUDataActRelevanceRule,
    GDPRArticle22RelevanceRule,
    RuleRegistry,
)


MACHINERY_ID = "ANNEX_I_A_01_MACHINERY_DIRECTIVE_2006_42_EC"
MACHINERY_REFERENCE = "Annex I, Section A, point 1"


class AIActHighRiskProductSafetyRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = AIActHighRiskProductSafetyRule()
        self.engine = AssessmentEngine(RuleRegistry([self.rule]))

    @staticmethod
    def _positive_facts(*, safety_component: bool = False) -> AssessmentFacts:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = (
            TriState.NO if safety_component else TriState.YES
        )
        facts.product_regulation.ai_is_safety_component = (
            TriState.YES if safety_component else TriState.NO
        )
        facts.product_regulation.product_type = "machinery"
        facts.product_regulation.annex_i_instrument = MACHINERY_ID
        facts.product_regulation.annex_i_instrument_confirmed = TriState.YES
        facts.product_regulation.third_party_conformity_required = TriState.YES
        return facts

    def _run_one(self, facts: AssessmentFacts):
        result = self.engine.run(facts)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.missing_fact_requirements, [])
        self.assertEqual(
            result.executed_rule_ids,
            ["AI_ACT_HIGH_RISK_PRODUCT_SAFETY"],
        )
        self.assertEqual(len(result.findings), 1)
        return result.findings[0]

    def test_product_path_potentially_applies(self) -> None:
        finding = self._run_one(self._positive_facts())

        self.assertEqual(finding.framework, RegulatoryFramework.EU_AI_ACT)
        self.assertEqual(
            finding.category,
            FindingCategory.HIGH_RISK_ARTICLE_6_1,
        )
        self.assertEqual(finding.status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertEqual(
            finding.reason_codes,
            [
                "AI_IS_PRODUCT",
                "ANNEX_I_COVERAGE_CONFIRMED",
                "THIRD_PARTY_CONFORMITY_REQUIRED",
            ],
        )
        self.assertTrue(finding.requires_legal_review)

    def test_safety_component_path_potentially_applies(self) -> None:
        finding = self._run_one(
            self._positive_facts(safety_component=True)
        )

        self.assertEqual(finding.status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertEqual(
            finding.reason_codes[0],
            "AI_IS_SAFETY_COMPONENT",
        )
        self.assertEqual(
            [basis.citation for basis in finding.legal_basis],
            [
                "Article 3(14)",
                "Article 6(1)(a)",
                "Article 6(1)(b)",
                MACHINERY_REFERENCE,
            ],
        )

    def test_positive_or_branch_short_circuits_unknown_other_branch(self) -> None:
        facts = self._positive_facts()
        facts.product_regulation.ai_is_safety_component = TriState.UNKNOWN

        finding = self._run_one(facts)

        self.assertEqual(finding.status, FindingStatus.POTENTIALLY_APPLIES)
        self.assertEqual(finding.reason_codes[0], "AI_IS_PRODUCT")

    def test_product_branch_is_deterministic_when_both_branches_are_yes(
        self,
    ) -> None:
        facts = self._positive_facts()
        facts.product_regulation.ai_is_safety_component = TriState.YES

        finding = self._run_one(facts)

        self.assertEqual(finding.reason_codes[0], "AI_IS_PRODUCT")
        self.assertNotIn(
            "Article 3(14)",
            [basis.citation for basis in finding.legal_basis],
        )

    def test_both_product_relationship_branches_no_is_negative(self) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.NO
        facts.product_regulation.ai_is_safety_component = TriState.NO

        finding = self._run_one(facts)

        self.assertEqual(finding.status, FindingStatus.DOES_NOT_APPLY)
        self.assertEqual(
            finding.reason_codes,
            ["NEITHER_AI_PRODUCT_NOR_SAFETY_COMPONENT"],
        )
        self.assertFalse(finding.requires_legal_review)

    def test_negative_finding_is_explicitly_limited_to_article_6_1(
        self,
    ) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.NO
        facts.product_regulation.ai_is_safety_component = TriState.NO

        finding = self._run_one(facts)

        combined_text = f"{finding.title} {finding.summary}".casefold()
        self.assertIn("article 6(1)", combined_text)
        self.assertIn("does not exclude article 6(2)", combined_text)
        self.assertIn("another annex iii category", combined_text)
        self.assertNotIn("not high-risk", combined_text)
        self.assertNotIn("not high risk", combined_text)
        self.assertNotIn("compliant", combined_text)
        self.assertNotIn("no further", combined_text)

    def test_annex_i_confirmation_no_is_negative(self) -> None:
        facts = self._positive_facts()
        facts.product_regulation.annex_i_instrument_confirmed = TriState.NO
        facts.product_regulation.third_party_conformity_required = (
            TriState.UNKNOWN
        )

        finding = self._run_one(facts)

        self.assertEqual(finding.status, FindingStatus.DOES_NOT_APPLY)
        self.assertEqual(
            finding.reason_codes,
            ["ANNEX_I_COVERAGE_NOT_CONFIRMED"],
        )

    def test_no_third_party_conformity_requirement_is_negative(self) -> None:
        facts = self._positive_facts()
        facts.product_regulation.third_party_conformity_required = TriState.NO

        finding = self._run_one(facts)

        self.assertEqual(finding.status, FindingStatus.DOES_NOT_APPLY)
        self.assertEqual(
            finding.reason_codes,
            ["NO_THIRD_PARTY_CONFORMITY_REQUIREMENT"],
        )

    def test_regulated_product_does_not_replace_product_relationship(self) -> None:
        facts = self._positive_facts()
        facts.product_regulation.ai_is_product = TriState.NO
        facts.product_regulation.ai_is_safety_component = TriState.NO

        finding = self._run_one(facts)

        self.assertEqual(finding.status, FindingStatus.DOES_NOT_APPLY)
        self.assertEqual(
            finding.reason_codes,
            ["NEITHER_AI_PRODUCT_NOR_SAFETY_COMPONENT"],
        )

    def test_data_act_facts_do_not_infer_article_6_1(self) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.NO
        facts.product_regulation.ai_is_safety_component = TriState.NO
        facts.data_act.connected_product = TriState.YES
        facts.data_act.related_service = TriState.YES
        facts.data_act.data_generated = TriState.YES

        finding = self._run_one(facts)

        self.assertEqual(finding.status, FindingStatus.DOES_NOT_APPLY)
        self.assertEqual(
            finding.reason_codes,
            ["NEITHER_AI_PRODUCT_NOR_SAFETY_COMPONENT"],
        )

    def test_both_unknown_relationship_facts_are_missing(self) -> None:
        result = self.engine.run(AssessmentFacts())

        self.assertEqual(result.findings, [])
        self.assertEqual(result.executed_rule_ids, [])
        self.assertEqual(result.failures, [])
        missing = result.missing_fact_requirements[0].missing_facts
        self.assertEqual(
            [item.fact_path for item in missing],
            [
                "product_regulation.ai_is_product",
                "product_regulation.ai_is_safety_component",
            ],
        )
        self.assertTrue(
            all(item.reason is MissingFactReason.UNKNOWN for item in missing)
        )

    def test_unresolved_requirement_order_and_serialization_are_stable(
        self,
    ) -> None:
        facts = AssessmentFacts()
        validator = FactRequirementValidator()

        first = validator.validate(self.rule, facts)
        second = validator.validate(self.rule, facts)

        expected_paths = [
            "product_regulation.ai_is_product",
            "product_regulation.ai_is_safety_component",
        ]
        self.assertEqual(first.required_fact_paths, expected_paths)
        self.assertEqual(
            [item.fact_path for item in first.missing_facts],
            expected_paths,
        )
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_requirement_evaluation_does_not_mutate_facts(self) -> None:
        facts = self._positive_facts()
        facts.product_regulation.ai_is_safety_component = TriState.UNKNOWN
        facts.product_regulation.third_party_conformity_required = (
            TriState.UNKNOWN
        )
        before = facts.to_dict()

        requirement = FactRequirementValidator().validate(self.rule, facts)

        self.assertEqual(facts.to_dict(), before)
        self.assertEqual(
            requirement.required_fact_paths,
            ["product_regulation.third_party_conformity_required"],
        )

    def test_one_no_and_one_unknown_requires_only_unresolved_or_branch(
        self,
    ) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.NO

        result = self.engine.run(facts)

        missing = result.missing_fact_requirements[0].missing_facts
        self.assertEqual(
            [item.fact_path for item in missing],
            ["product_regulation.ai_is_safety_component"],
        )

    def test_satisfied_relationship_with_missing_instrument_is_reported(
        self,
    ) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.YES

        result = self.engine.run(facts)

        missing = result.missing_fact_requirements[0].missing_facts
        self.assertEqual(
            [item.fact_path for item in missing],
            [
                "product_regulation.annex_i_instrument",
                "product_regulation.annex_i_instrument_confirmed",
            ],
        )

    def test_selected_instrument_with_unknown_confirmation_is_missing(
        self,
    ) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.YES
        facts.product_regulation.annex_i_instrument = MACHINERY_ID

        result = self.engine.run(facts)

        missing = result.missing_fact_requirements[0].missing_facts
        self.assertEqual(
            [item.fact_path for item in missing],
            ["product_regulation.annex_i_instrument_confirmed"],
        )

    def test_unknown_third_party_fact_is_missing_after_confirmed_coverage(
        self,
    ) -> None:
        facts = self._positive_facts()
        facts.product_regulation.third_party_conformity_required = (
            TriState.UNKNOWN
        )

        result = self.engine.run(facts)

        missing = result.missing_fact_requirements[0].missing_facts
        self.assertEqual(
            [item.fact_path for item in missing],
            ["product_regulation.third_party_conformity_required"],
        )

    def test_or_short_circuit_never_requests_unused_unknown_branch(self) -> None:
        facts = self._positive_facts()
        facts.product_regulation.ai_is_safety_component = TriState.UNKNOWN
        facts.product_regulation.third_party_conformity_required = (
            TriState.UNKNOWN
        )

        result = self.engine.run(facts)

        missing = result.missing_fact_requirements[0].missing_facts
        self.assertEqual(
            result.missing_fact_requirements[0].required_fact_paths,
            ["product_regulation.third_party_conformity_required"],
        )
        self.assertEqual(
            [item.fact_path for item in missing],
            ["product_regulation.third_party_conformity_required"],
        )

    def test_confirmation_yes_without_instrument_is_undetermined(self) -> None:
        facts = AssessmentFacts()
        facts.product_regulation.ai_is_product = TriState.YES
        facts.product_regulation.annex_i_instrument_confirmed = TriState.YES
        facts.product_regulation.third_party_conformity_required = TriState.YES

        finding = self._run_one(facts)

        self.assertEqual(finding.status, FindingStatus.UNDETERMINED)
        self.assertEqual(
            finding.reason_codes,
            [
                "ANNEX_I_INSTRUMENT_MISSING",
                "INCONSISTENT_PRODUCT_REGULATION_FACTS",
            ],
        )
        self.assertTrue(finding.requires_legal_review)

    def test_unknown_catalogue_id_is_undetermined(self) -> None:
        facts = self._positive_facts()
        facts.product_regulation.annex_i_instrument = "ANNEX_I_A_99_UNKNOWN"

        finding = self._run_one(facts)

        self.assertEqual(finding.status, FindingStatus.UNDETERMINED)
        self.assertEqual(
            finding.reason_codes,
            [
                "ANNEX_I_INSTRUMENT_INVALID",
                "INCONSISTENT_PRODUCT_REGULATION_FACTS",
            ],
        )
        self.assertEqual(
            [basis.citation for basis in finding.legal_basis],
            ["Article 6(1)(a)", "Article 6(1)(b)"],
        )
        self.assertFalse(
            any(
                basis.citation.startswith("Annex I")
                for basis in finding.legal_basis
            )
        )

    def test_confirmation_no_with_conformity_yes_is_undetermined(self) -> None:
        facts = self._positive_facts()
        facts.product_regulation.annex_i_instrument_confirmed = TriState.NO

        finding = self._run_one(facts)

        self.assertEqual(finding.status, FindingStatus.UNDETERMINED)
        self.assertEqual(
            finding.reason_codes,
            ["INCONSISTENT_PRODUCT_REGULATION_FACTS"],
        )

    def test_malformed_fact_type_is_captured_as_rule_failure(self) -> None:
        facts = self._positive_facts()
        facts.product_regulation.ai_is_product = "yes"  # type: ignore[assignment]

        result = self.engine.run(facts)

        self.assertEqual(result.findings, [])
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(
            result.failures[0].error_type,
            "InvalidProductRegulationFactsError",
        )
        self.assertEqual(
            result.failures[0].framework,
            RegulatoryFramework.EU_AI_ACT,
        )

    def test_malformed_namespace_is_captured_as_rule_failure(self) -> None:
        facts = AssessmentFacts()
        facts.product_regulation = {}  # type: ignore[assignment]

        result = self.engine.run(facts)

        self.assertEqual(result.findings, [])
        self.assertEqual(len(result.failures), 1)
        self.assertEqual(result.failures[0].error_type, "TypeError")
        self.assertEqual(
            result.failures[0].framework,
            RegulatoryFramework.EU_AI_ACT,
        )

    def test_malformed_serialized_fact_fails_safely(self) -> None:
        with self.assertRaises(ValueError):
            ProductRegulationFacts.from_dict(
                {"ai_is_product": "not-a-tristate"}
            )

    def test_product_path_uses_atomic_ai_act_legal_references(self) -> None:
        finding = self._run_one(self._positive_facts())

        self.assertEqual(
            [basis.citation for basis in finding.legal_basis],
            ["Article 6(1)(a)", "Article 6(1)(b)", MACHINERY_REFERENCE],
        )
        self.assertEqual(
            {basis.instrument for basis in finding.legal_basis},
            {"EU_AI_ACT"},
        )

    def test_reason_code_and_serialization_order_is_deterministic(self) -> None:
        finding = self._run_one(self._positive_facts())

        first = json.dumps(finding.to_dict(), sort_keys=True)
        second = json.dumps(finding.to_dict(), sort_keys=True)
        self.assertEqual(first, second)
        self.assertEqual(
            finding.reason_codes,
            [
                "AI_IS_PRODUCT",
                "ANNEX_I_COVERAGE_CONFIRMED",
                "THIRD_PARTY_CONFORMITY_REQUIRED",
            ],
        )

    def test_rule_metadata_and_presentation_keys_are_stable(self) -> None:
        self.assertEqual(
            self.rule.rule_id,
            "AI_ACT_HIGH_RISK_PRODUCT_SAFETY",
        )
        self.assertEqual(self.rule.version, "2026.1")
        self.assertEqual(self.rule.framework, RegulatoryFramework.EU_AI_ACT)
        self.assertEqual(
            self.rule.issue_code,
            "AIA_HIGH_RISK_ARTICLE_6_1_PRELIMINARY",
        )
        self.assertEqual(
            set(self.rule.presentation_keys),
            {
                "potentially_applies",
                "does_not_apply",
                "undetermined",
            },
        )

    def test_existing_rules_keep_their_results(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "Recruitment system ranking candidates"
        facts.use_context.materially_influences_decision = TriState.YES
        facts.data_protection.personal_data_processed = TriState.YES
        facts.data_protection.automated_individual_decision = TriState.YES
        facts.data_act.connected_product = TriState.YES
        facts.data_act.related_service = TriState.NO
        facts.data_act.data_generated = TriState.YES
        engine = AssessmentEngine(
            RuleRegistry(
                [
                    AIActHighRiskEmploymentRule(),
                    GDPRArticle22RelevanceRule(),
                    EUDataActRelevanceRule(),
                ]
            )
        )

        result = engine.run(facts)

        self.assertEqual(result.failures, [])
        self.assertEqual(
            [finding.status for finding in result.findings],
            [
                FindingStatus.POTENTIALLY_APPLIES,
                FindingStatus.POTENTIALLY_APPLIES,
                FindingStatus.POTENTIALLY_APPLIES,
            ],
        )
        self.assertEqual(
            [finding.rule_id for finding in result.findings],
            [
                "AI_ACT_HIGH_RISK_EMPLOYMENT",
                "GDPR_ARTICLE22_RELEVANCE",
                "EU_DATA_ACT_RELEVANCE",
            ],
        )

    def test_existing_rules_keep_static_requirement_behavior(self) -> None:
        facts = AssessmentFacts()
        validator = FactRequirementValidator()
        existing_rules = (
            AIActHighRiskEmploymentRule(),
            GDPRArticle22RelevanceRule(),
            EUDataActRelevanceRule(),
        )

        for rule in existing_rules:
            with self.subTest(rule=rule.rule_id):
                before = facts.to_dict()
                requirement = validator.validate(rule, facts)
                self.assertEqual(
                    requirement.required_fact_paths,
                    list(rule.required_fact_paths),
                )
                self.assertEqual(
                    [item.fact_path for item in requirement.missing_facts],
                    list(rule.required_fact_paths),
                )
                self.assertEqual(facts.to_dict(), before)

    def test_existing_fixtures_do_not_auto_trigger_new_rule(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        for fixture_name in (
            "recruitment_ai_case.json",
            "industrial_ai_case.json",
        ):
            with self.subTest(fixture=fixture_name):
                payload = json.loads(
                    (project_root / "tests" / "fixtures" / fixture_name)
                    .read_text(encoding="utf-8")
                )
                product_facts = ProductRegulationFacts.from_dict(
                    payload["facts"].get("product_regulation", {})
                )
                facts = AssessmentFacts(product_regulation=product_facts)

                result = self.engine.run(facts)

                self.assertEqual(result.findings, [])
                self.assertEqual(result.failures, [])
                self.assertEqual(len(result.missing_fact_requirements), 1)


if __name__ == "__main__":
    unittest.main()
