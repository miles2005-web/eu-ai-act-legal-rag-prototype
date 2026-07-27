"""Tests for deterministic missing-fact questionnaire planning."""

from __future__ import annotations

import json
import unittest

from src.assessment.questionnaire import (
    AnswerType,
    DuplicateQuestionError,
    InvalidQuestionFactPathError,
    Question,
    QuestionnaireEngine,
    QuestionRegistry,
    QuestionOption,
)
from src.assessment.requirements import (
    MissingFact,
    MissingFactReason,
    RuleRequirementResult,
)


class QuestionnaireFoundationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.material_question = Question(
            question_id="EMPLOYMENT-MATERIAL-INFLUENCE",
            text="Does the AI output materially influence an employment decision?",
            fact_path="use_context.materially_influences_decision",
            answer_type=AnswerType.TRI_STATE,
            required=True,
            legal_relevance=("Article 6",),
        )
        self.domain_question = Question(
            question_id="USE-DOMAIN",
            text="In which domain is the AI system used?",
            fact_path="use_context.domain",
            answer_type=AnswerType.SINGLE_CHOICE,
            options=(
                QuestionOption(value="employment", label="Employment"),
                QuestionOption(value="other", label="Other"),
            ),
            required=True,
            legal_relevance=("Annex III",),
        )
        self.registry = QuestionRegistry(
            [self.material_question, self.domain_question]
        )

    def test_registry_retrieval_and_order_are_deterministic(self) -> None:
        self.assertIs(
            self.registry.get("EMPLOYMENT-MATERIAL-INFLUENCE"),
            self.material_question,
        )
        self.assertEqual(
            [question.question_id for question in self.registry.all()],
            ["EMPLOYMENT-MATERIAL-INFLUENCE", "USE-DOMAIN"],
        )
        with self.assertRaises(DuplicateQuestionError):
            self.registry.register(self.material_question)

    def test_engine_deduplicates_and_uses_registry_order(self) -> None:
        requirements = [
            RuleRequirementResult(
                rule_id="RULE-ONE",
                rule_version="1.0",
                required_fact_paths=[
                    "use_context.domain",
                    "use_context.materially_influences_decision",
                ],
                missing_facts=[
                    MissingFact(
                        fact_path="use_context.domain",
                        reason=MissingFactReason.UNKNOWN,
                    ),
                    MissingFact(
                        fact_path="use_context.materially_influences_decision",
                        reason=MissingFactReason.UNKNOWN,
                    ),
                ],
            ),
            RuleRequirementResult(
                rule_id="RULE-TWO",
                rule_version="1.0",
                required_fact_paths=["use_context.domain", "system.name"],
                missing_facts=[
                    MissingFact(
                        fact_path="use_context.domain",
                        reason=MissingFactReason.UNKNOWN,
                    ),
                    MissingFact(
                        fact_path="system.name",
                        reason=MissingFactReason.NOT_PROVIDED,
                    ),
                    MissingFact(
                        fact_path="use_context.not_a_fact",
                        reason=MissingFactReason.PATH_NOT_FOUND,
                    ),
                ],
            ),
        ]

        plan = QuestionnaireEngine(self.registry).build(requirements)

        self.assertEqual(
            [question.question_id for question in plan.questions],
            ["EMPLOYMENT-MATERIAL-INFLUENCE", "USE-DOMAIN"],
        )
        self.assertEqual(
            plan.unmapped_fact_paths,
            ["system.name", "use_context.not_a_fact"],
        )
        json.dumps(plan.to_dict())

    def test_registry_rejects_nonexistent_fact_path(self) -> None:
        invalid_question = Question(
            question_id="INVALID",
            text="Invalid question",
            fact_path="use_context.not_a_fact",
            answer_type=AnswerType.TEXT,
        )

        with self.assertRaises(InvalidQuestionFactPathError):
            self.registry.register(invalid_question)


if __name__ == "__main__":
    unittest.main()
