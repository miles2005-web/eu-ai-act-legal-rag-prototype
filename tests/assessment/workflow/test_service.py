"""Tests for assessment workflow orchestration."""

from __future__ import annotations

import unittest

from src.assessment import (
    AssessmentCaseService,
    AssessmentEngine,
    AssessmentFacts,
    AssessmentRunStatus,
    AssessmentWorkflowService,
    AuthorityLevel,
    Evidence,
    FindingStatus,
    InMemoryEvidenceService,
    ReportBuilder,
    TriState,
)
from src.assessment.facts import UseDomain
from src.assessment.rules import AIActHighRiskEmploymentRule, RuleRegistry


class AssessmentWorkflowServiceTests(unittest.TestCase):
    def _build_workflow(
        self,
        *,
        facts: AssessmentFacts,
        evidence: list[Evidence] | None = None,
        case_id: str = "case-workflow",
    ) -> tuple[AssessmentWorkflowService, AssessmentCaseService]:
        case_service = AssessmentCaseService()
        case_service.create_case(
            "Workflow case",
            facts=facts,
            case_id=case_id,
        )
        workflow = AssessmentWorkflowService(
            case_service=case_service,
            assessment_engine=AssessmentEngine(
                RuleRegistry([AIActHighRiskEmploymentRule()])
            ),
            evidence_service=InMemoryEvidenceService(evidence or []),
            report_builder=ReportBuilder(),
        )
        return workflow, case_service

    @staticmethod
    def _complete_employment_facts() -> AssessmentFacts:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "Recruitment system ranking candidates"
        facts.use_context.materially_influences_decision = TriState.YES
        return facts

    @staticmethod
    def _employment_evidence() -> list[Evidence]:
        return [
            Evidence(
                evidence_id="article-6-evidence",
                legal_source="EU_AI_ACT",
                citation="Article 6",
                excerpt="Article 6 supporting excerpt.",
                document_version="2024/1689",
                authority_level=AuthorityLevel.BINDING_LEGISLATION,
            ),
            Evidence(
                evidence_id="annex-iii-evidence",
                legal_source="EU_AI_ACT",
                citation="Annex III point 4(a)",
                excerpt="Annex III point 4(a) supporting excerpt.",
                document_version="2024/1689",
                authority_level=AuthorityLevel.BINDING_LEGISLATION,
            ),
        ]

    def test_successful_workflow_returns_report_and_completed_run(self) -> None:
        workflow, case_service = self._build_workflow(
            facts=self._complete_employment_facts(),
            evidence=self._employment_evidence(),
        )

        report = workflow.run("case-workflow")
        runs = workflow.runs_for_case("case-workflow")

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, AssessmentRunStatus.COMPLETED)
        self.assertEqual(report.assessment_run_reference, runs[0].id)
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(
            report.findings[0].status,
            FindingStatus.POTENTIALLY_APPLIES,
        )
        self.assertEqual(report.findings[0].assessment_run_id, runs[0].id)

        updated_facts = self._complete_employment_facts()
        updated_facts.use_context.task = "Changed after assessment"
        case_service.update_facts("case-workflow", updated_facts)
        self.assertEqual(
            workflow.get_run(runs[0].id).facts_snapshot.use_context.task,
            "Recruitment system ranking candidates",
        )

    def test_missing_facts_flow_returns_report_without_finding(self) -> None:
        facts = AssessmentFacts()
        facts.use_context.domain = UseDomain.EMPLOYMENT
        facts.use_context.task = "Recruitment system ranking candidates"
        workflow, _ = self._build_workflow(facts=facts)

        report = workflow.run("case-workflow")
        run = workflow.runs_for_case("case-workflow")[0]

        self.assertEqual(run.status, AssessmentRunStatus.COMPLETED)
        self.assertEqual(report.assessment_run_reference, run.id)
        self.assertEqual(report.findings, [])
        self.assertEqual(len(report.missing_information), 1)
        self.assertEqual(
            report.missing_information[0].fact_path,
            "use_context.materially_influences_decision",
        )

    def test_evidence_binding_flow_preserves_finding_relationship(self) -> None:
        workflow, _ = self._build_workflow(
            facts=self._complete_employment_facts(),
            evidence=self._employment_evidence(),
        )

        report = workflow.run("case-workflow")

        self.assertEqual(
            [evidence.evidence_id for evidence in report.evidence],
            ["article-6-evidence", "annex-iii-evidence"],
        )
        self.assertEqual(len(report.evidence_bindings), 1)
        self.assertEqual(
            report.evidence_bindings[0].finding_id,
            report.findings[0].finding_id,
        )
        self.assertEqual(
            report.evidence_bindings[0].evidence_refs,
            ["article-6-evidence", "annex-iii-evidence"],
        )


if __name__ == "__main__":
    unittest.main()
