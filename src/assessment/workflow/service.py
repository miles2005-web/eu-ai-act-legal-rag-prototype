"""Application service coordinating the complete assessment workflow."""

from __future__ import annotations

from copy import deepcopy

from src.assessment.case import AssessmentCaseService
from src.assessment.engine import AssessmentEngine
from src.assessment.evidence import EvidenceService
from src.assessment.models import AssessmentRun, AssessmentRunStatus, utc_now
from src.assessment.report import AssessmentReport, ReportBuilder
from src.assessment.results import AssessmentResult


class AssessmentRunNotFoundError(KeyError):
    """Raised when an orchestration run is absent from in-memory history."""


class AssessmentWorkflowService:
    """Coordinate case, reasoning, evidence, and reporting domain services."""

    def __init__(
        self,
        *,
        case_service: AssessmentCaseService,
        assessment_engine: AssessmentEngine,
        evidence_service: EvidenceService,
        report_builder: ReportBuilder,
    ) -> None:
        if not isinstance(case_service, AssessmentCaseService):
            raise TypeError("case_service must be an AssessmentCaseService")
        if not isinstance(assessment_engine, AssessmentEngine):
            raise TypeError("assessment_engine must be an AssessmentEngine")
        if not isinstance(evidence_service, EvidenceService):
            raise TypeError("evidence_service must implement EvidenceService")
        if not isinstance(report_builder, ReportBuilder):
            raise TypeError("report_builder must be a ReportBuilder")

        self._case_service = case_service
        self._assessment_engine = assessment_engine
        self._evidence_service = evidence_service
        self._report_builder = report_builder
        self._runs_by_id: dict[str, AssessmentRun] = {}
        self._run_ids_by_case: dict[str, list[str]] = {}

    def run(self, case_id: str) -> AssessmentReport:
        """Execute one traceable workflow for the case's current fact snapshot."""

        assessment_case = self._case_service.get_case(case_id)
        assessment_run = AssessmentRun(
            case_id=assessment_case.case_id,
            facts_snapshot=assessment_case.current_facts,
            ruleset_version=self._assessment_engine.engine_version,
            status=AssessmentRunStatus.RUNNING,
        )
        assessment_result: AssessmentResult | None = None

        try:
            assessment_result = self._assessment_engine.run(
                assessment_run.facts_snapshot
            )
            assessment_result.assessment_run_id = assessment_run.id
            for finding in assessment_result.findings:
                finding.assessment_run_id = assessment_run.id

            assessment_run.findings = deepcopy(assessment_result.findings)
            evidence_result = self._evidence_service.resolve(
                assessment_result.findings
            )
            report = self._report_builder.build(
                assessment_result,
                evidence_result,
            )

            assessment_run.status = AssessmentRunStatus.COMPLETED
            assessment_run.completed_at = assessment_result.timestamp
            self._store_run(assessment_run)
            return report
        except Exception as exc:
            assessment_run.status = AssessmentRunStatus.FAILED
            assessment_run.completed_at = (
                assessment_result.timestamp
                if assessment_result is not None
                else utc_now()
            )
            assessment_run.error_message = str(exc) or type(exc).__name__
            self._store_run(assessment_run)
            raise

    def get_run(self, run_id: str) -> AssessmentRun:
        """Return an isolated historical run snapshot."""

        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        try:
            return deepcopy(self._runs_by_id[run_id])
        except KeyError as exc:
            raise AssessmentRunNotFoundError(
                f"Assessment run {run_id!r} was not found"
            ) from exc

    def runs_for_case(self, case_id: str) -> tuple[AssessmentRun, ...]:
        """Return case runs in workflow execution order."""

        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        return tuple(
            deepcopy(self._runs_by_id[run_id])
            for run_id in self._run_ids_by_case.get(case_id, ())
        )

    def _store_run(self, assessment_run: AssessmentRun) -> None:
        if assessment_run.id in self._runs_by_id:
            raise ValueError(
                f"Assessment run {assessment_run.id!r} is already stored"
            )
        self._runs_by_id[assessment_run.id] = deepcopy(assessment_run)
        self._run_ids_by_case.setdefault(assessment_run.case_id, []).append(
            assessment_run.id
        )

