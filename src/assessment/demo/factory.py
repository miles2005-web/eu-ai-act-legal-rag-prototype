"""Dependency wiring for the reusable EU AI Act assessment workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.assessment.case import AssessmentCaseService
from src.assessment.engine import AssessmentEngine
from src.assessment.evidence import (
    InMemoryEvidenceService,
    VectorStoreJSONEvidenceRetriever,
)
from src.assessment.report import ReportBuilder
from src.assessment.rules import AIActHighRiskEmploymentRule, RuleRegistry
from src.assessment.workflow import AssessmentWorkflowService


DEFAULT_VECTOR_STORE_PATH = Path(__file__).resolve().parents[3] / "vector_store.json"


@dataclass(frozen=True, slots=True)
class AssessmentWorkflowBundle:
    """Configured services exposed to application entry points."""

    workflow: AssessmentWorkflowService
    case_service: AssessmentCaseService
    engine: AssessmentEngine
    rule_registry: RuleRegistry
    evidence_retriever: VectorStoreJSONEvidenceRetriever
    evidence_service: InMemoryEvidenceService
    report_builder: ReportBuilder


def create_assessment_workflow(
    *,
    vector_store_path: str | Path = DEFAULT_VECTOR_STORE_PATH,
    evidence_limit: int = 5,
) -> AssessmentWorkflowBundle:
    """Create the shared demonstration assessment configuration.

    The factory is a composition root only: it registers existing rules,
    resolves their supporting corpus records, and wires existing services.
    It contains no legal evaluation or workflow orchestration logic.
    """

    if not isinstance(evidence_limit, int) or isinstance(evidence_limit, bool):
        raise TypeError("evidence_limit must be an integer")
    if evidence_limit <= 0:
        raise ValueError("evidence_limit must be greater than zero")

    case_service = AssessmentCaseService()
    rule_registry = RuleRegistry([AIActHighRiskEmploymentRule()])
    engine = AssessmentEngine(rule_registry)
    evidence_retriever = VectorStoreJSONEvidenceRetriever(vector_store_path)

    evidence_by_id = {}
    for rule in rule_registry:
        for legal_basis in rule.legal_basis:
            for evidence in evidence_retriever.retrieve(
                legal_basis.instrument,
                legal_basis.citation,
                limit=evidence_limit,
            ):
                evidence_by_id.setdefault(evidence.evidence_id, evidence)

    evidence_service = InMemoryEvidenceService(evidence_by_id.values())
    report_builder = ReportBuilder()
    workflow = AssessmentWorkflowService(
        case_service=case_service,
        assessment_engine=engine,
        evidence_service=evidence_service,
        report_builder=report_builder,
    )

    return AssessmentWorkflowBundle(
        workflow=workflow,
        case_service=case_service,
        engine=engine,
        rule_registry=rule_registry,
        evidence_retriever=evidence_retriever,
        evidence_service=evidence_service,
        report_builder=report_builder,
    )
