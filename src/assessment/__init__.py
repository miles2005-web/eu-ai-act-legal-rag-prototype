"""Domain foundation for structured EU AI Act assessments."""

from src.assessment.engine import AssessmentEngine, RuleOutputError
from src.assessment.evidence import (
    AuthorityLevel,
    DuplicateEvidenceError,
    Evidence,
    EvidenceService,
    EvidenceServiceResult,
    FindingEvidenceBinding,
    InMemoryEvidenceService,
    LegalEvidenceRetriever,
    VectorStoreFormatError,
    VectorStoreJSONEvidenceRetriever,
)
from src.assessment.facts import AssessmentFacts
from src.assessment.findings import Finding, FindingCategory, FindingStatus, LegalBasis
from src.assessment.models import AssessmentRun, AssessmentRunStatus, TriState
from src.assessment.requirements import (
    FactRequirementValidator,
    MissingFact,
    MissingFactReason,
    RuleRequirementResult,
)
from src.assessment.results import AssessmentResult, RuleExecutionFailure

__all__ = [
    "AssessmentEngine",
    "AssessmentFacts",
    "AssessmentResult",
    "AssessmentRun",
    "AssessmentRunStatus",
    "AuthorityLevel",
    "DuplicateEvidenceError",
    "Evidence",
    "EvidenceService",
    "EvidenceServiceResult",
    "FactRequirementValidator",
    "Finding",
    "FindingCategory",
    "FindingEvidenceBinding",
    "FindingStatus",
    "InMemoryEvidenceService",
    "LegalEvidenceRetriever",
    "LegalBasis",
    "MissingFact",
    "MissingFactReason",
    "RuleExecutionFailure",
    "RuleOutputError",
    "RuleRequirementResult",
    "TriState",
    "VectorStoreFormatError",
    "VectorStoreJSONEvidenceRetriever",
]
