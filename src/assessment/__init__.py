"""Domain foundation for structured EU AI Act assessments."""

from src.assessment.case import (
    AssessmentCase,
    AssessmentCaseNotFoundError,
    AssessmentCaseSchemaMismatchError,
    AssessmentCaseService,
    DuplicateAssessmentCaseError,
)
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
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import AssessmentRun, AssessmentRunStatus, TriState
from src.assessment.questionnaire import (
    AnswerType,
    DuplicateQuestionError,
    DuplicateQuestionFactPathError,
    InvalidQuestionFactPathError,
    Question,
    QuestionnaireEngine,
    QuestionnairePlan,
    QuestionNotFoundError,
    QuestionOption,
    QuestionRegistry,
)
from src.assessment.requirements import (
    FactRequirementValidator,
    MissingFact,
    MissingFactReason,
    RuleRequirementResult,
)
from src.assessment.report import (
    AssessmentReport,
    FrameworkFindings,
    MissingInformation,
    ReportBuildError,
    ReportBuilder,
    RuleVersionMetadata,
)
from src.assessment.results import AssessmentResult, RuleExecutionFailure
from src.assessment.workflow import (
    AssessmentRunNotFoundError,
    AssessmentWorkflowService,
)

__all__ = [
    "AssessmentCase",
    "AssessmentCaseNotFoundError",
    "AssessmentCaseSchemaMismatchError",
    "AssessmentCaseService",
    "AssessmentEngine",
    "AssessmentFacts",
    "AssessmentResult",
    "AssessmentReport",
    "AssessmentRun",
    "AssessmentRunNotFoundError",
    "AssessmentRunStatus",
    "AssessmentWorkflowService",
    "AnswerType",
    "AuthorityLevel",
    "DuplicateEvidenceError",
    "DuplicateAssessmentCaseError",
    "DuplicateQuestionError",
    "DuplicateQuestionFactPathError",
    "Evidence",
    "EvidenceService",
    "EvidenceServiceResult",
    "FactRequirementValidator",
    "Finding",
    "FindingCategory",
    "FindingEvidenceBinding",
    "FindingStatus",
    "FrameworkFindings",
    "InMemoryEvidenceService",
    "InvalidQuestionFactPathError",
    "LegalEvidenceRetriever",
    "LegalBasis",
    "MissingFact",
    "MissingFactReason",
    "MissingInformation",
    "Question",
    "QuestionnaireEngine",
    "QuestionnairePlan",
    "QuestionNotFoundError",
    "QuestionOption",
    "QuestionRegistry",
    "RegulatoryFramework",
    "ReportBuildError",
    "ReportBuilder",
    "RuleExecutionFailure",
    "RuleOutputError",
    "RuleRequirementResult",
    "RuleVersionMetadata",
    "TriState",
    "VectorStoreFormatError",
    "VectorStoreJSONEvidenceRetriever",
]
