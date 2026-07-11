"""Evidence domain models for structured legal assessments."""

from src.assessment.evidence.models import (
    AuthorityLevel,
    Evidence,
    FindingEvidenceBinding,
)
from src.assessment.evidence.retriever import (
    LegalEvidenceRetriever,
    VectorStoreFormatError,
    VectorStoreJSONEvidenceRetriever,
)
from src.assessment.evidence.service import (
    DuplicateEvidenceError,
    EvidenceService,
    EvidenceServiceResult,
    InMemoryEvidenceService,
)

__all__ = [
    "AuthorityLevel",
    "DuplicateEvidenceError",
    "Evidence",
    "EvidenceService",
    "EvidenceServiceResult",
    "FindingEvidenceBinding",
    "InMemoryEvidenceService",
    "LegalEvidenceRetriever",
    "VectorStoreFormatError",
    "VectorStoreJSONEvidenceRetriever",
]
