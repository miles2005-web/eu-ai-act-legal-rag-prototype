"""Evidence domain models for structured legal assessments."""

from src.assessment.evidence.catalog import (
    LegalSource,
    LegalSourceCatalog,
    LegalSourceCatalogError,
    load_legal_source_catalog,
)
from src.assessment.evidence.citations import expand_citation_reference
from src.assessment.evidence.corpus_metadata import (
    CORPUS_METADATA_SCHEMA_VERSION,
    CorpusMetadataV2,
    normalize_evidence_excerpt,
    normalized_excerpt_hash,
    stable_evidence_digest,
    stable_evidence_id,
)
from src.assessment.evidence.models import (
    AuthorityLevel,
    Evidence,
    FindingEvidenceBinding,
)
from src.assessment.evidence.retriever import (
    LegalEvidenceRetriever,
    MultiCorpusLegalEvidenceRetriever,
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
    "CORPUS_METADATA_SCHEMA_VERSION",
    "CorpusMetadataV2",
    "DuplicateEvidenceError",
    "Evidence",
    "EvidenceService",
    "EvidenceServiceResult",
    "FindingEvidenceBinding",
    "InMemoryEvidenceService",
    "LegalEvidenceRetriever",
    "MultiCorpusLegalEvidenceRetriever",
    "LegalSource",
    "LegalSourceCatalog",
    "LegalSourceCatalogError",
    "VectorStoreFormatError",
    "VectorStoreJSONEvidenceRetriever",
    "load_legal_source_catalog",
    "normalize_evidence_excerpt",
    "normalized_excerpt_hash",
    "stable_evidence_digest",
    "stable_evidence_id",
    "expand_citation_reference",
]
