"""Provider-neutral evidence resolution for assessment findings."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field

from src.assessment.evidence.models import Evidence, FindingEvidenceBinding
from src.assessment.findings import Finding, LegalBasis
from src.assessment.models import SerializableModel


class DuplicateEvidenceError(ValueError):
    """Raised when an in-memory provider receives a duplicate evidence ID."""


@dataclass(slots=True)
class EvidenceServiceResult(SerializableModel):
    """Resolved evidence records and their finding relationships."""

    evidence: list[Evidence] = field(default_factory=list)
    bindings: list[FindingEvidenceBinding] = field(default_factory=list)


class EvidenceService(ABC):
    """Contract for resolving legal-basis references attached to Findings."""

    @abstractmethod
    def resolve(self, findings: Iterable[Finding]) -> EvidenceServiceResult:
        """Resolve supporting evidence without changing the input findings."""

        raise NotImplementedError


class InMemoryEvidenceService(EvidenceService):
    """Deterministic mock provider backed by pre-supplied Evidence objects.

    Records are matched to ``LegalBasis`` values by normalized legal source and
    citation. This provider performs no semantic, keyword, or vector search.
    """

    def __init__(self, evidence: Iterable[Evidence] | None = None) -> None:
        self._evidence_by_id: dict[str, Evidence] = {}
        self._ids_by_reference: dict[tuple[str, str], list[str]] = {}
        if evidence is not None:
            self.add_many(evidence)

    def add(self, evidence: Evidence) -> Evidence:
        """Add one mock evidence record and preserve provider isolation."""

        if not isinstance(evidence, Evidence):
            raise TypeError("evidence must be an Evidence instance")
        if evidence.evidence_id in self._evidence_by_id:
            raise DuplicateEvidenceError(
                f"Evidence ID {evidence.evidence_id!r} is already registered"
            )

        stored = deepcopy(evidence)
        self._evidence_by_id[stored.evidence_id] = stored
        key = self._evidence_key(stored.legal_source, stored.citation)
        self._ids_by_reference.setdefault(key, []).append(stored.evidence_id)
        return deepcopy(stored)

    def add_many(self, evidence: Iterable[Evidence]) -> None:
        """Add mock evidence records in iterable order."""

        for item in evidence:
            self.add(item)

    def resolve(self, findings: Iterable[Finding]) -> EvidenceServiceResult:
        """Resolve evidence in finding and legal-basis declaration order."""

        finding_snapshot = tuple(findings)
        if any(not isinstance(finding, Finding) for finding in finding_snapshot):
            raise TypeError("findings must contain only Finding instances")

        resolved_by_id: dict[str, Evidence] = {}
        bindings: list[FindingEvidenceBinding] = []

        for finding in finding_snapshot:
            finding_evidence_ids: list[str] = []
            for legal_basis in finding.legal_basis:
                for evidence_id in self._matching_ids(legal_basis):
                    if evidence_id not in finding_evidence_ids:
                        finding_evidence_ids.append(evidence_id)
                    if evidence_id not in resolved_by_id:
                        resolved_by_id[evidence_id] = deepcopy(
                            self._evidence_by_id[evidence_id]
                        )

            if finding_evidence_ids:
                bindings.append(
                    FindingEvidenceBinding(
                        finding_id=finding.finding_id,
                        evidence_refs=finding_evidence_ids,
                    )
                )

        return EvidenceServiceResult(
            evidence=list(resolved_by_id.values()),
            bindings=bindings,
        )

    def __len__(self) -> int:
        return len(self._evidence_by_id)

    def _matching_ids(self, legal_basis: LegalBasis) -> tuple[str, ...]:
        if not isinstance(legal_basis, LegalBasis):
            raise TypeError("finding legal_basis must contain LegalBasis values")
        key = self._evidence_key(legal_basis.instrument, legal_basis.citation)
        return tuple(self._ids_by_reference.get(key, ()))

    @staticmethod
    def _evidence_key(legal_source: str, citation: str) -> tuple[str, str]:
        return legal_source.strip().casefold(), citation.strip().casefold()

