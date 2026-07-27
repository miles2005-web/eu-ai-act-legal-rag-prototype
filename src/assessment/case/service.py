"""In-memory lifecycle operations for assessment cases."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime

from src.assessment.case.models import AssessmentCase
from src.assessment.facts import AssessmentFacts
from src.assessment.models import new_identifier, utc_now


class AssessmentCaseNotFoundError(KeyError):
    """Raised when a case ID is absent from the in-memory service."""


class DuplicateAssessmentCaseError(ValueError):
    """Raised when a requested case ID already exists."""


class AssessmentCaseSchemaMismatchError(ValueError):
    """Raised when facts use a different schema from the existing case."""


class AssessmentCaseService:
    """Isolated in-memory storage for current case facts only."""

    def __init__(self, *, clock: Callable[[], datetime] = utc_now) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._cases: dict[str, AssessmentCase] = {}

    def create_case(
        self,
        name: str,
        *,
        description: str | None = None,
        facts: AssessmentFacts | None = None,
        case_id: str | None = None,
    ) -> AssessmentCase:
        """Create and return an isolated assessment case."""

        current_facts = facts if facts is not None else AssessmentFacts()
        if not isinstance(current_facts, AssessmentFacts):
            raise TypeError("facts must be an AssessmentFacts instance or None")
        current_facts.validate_schema_consistency()
        stable_case_id = case_id if case_id is not None else new_identifier()
        if stable_case_id in self._cases:
            raise DuplicateAssessmentCaseError(
                f"Case ID {stable_case_id!r} already exists"
            )

        now = self._clock()
        assessment_case = AssessmentCase(
            case_id=stable_case_id,
            name=name,
            description=description,
            current_facts=current_facts,
            created_at=now,
            updated_at=now,
            schema_version=current_facts.schema_version,
        )
        self._cases[stable_case_id] = deepcopy(assessment_case)
        return deepcopy(assessment_case)

    def update_facts(
        self,
        case_id: str,
        facts: AssessmentFacts,
    ) -> AssessmentCase:
        """Replace current case facts without changing historical snapshots."""

        if not isinstance(facts, AssessmentFacts):
            raise TypeError("facts must be an AssessmentFacts instance")
        facts.validate_schema_consistency()
        stored_case = self._get_stored_case(case_id)
        if facts.schema_version != stored_case.schema_version:
            raise AssessmentCaseSchemaMismatchError(
                f"Case schema {stored_case.schema_version!r} does not match "
                f"facts schema {facts.schema_version!r}"
            )
        if facts.schema_version == AssessmentFacts.V3_SCHEMA_VERSION:
            previous_ids = stored_case.current_facts.active_entity_ids()
            current_ids = facts.active_entity_ids()
            removed_ids = previous_ids.difference(current_ids)
            retired_ids = set(facts.retired_entity_ids)
            previous_retired_ids = set(
                stored_case.current_facts.retired_entity_ids
            )
            if not previous_retired_ids.issubset(retired_ids):
                raise ValueError("retired v3 entity IDs cannot be removed")
            if not removed_ids.issubset(retired_ids):
                raise ValueError(
                    "removed v3 entity IDs must be recorded in "
                    f"retired_entity_ids: {sorted(removed_ids - retired_ids)!r}"
                )

        updated_at = self._clock()
        if not isinstance(updated_at, datetime) or updated_at.utcoffset() is None:
            raise TypeError("clock must return a timezone-aware datetime")
        if updated_at < stored_case.updated_at:
            raise ValueError("clock returned a timestamp before the last update")

        stored_case.current_facts = deepcopy(facts)
        stored_case.updated_at = updated_at
        return deepcopy(stored_case)

    def get_case(self, case_id: str) -> AssessmentCase:
        """Retrieve an isolated copy of a case by stable ID."""

        return deepcopy(self._get_stored_case(case_id))

    def __len__(self) -> int:
        return len(self._cases)

    def _get_stored_case(self, case_id: str) -> AssessmentCase:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        try:
            return self._cases[case_id]
        except KeyError as exc:
            raise AssessmentCaseNotFoundError(
                f"Case ID {case_id!r} was not found"
            ) from exc
