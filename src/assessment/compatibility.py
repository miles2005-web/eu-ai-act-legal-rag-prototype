"""Explicit, inactive compatibility boundaries for versioned assessment data."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.assessment.facts import AssessmentFacts
from src.assessment.report.models import AssessmentReport


class AssessmentFactsCompatibilityAdapter:
    """Read v2/v3 facts and explicitly derive a v3 draft when requested."""

    VERSION = "1.0.0"

    @staticmethod
    def read(payload: dict[str, Any]) -> AssessmentFacts:
        return AssessmentFacts.from_dict(payload)

    @staticmethod
    def derive_v3(source: AssessmentFacts) -> AssessmentFacts:
        """Return a new v3 draft; never mutate or manufacture scoped entities."""

        if not isinstance(source, AssessmentFacts):
            raise TypeError("source must be AssessmentFacts")
        if source.schema_version == AssessmentFacts.V3_SCHEMA_VERSION:
            return deepcopy(source).make_editable()
        if source.schema_version != AssessmentFacts.V2_SCHEMA_VERSION:
            raise ValueError("only AssessmentFacts 2.0.0 can be adapted")
        payload = source.to_dict()
        payload["schema_version"] = AssessmentFacts.V3_SCHEMA_VERSION
        payload["source_schema_version"] = source.schema_version
        return AssessmentFacts.from_dict(payload).make_editable()


class AssessmentReportCompatibilityReader:
    """Read Report 1.0.0 or 2.0.0 without converting the source snapshot."""

    VERSION = "1.0.0"

    @staticmethod
    def read(payload: dict[str, Any]) -> AssessmentReport:
        return AssessmentReport.from_dict(payload)
