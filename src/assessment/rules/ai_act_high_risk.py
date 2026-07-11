"""Preliminary EU AI Act high-risk screening for employment-related AI."""

from __future__ import annotations

import re

from src.assessment.facts import AssessmentFacts, UseDomain
from src.assessment.findings import (
    Finding,
    FindingCategory,
    FindingStatus,
    FindingTraceEntry,
    LegalBasis,
)
from src.assessment.models import TriState
from src.assessment.rules.base import AssessmentRule


class AIActHighRiskEmploymentRule(AssessmentRule):
    """Screen for a potential Annex III employment-related classification.

    This rule is deliberately preliminary. It does not evaluate every element,
    exception, or classification route under Article 6, and a positive result
    must not be treated as a definitive high-risk classification.
    """

    rule_id = "AI_ACT_HIGH_RISK_EMPLOYMENT"
    version = "2026.1"
    category = FindingCategory.HIGH_RISK
    required_fact_paths = (
        "use_context.domain",
        "use_context.task",
        "use_context.materially_influences_decision",
    )
    legal_basis = (
        LegalBasis(
            instrument="EU_AI_ACT",
            citation="Article 6",
            anchor="article:6",
        ),
        LegalBasis(
            instrument="EU_AI_ACT",
            citation="Annex III point 4(a)",
            anchor="annex:III:point:4a",
        ),
    )

    _PERSON_TARGET_TERMS = (
        "applicant",
        "candidate",
        "employee",
        "job seeker",
        "personnel",
        "staff",
        "worker",
    )

    def evaluate(self, facts: AssessmentFacts) -> Finding:
        """Return a cautious preliminary result for complete required facts."""

        employment_context = facts.use_context.domain is UseDomain.EMPLOYMENT
        matched_functions = self._matched_employment_functions(
            facts.use_context.task or ""
        )
        materially_influences = (
            facts.use_context.materially_influences_decision is TriState.YES
        )
        potentially_applies = (
            employment_context and bool(matched_functions) and materially_influences
        )

        trace = [
            FindingTraceEntry(
                description="Checked whether the use context is employment-related.",
                fact_refs=["use_context.domain"],
                result="employment_context" if employment_context else "not_employment_context",
            ),
            FindingTraceEntry(
                description=(
                    "Checked for recruitment, selection, evaluation, ranking, "
                    "or recommendation of candidates or workers."
                ),
                fact_refs=["use_context.task"],
                result=(
                    ",".join(matched_functions)
                    if matched_functions
                    else "no_matching_employment_function"
                ),
            ),
            FindingTraceEntry(
                description=(
                    "Checked whether AI output materially influences access to "
                    "employment or employment-related opportunities."
                ),
                fact_refs=["use_context.materially_influences_decision"],
                result=(
                    "material_influence"
                    if materially_influences
                    else "no_material_influence"
                ),
            ),
        ]

        if potentially_applies:
            return Finding(
                category=self.category,
                issue_code="AIA_HIGH_RISK_EMPLOYMENT_PRELIMINARY",
                status=FindingStatus.POTENTIALLY_APPLIES,
                title="Employment-related high-risk classification potentially applies",
                summary=(
                    "The available facts match the preliminary employment-related "
                    "screening conditions under Article 6 and Annex III point 4(a). "
                    "This is not a definitive high-risk classification and requires "
                    "further legal assessment."
                ),
                rule_id=self.rule_id,
                rule_version=self.version,
                fact_refs=list(self.required_fact_paths),
                reason_codes=[
                    "EMPLOYMENT_CONTEXT",
                    *[
                        f"EMPLOYMENT_FUNCTION_{function.upper()}"
                        for function in matched_functions
                    ],
                    "MATERIAL_DECISION_INFLUENCE",
                ],
                legal_basis=list(self.legal_basis),
                requires_legal_review=True,
                trace=trace,
            )

        return Finding(
            category=self.category,
            issue_code="AIA_HIGH_RISK_EMPLOYMENT_PRELIMINARY",
            status=FindingStatus.DOES_NOT_APPLY,
            title="Employment-related high-risk screening criteria not met",
            summary=(
                "The available facts do not satisfy all preliminary conditions for "
                "this employment-related category. This does not determine whether "
                "another EU AI Act high-risk category may apply."
            ),
            rule_id=self.rule_id,
            rule_version=self.version,
            fact_refs=list(self.required_fact_paths),
            reason_codes=self._negative_reason_codes(
                employment_context=employment_context,
                matched_functions=matched_functions,
                materially_influences=materially_influences,
            ),
            legal_basis=list(self.legal_basis),
            trace=trace,
        )

    @classmethod
    def _matched_employment_functions(cls, task: str) -> tuple[str, ...]:
        normalized = re.sub(r"[_-]+", " ", task.casefold())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        has_person_target = any(
            target in normalized for target in cls._PERSON_TARGET_TERMS
        )

        matches: list[str] = []
        if "recruit" in normalized:
            matches.append("recruitment")
        if "select" in normalized:
            matches.append("selection")
        if "evaluat" in normalized and (
            has_person_target or "performance evaluat" in normalized
        ):
            matches.append("evaluation")
        if "rank" in normalized and has_person_target:
            matches.append("ranking")
        if "recommend" in normalized and has_person_target:
            matches.append("recommendation")
        return tuple(matches)

    @staticmethod
    def _negative_reason_codes(
        *,
        employment_context: bool,
        matched_functions: tuple[str, ...],
        materially_influences: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if not employment_context:
            reasons.append("NOT_EMPLOYMENT_CONTEXT")
        if not matched_functions:
            reasons.append("NO_LISTED_EMPLOYMENT_FUNCTION")
        if not materially_influences:
            reasons.append("NO_MATERIAL_DECISION_INFLUENCE")
        return reasons

