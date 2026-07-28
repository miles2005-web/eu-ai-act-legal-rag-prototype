"""Preliminary GDPR Article 22 relevance screening."""

from __future__ import annotations

from src.assessment.facts import AssessmentFacts
from src.assessment.findings import (
    Finding,
    FindingCategory,
    FindingStatus,
    FindingTraceEntry,
    LegalBasis,
)
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import TriState
from src.assessment.rules.base import AssessmentRule


class GDPRArticle22RelevanceRule(AssessmentRule):
    """Identify cases requiring a fuller GDPR Article 22 assessment.

    This trigger does not determine whether Article 22 applies, whether an
    exception is available, whether safeguards are adequate, or whether the
    processing complies with the GDPR.
    """

    rule_id = "GDPR_ARTICLE22_RELEVANCE"
    version = "2026.1"
    planning_ordering_key = "030"
    framework = RegulatoryFramework.GDPR
    category = FindingCategory.DATA_PROTECTION
    required_fact_paths = (
        "data_protection.personal_data_processed",
        "data_protection.automated_individual_decision",
        "use_context.materially_influences_decision",
    )
    legal_basis = (
        LegalBasis(
            instrument="GDPR",
            citation="Article 22(1)",
            anchor="article:22:paragraph:1",
        ),
    )

    def evaluate(self, facts: AssessmentFacts) -> Finding:
        """Return a cautious relevance result for complete required facts."""

        personal_data = (
            facts.data_protection.personal_data_processed is TriState.YES
        )
        automated_decision = (
            facts.data_protection.automated_individual_decision
            is TriState.YES
        )
        material_influence = (
            facts.use_context.materially_influences_decision is TriState.YES
        )
        relevant = personal_data and automated_decision and material_influence

        trace = [
            FindingTraceEntry(
                description="Checked whether personal data are processed.",
                fact_refs=["data_protection.personal_data_processed"],
                result=(
                    "personal_data_processed"
                    if personal_data
                    else "personal_data_not_processed"
                ),
            ),
            FindingTraceEntry(
                description=(
                    "Checked whether an automated individual decision is involved."
                ),
                fact_refs=[
                    "data_protection.automated_individual_decision"
                ],
                result=(
                    "automated_individual_decision"
                    if automated_decision
                    else "no_automated_individual_decision"
                ),
            ),
            FindingTraceEntry(
                description=(
                    "Checked whether the decision materially influences an individual."
                ),
                fact_refs=["use_context.materially_influences_decision"],
                result=(
                    "material_influence"
                    if material_influence
                    else "no_material_influence"
                ),
            ),
        ]

        if relevant:
            return Finding(
                framework=self.framework,
                category=self.category,
                issue_code="GDPR_ARTICLE22_RELEVANCE",
                status=FindingStatus.POTENTIALLY_APPLIES,
                title="GDPR Article 22 assessment may be relevant",
                summary=(
                    "The available facts indicate personal-data processing, an "
                    "automated individual decision, and material influence on an "
                    "individual. A fuller GDPR Article 22 assessment is warranted. "
                    "This trigger does not determine that Article 22 applies or "
                    "that the processing is non-compliant."
                ),
                rule_id=self.rule_id,
                rule_version=self.version,
                fact_refs=list(self.required_fact_paths),
                reason_codes=[
                    "PERSONAL_DATA_PROCESSED",
                    "AUTOMATED_INDIVIDUAL_DECISION",
                    "MATERIAL_INDIVIDUAL_INFLUENCE",
                ],
                legal_basis=list(self.legal_basis),
                requires_legal_review=True,
                trace=trace,
            )

        return Finding(
            framework=self.framework,
            category=self.category,
            issue_code="GDPR_ARTICLE22_RELEVANCE",
            status=FindingStatus.DOES_NOT_APPLY,
            title="GDPR Article 22 relevance trigger not met",
            summary=(
                "The known facts do not satisfy all conditions for this "
                "preliminary Article 22 relevance trigger. This is not a general "
                "GDPR compliance conclusion."
            ),
            rule_id=self.rule_id,
            rule_version=self.version,
            fact_refs=list(self.required_fact_paths),
            reason_codes=self._negative_reason_codes(
                personal_data=personal_data,
                automated_decision=automated_decision,
                material_influence=material_influence,
            ),
            legal_basis=list(self.legal_basis),
            trace=trace,
        )

    @staticmethod
    def _negative_reason_codes(
        *,
        personal_data: bool,
        automated_decision: bool,
        material_influence: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if not personal_data:
            reasons.append("NO_PERSONAL_DATA_PROCESSING")
        if not automated_decision:
            reasons.append("NO_AUTOMATED_INDIVIDUAL_DECISION")
        if not material_influence:
            reasons.append("NO_MATERIAL_INDIVIDUAL_INFLUENCE")
        return reasons
