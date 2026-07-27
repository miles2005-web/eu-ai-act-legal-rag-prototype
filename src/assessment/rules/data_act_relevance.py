"""Preliminary EU Data Act relevance screening."""

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


class EUDataActRelevanceRule(AssessmentRule):
    """Identify cases warranting further EU Data Act assessment.

    This trigger does not determine final scope, actor roles, obligations,
    exemptions, or compliance under Regulation (EU) 2023/2854.
    """

    rule_id = "EU_DATA_ACT_RELEVANCE"
    version = "2026.1"
    planning_ordering_key = "040"
    framework = RegulatoryFramework.EU_DATA_ACT
    category = FindingCategory.DATA_GOVERNANCE
    required_fact_paths = (
        "data_act.connected_product",
        "data_act.related_service",
        "data_act.data_generated",
    )
    legal_basis = (
        LegalBasis(
            instrument="EU_DATA_ACT",
            citation="Article 1(1)(a)",
            anchor="article:1:paragraph:1:point:a",
        ),
        LegalBasis(
            instrument="EU_DATA_ACT",
            citation="Article 2(5)-(6)",
            anchor="article:2:points:5-6",
        ),
    )

    def evaluate(self, facts: AssessmentFacts) -> Finding:
        """Return a preliminary relevance result for complete required facts."""

        connected_product = facts.data_act.connected_product is TriState.YES
        related_service = facts.data_act.related_service is TriState.YES
        data_generated = facts.data_act.data_generated is TriState.YES
        relevant_context = connected_product or related_service
        potentially_relevant = relevant_context and data_generated

        trace = [
            FindingTraceEntry(
                description="Checked whether a connected product is involved.",
                fact_refs=["data_act.connected_product"],
                result=(
                    "connected_product"
                    if connected_product
                    else "no_connected_product"
                ),
            ),
            FindingTraceEntry(
                description="Checked whether a related service is involved.",
                fact_refs=["data_act.related_service"],
                result=(
                    "related_service"
                    if related_service
                    else "no_related_service"
                ),
            ),
            FindingTraceEntry(
                description=(
                    "Checked whether the product or service generates data."
                ),
                fact_refs=["data_act.data_generated"],
                result=(
                    "data_generated"
                    if data_generated
                    else "no_data_generated"
                ),
            ),
        ]

        if potentially_relevant:
            return Finding(
                framework=self.framework,
                category=self.category,
                issue_code="EU_DATA_ACT_RELEVANCE",
                status=FindingStatus.POTENTIALLY_APPLIES,
                title="Data Act relevance potentially applies",
                summary=(
                    "The known facts indicate a connected product or related "
                    "service that generates data. Further EU Data Act assessment "
                    "is warranted. This preliminary trigger does not determine "
                    "final scope, obligations, exemptions, or compliance."
                ),
                rule_id=self.rule_id,
                rule_version=self.version,
                fact_refs=list(self.required_fact_paths),
                reason_codes=[
                    *(
                        ["CONNECTED_PRODUCT"]
                        if connected_product
                        else []
                    ),
                    *(
                        ["RELATED_SERVICE"]
                        if related_service
                        else []
                    ),
                    "DATA_GENERATED",
                ],
                legal_basis=list(self.legal_basis),
                requires_legal_review=True,
                trace=trace,
            )

        return Finding(
            framework=self.framework,
            category=self.category,
            issue_code="EU_DATA_ACT_RELEVANCE",
            status=FindingStatus.DOES_NOT_APPLY,
            title="Data Act relevance trigger not met",
            summary=(
                "The complete facts do not satisfy this preliminary Data Act "
                "relevance trigger. This result does not determine other Data Act "
                "contexts or general regulatory compliance."
            ),
            rule_id=self.rule_id,
            rule_version=self.version,
            fact_refs=list(self.required_fact_paths),
            reason_codes=self._negative_reason_codes(
                relevant_context=relevant_context,
                data_generated=data_generated,
            ),
            legal_basis=list(self.legal_basis),
            trace=trace,
        )

    @staticmethod
    def _negative_reason_codes(
        *,
        relevant_context: bool,
        data_generated: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if not relevant_context:
            reasons.append("NO_CONNECTED_PRODUCT_OR_RELATED_SERVICE")
        if not data_generated:
            reasons.append("NO_DATA_GENERATED")
        return reasons
