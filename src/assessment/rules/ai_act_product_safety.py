"""EU AI Act Article 6(1) product-safety high-risk screening."""

from __future__ import annotations

from types import MappingProxyType

from src.assessment.facts import AssessmentFacts, ProductRegulationFacts
from src.assessment.findings import (
    Finding,
    FindingCategory,
    FindingStatus,
    FindingTraceEntry,
    LegalBasis,
)
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import TriState
from src.assessment.product_regulation import (
    AnnexIInstrument,
    AnnexIInstrumentCatalog,
    AnnexIInstrumentNotFoundError,
    load_annex_i_instrument_catalog,
    validate_product_regulation_facts,
)
from src.assessment.rules.base import AssessmentRule


class AIActHighRiskProductSafetyRule(AssessmentRule):
    """Preliminarily screen the product-safety route in Article 6(1).

    The rule evaluates only authored facts. Catalogue selection identifies an
    Annex I instrument but never establishes coverage, conformity-assessment
    requirements, or legal applicability by itself.
    """

    framework = RegulatoryFramework.EU_AI_ACT
    rule_id = "AI_ACT_HIGH_RISK_PRODUCT_SAFETY"
    version = "2026.1"
    category = FindingCategory.HIGH_RISK_ARTICLE_6_1
    issue_code = "AIA_HIGH_RISK_ARTICLE_6_1_PRELIMINARY"
    required_fact_paths = (
        "product_regulation.ai_is_product",
        "product_regulation.ai_is_safety_component",
        "product_regulation.annex_i_instrument",
        "product_regulation.annex_i_instrument_confirmed",
        "product_regulation.third_party_conformity_required",
    )
    legal_basis = (
        LegalBasis(
            instrument="EU_AI_ACT",
            citation="Article 6(1)(a)",
            anchor="article:6:paragraph:1:point:a",
        ),
        LegalBasis(
            instrument="EU_AI_ACT",
            citation="Article 6(1)(b)",
            anchor="article:6:paragraph:1:point:b",
        ),
    )
    presentation_keys = MappingProxyType(
        {
            FindingStatus.POTENTIALLY_APPLIES.value: MappingProxyType(
                {
                    "title": "finding.ai_act_product_safety.positive.title",
                    "summary": "finding.ai_act_product_safety.positive.summary",
                }
            ),
            FindingStatus.DOES_NOT_APPLY.value: MappingProxyType(
                {
                    "title": "finding.ai_act_product_safety.negative.title",
                    "summary": "finding.ai_act_product_safety.negative.summary",
                }
            ),
            FindingStatus.UNDETERMINED.value: MappingProxyType(
                {
                    "title": "finding.ai_act_product_safety.undetermined.title",
                    "summary": "finding.ai_act_product_safety.undetermined.summary",
                }
            ),
        }
    )

    _PRODUCT_PATH = "product_regulation.ai_is_product"
    _SAFETY_PATH = "product_regulation.ai_is_safety_component"
    _INSTRUMENT_PATH = "product_regulation.annex_i_instrument"
    _CONFIRMATION_PATH = "product_regulation.annex_i_instrument_confirmed"
    _CONFORMITY_PATH = "product_regulation.third_party_conformity_required"

    def __init__(
        self,
        catalog: AnnexIInstrumentCatalog | None = None,
    ) -> None:
        if catalog is not None and not isinstance(
            catalog, AnnexIInstrumentCatalog
        ):
            raise TypeError("catalog must be an AnnexIInstrumentCatalog")
        self._catalog = catalog or load_annex_i_instrument_catalog()

    def required_fact_paths_for(
        self,
        facts: AssessmentFacts,
    ) -> tuple[str, ...]:
        """Return only facts still capable of changing this rule's outcome."""

        if not isinstance(facts, AssessmentFacts):
            raise TypeError("facts must be an AssessmentFacts instance")
        product_facts = facts.product_regulation
        if not isinstance(product_facts, ProductRegulationFacts):
            # Return no missing paths so evaluation runs and the existing
            # engine failure contract captures the malformed namespace.
            return ()
        relation_values = (
            product_facts.ai_is_product,
            product_facts.ai_is_safety_component,
        )

        invalid_relation_paths = tuple(
            path
            for path, value in zip(
                (self._PRODUCT_PATH, self._SAFETY_PATH),
                relation_values,
                strict=True,
            )
            if not isinstance(value, TriState)
        )
        if invalid_relation_paths:
            return invalid_relation_paths

        ai_is_product, ai_is_safety_component = relation_values
        if not (
            ai_is_product is TriState.YES
            or ai_is_safety_component is TriState.YES
        ):
            unresolved = tuple(
                path
                for path, value in zip(
                    (self._PRODUCT_PATH, self._SAFETY_PATH),
                    relation_values,
                    strict=True,
                )
                if value is TriState.UNKNOWN
            )
            return unresolved or (self._PRODUCT_PATH, self._SAFETY_PATH)

        confirmation = product_facts.annex_i_instrument_confirmed
        if not isinstance(confirmation, TriState):
            return (self._CONFIRMATION_PATH,)

        if confirmation is TriState.UNKNOWN:
            requirements = [self._CONFIRMATION_PATH]
            if product_facts.annex_i_instrument is None:
                requirements.insert(0, self._INSTRUMENT_PATH)
            return tuple(requirements)

        if confirmation is TriState.NO:
            if not isinstance(
                product_facts.third_party_conformity_required,
                TriState,
            ):
                return (self._CONFORMITY_PATH,)
            return ()

        # A confirmed-YES value without an instrument is contradictory and is
        # intentionally executed so the rule can return an undetermined
        # Finding rather than misreporting an ordinary missing answer.
        if product_facts.annex_i_instrument is None:
            return ()
        if not isinstance(product_facts.annex_i_instrument, str):
            return (self._INSTRUMENT_PATH,)

        conformity = product_facts.third_party_conformity_required
        if not isinstance(conformity, TriState):
            return (self._CONFORMITY_PATH,)
        if conformity is TriState.UNKNOWN:
            return (self._CONFORMITY_PATH,)
        return ()

    def evaluate(self, facts: AssessmentFacts) -> Finding:
        """Evaluate Article 6(1) using confirmed, catalogue-backed facts."""

        product_facts = facts.product_regulation
        instrument: AnnexIInstrument | None
        try:
            instrument = validate_product_regulation_facts(
                product_facts,
                catalog=self._catalog,
            )
        except AnnexIInstrumentNotFoundError:
            return self._undetermined_finding(
                product_facts,
                reason_codes=[
                    "ANNEX_I_INSTRUMENT_INVALID",
                    "INCONSISTENT_PRODUCT_REGULATION_FACTS",
                ],
            )

        if (
            product_facts.annex_i_instrument_confirmed is TriState.YES
            and instrument is None
        ):
            return self._undetermined_finding(
                product_facts,
                reason_codes=[
                    "ANNEX_I_INSTRUMENT_MISSING",
                    "INCONSISTENT_PRODUCT_REGULATION_FACTS",
                ],
            )
        if (
            product_facts.annex_i_instrument_confirmed is TriState.NO
            and product_facts.third_party_conformity_required is TriState.YES
        ):
            return self._undetermined_finding(
                product_facts,
                instrument=instrument,
                reason_codes=["INCONSISTENT_PRODUCT_REGULATION_FACTS"],
            )

        relation_reason = self._satisfied_relation_reason(product_facts)
        if relation_reason is None:
            return self._negative_finding(
                product_facts,
                instrument=instrument,
                reason_codes=["NEITHER_AI_PRODUCT_NOR_SAFETY_COMPONENT"],
            )
        if product_facts.annex_i_instrument_confirmed is TriState.NO:
            return self._negative_finding(
                product_facts,
                instrument=instrument,
                reason_codes=["ANNEX_I_COVERAGE_NOT_CONFIRMED"],
            )
        if product_facts.third_party_conformity_required is TriState.NO:
            return self._negative_finding(
                product_facts,
                instrument=instrument,
                reason_codes=["NO_THIRD_PARTY_CONFORMITY_REQUIREMENT"],
            )

        if (
            instrument is None
            or product_facts.annex_i_instrument_confirmed is not TriState.YES
            or product_facts.third_party_conformity_required is not TriState.YES
        ):
            # The requirement validator prevents ordinary unknown values from
            # reaching evaluation. Reaching this branch therefore indicates a
            # structurally invalid direct invocation.
            return self._undetermined_finding(
                product_facts,
                instrument=instrument,
                reason_codes=["INCONSISTENT_PRODUCT_REGULATION_FACTS"],
            )

        reason_codes = [
            relation_reason,
            "ANNEX_I_COVERAGE_CONFIRMED",
            "THIRD_PARTY_CONFORMITY_REQUIRED",
        ]
        return Finding(
            framework=self.framework,
            category=self.category,
            issue_code=self.issue_code,
            status=FindingStatus.POTENTIALLY_APPLIES,
            title="Product-safety high-risk classification potentially applies",
            summary=(
                "The confirmed facts indicate that the AI system is itself a "
                "covered product or performs a safety function within one, the "
                "product is covered by a listed Annex I instrument, and "
                "third-party conformity assessment is required. Article 6(1) "
                "may therefore classify the AI system as high-risk. This is a "
                "preliminary screening result and requires product-law and "
                "legal review."
            ),
            rule_id=self.rule_id,
            rule_version=self.version,
            fact_refs=list(self.required_fact_paths),
            reason_codes=reason_codes,
            legal_basis=self._authored_legal_basis(
                instrument,
                safety_component_relied_upon=(
                    relation_reason == "AI_IS_SAFETY_COMPONENT"
                ),
            ),
            requires_legal_review=True,
            trace=self._trace(product_facts, instrument),
        )

    @staticmethod
    def _satisfied_relation_reason(
        facts: ProductRegulationFacts,
    ) -> str | None:
        # Product is the deterministic first branch. If it is satisfied, an
        # unknown safety-component answer cannot block or alter the result.
        if facts.ai_is_product is TriState.YES:
            return "AI_IS_PRODUCT"
        if facts.ai_is_safety_component is TriState.YES:
            return "AI_IS_SAFETY_COMPONENT"
        return None

    def _negative_finding(
        self,
        facts: ProductRegulationFacts,
        *,
        instrument: AnnexIInstrument | None,
        reason_codes: list[str],
    ) -> Finding:
        relation_reason = self._satisfied_relation_reason(facts)
        return Finding(
            framework=self.framework,
            category=self.category,
            issue_code=self.issue_code,
            status=FindingStatus.DOES_NOT_APPLY,
            title="Article 6(1) product-safety screening criteria not met",
            summary=(
                "The confirmed facts do not meet this Article 6(1) "
                "product-safety route. This does not exclude Article 6(2), "
                "another Annex III category, or another applicable law."
            ),
            rule_id=self.rule_id,
            rule_version=self.version,
            fact_refs=list(self.required_fact_paths),
            reason_codes=reason_codes,
            legal_basis=self._authored_legal_basis(
                instrument,
                safety_component_relied_upon=(
                    relation_reason == "AI_IS_SAFETY_COMPONENT"
                ),
            ),
            trace=self._trace(facts, instrument),
        )

    def _undetermined_finding(
        self,
        facts: ProductRegulationFacts,
        *,
        reason_codes: list[str],
        instrument: AnnexIInstrument | None = None,
    ) -> Finding:
        relation_reason = self._satisfied_relation_reason(facts)
        return Finding(
            framework=self.framework,
            category=self.category,
            issue_code=self.issue_code,
            status=FindingStatus.UNDETERMINED,
            title="Article 6(1) assessment requires fact reconciliation",
            summary=(
                "The product-regulation facts are internally inconsistent or "
                "contain an invalid Annex I selection. Reconcile the facts "
                "before drawing a preliminary Article 6(1) conclusion."
            ),
            rule_id=self.rule_id,
            rule_version=self.version,
            fact_refs=list(self.required_fact_paths),
            reason_codes=reason_codes,
            legal_basis=self._authored_legal_basis(
                instrument,
                safety_component_relied_upon=(
                    relation_reason == "AI_IS_SAFETY_COMPONENT"
                ),
            ),
            requires_legal_review=True,
            trace=self._trace(facts, instrument),
        )

    def _authored_legal_basis(
        self,
        instrument: AnnexIInstrument | None,
        *,
        safety_component_relied_upon: bool,
    ) -> list[LegalBasis]:
        basis: list[LegalBasis] = []
        if safety_component_relied_upon:
            basis.append(
                LegalBasis(
                    instrument="EU_AI_ACT",
                    citation="Article 3(14)",
                    anchor="article:3:point:14",
                )
            )
        basis.extend(self.legal_basis)
        if instrument is not None:
            basis.append(
                LegalBasis(
                    instrument="EU_AI_ACT",
                    citation=instrument.canonical_reference,
                    anchor=(
                        "annex:I:section:"
                        f"{instrument.annex_section.value}:point:"
                        f"{instrument.annex_point}"
                    ),
                )
            )
        return basis

    def _trace(
        self,
        facts: ProductRegulationFacts,
        instrument: AnnexIInstrument | None,
    ) -> list[FindingTraceEntry]:
        relation_reason = self._satisfied_relation_reason(facts)
        if relation_reason is None:
            relation_result = (
                "unresolved_product_relationship"
                if TriState.UNKNOWN
                in (facts.ai_is_product, facts.ai_is_safety_component)
                else "neither_ai_product_nor_safety_component"
            )
        else:
            relation_result = relation_reason
        return [
            FindingTraceEntry(
                description=(
                    "Checked whether the AI system is itself a product or is "
                    "used as a safety component of one."
                ),
                fact_refs=[self._PRODUCT_PATH, self._SAFETY_PATH],
                result=relation_result,
            ),
            FindingTraceEntry(
                description=(
                    "Validated the selected Annex I instrument and its "
                    "separate coverage confirmation."
                ),
                fact_refs=[self._INSTRUMENT_PATH, self._CONFIRMATION_PATH],
                result=(
                    "annex_i_coverage_confirmed"
                    if instrument is not None
                    and facts.annex_i_instrument_confirmed is TriState.YES
                    else "annex_i_coverage_not_confirmed"
                ),
            ),
            FindingTraceEntry(
                description=(
                    "Checked whether the applicable product route requires "
                    "third-party conformity assessment."
                ),
                fact_refs=[self._CONFORMITY_PATH],
                result=(
                    "third_party_conformity_required"
                    if facts.third_party_conformity_required is TriState.YES
                    else "third_party_conformity_not_required_or_unresolved"
                ),
            ),
        ]


__all__ = ["AIActHighRiskProductSafetyRule"]
