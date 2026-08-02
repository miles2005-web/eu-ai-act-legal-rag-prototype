"""Isolated v0.6 planned execution path built on the F1 and F2A contracts.

This module deliberately does not replace :mod:`src.assessment.engine`.  A
caller must explicitly construct ``PlannedAssessmentEngine`` and supply an
already validated plan, expanded invocations, and expanded authorizations.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import re
from types import MappingProxyType
from typing import Any, ClassVar, Protocol, runtime_checkable

from src.assessment.baselines import AssessmentBaseline, RuleBaselineEntry
from src.assessment.context import AssessmentContext, PrerequisiteFindingSummary
from src.assessment.execution import RuleExecutionRecord, RuleExecutionStatus
from src.assessment.facts import AssessmentFacts
from src.assessment.findings import Finding, FindingStatus
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.fingerprints import canonicalize_v3_facts
from src.assessment.invocations import AuthorizedRuleInvocation, RuleInvocation
from src.assessment.models import (
    SerializableModel,
    freeze_value,
    to_primitive,
    validate_stable_identifier,
)
from src.assessment.requirements import FactRequirementValidator
from src.assessment.rules.base import AssessmentRule
from src.assessment.rules.planning import (
    RulePhase,
    RulePlanningMetadata,
    RulesetPlan,
)
from src.assessment.rules.registry import RuleRegistry
from src.assessment.scope import AssessmentScope, InvocationId


class PlannedExecutionInputError(ValueError):
    """Raised when F2B1 inputs are inconsistent before legal execution."""


class PlannedRuleOutputError(TypeError):
    """Raised when a planned rule violates the one-Finding output boundary."""


@runtime_checkable
class ContextAwareAssessmentRule(Protocol):
    """Narrow opt-in protocol for future rules that consume a context."""

    def evaluate_context(self, context: AssessmentContext) -> Finding | None:
        """Evaluate one immutable scoped context."""


@dataclass(frozen=True, slots=True)
class InvocationEvidenceRequirement(SerializableModel):
    """Explicit availability of a required Evidence proposition set."""

    invocation_id: InvocationId
    requirement_id: str
    available: bool
    unavailable_reason: str | None = None
    contract_version: str = "1.0.0"

    REQUIRED_SERIALIZED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"invocation_id", "requirement_id", "available"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "invocation_id", InvocationId(self.invocation_id))
        validate_stable_identifier(
            self.requirement_id,
            field_name="requirement_id",
        )
        if not isinstance(self.available, bool):
            raise TypeError("available must be a bool")
        if self.contract_version != "1.0.0":
            raise ValueError(
                "unsupported InvocationEvidenceRequirement contract_version"
            )
        if self.available and self.unavailable_reason is not None:
            raise ValueError(
                "available Evidence requirements cannot have an unavailable_reason"
            )
        if not self.available:
            if (
                not isinstance(self.unavailable_reason, str)
                or not self.unavailable_reason.strip()
            ):
                raise ValueError(
                    "unavailable Evidence requirements require a stable reason"
                )
            object.__setattr__(
                self,
                "unavailable_reason",
                validate_stable_identifier(
                    self.unavailable_reason.strip(),
                    field_name="unavailable_reason",
                ),
            )


@dataclass(frozen=True, slots=True)
class InvocationFindingAssociation(SerializableModel):
    """Canonical one-to-zero-or-one invocation/Finding association."""

    invocation_id: InvocationId
    finding_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "invocation_id", InvocationId(self.invocation_id))
        validate_stable_identifier(self.finding_id, field_name="finding_id")


@dataclass(frozen=True, slots=True)
class ExecutionDiagnostic(SerializableModel):
    """Stable, non-legal diagnostic for an unresolved invocation."""

    invocation_id: InvocationId
    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "invocation_id", InvocationId(self.invocation_id))
        validate_stable_identifier(self.code, field_name="code")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")
        object.__setattr__(self, "message", self.message.strip())


@dataclass(frozen=True, slots=True)
class _ValidatedPlannedExecutionInputs:
    """One immutable boundary shared by execution and fingerprinting."""

    facts: AssessmentFacts
    plan: RulesetPlan
    ordered_invocations: tuple[RuleInvocation, ...]
    ordered_authorizations: tuple[AuthorizedRuleInvocation, ...]
    ordered_evidence_requirements: tuple[InvocationEvidenceRequirement, ...]
    authorization_by_invocation: Mapping[InvocationId, AuthorizedRuleInvocation]
    evidence_by_invocation: Mapping[InvocationId, InvocationEvidenceRequirement]
    baseline: AssessmentBaseline


class PlannedExecutionFingerprintInput:
    """Immutable, strictly validated pre-execution fingerprint input."""

    CONTRACT_VERSION = "1.0.0"
    ENGINE_VERSION = "3.0.0"

    __slots__ = ("_validated", "_payload", "_digest", "_sealed")

    def __init__(
        self,
        *,
        facts_snapshot: AssessmentFacts,
        plan: RulesetPlan,
        invocations: Iterable[RuleInvocation],
        authorizations: Iterable[AuthorizedRuleInvocation],
        baseline: AssessmentBaseline,
        evidence_requirements: Iterable[InvocationEvidenceRequirement] = (),
    ) -> None:
        validated = _validate_planned_execution_inputs(
            facts_snapshot=facts_snapshot,
            plan=plan,
            invocations=invocations,
            authorizations=authorizations,
            baseline=baseline,
            evidence_requirements=evidence_requirements,
        )
        self._seal(validated)

    @classmethod
    def _from_validated(
        cls,
        validated: _ValidatedPlannedExecutionInputs,
    ) -> PlannedExecutionFingerprintInput:
        instance = cls.__new__(cls)
        instance._seal(validated)
        return instance

    def _seal(self, validated: _ValidatedPlannedExecutionInputs) -> None:
        payload = _fingerprint_payload(validated)
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        object.__setattr__(self, "_validated", validated)
        object.__setattr__(self, "_payload", freeze_value(payload))
        object.__setattr__(self, "_digest", sha256(canonical.encode()).hexdigest())
        object.__setattr__(self, "_sealed", True)

    @classmethod
    def from_canonical_payload(
        cls,
        payload: Mapping[str, Any],
    ) -> PlannedExecutionFingerprintInput:
        """Strictly hydrate canonical inputs and recompute all derived values."""

        if not isinstance(payload, Mapping):
            raise TypeError("fingerprint payload must be an object")
        expected = {
            "contract_version",
            "engine_version",
            "assessment_context_version",
            "execution_record_version",
            "report_schema_version",
            "facts",
            "ruleset_plan",
            "baseline",
            "invocations",
            "authorizations",
            "evidence_requirements",
        }
        _require_exact_fields(payload, expected, "fingerprint payload")
        if payload["contract_version"] != cls.CONTRACT_VERSION:
            raise ValueError("unsupported fingerprint contract version")
        if payload["engine_version"] != cls.ENGINE_VERSION:
            raise ValueError("unsupported fingerprint engine version")
        if payload["assessment_context_version"] != AssessmentContext.CONTRACT_VERSION:
            raise ValueError("unsupported fingerprint AssessmentContext version")
        if payload["execution_record_version"] != "1.0.0":
            raise ValueError("unsupported fingerprint execution-record version")
        if payload["report_schema_version"] != "2.0.0":
            raise ValueError("unsupported fingerprint report schema version")

        plan_payload = payload["ruleset_plan"]
        if not isinstance(plan_payload, Mapping):
            raise TypeError("ruleset_plan must be an object")
        _require_exact_fields(
            plan_payload,
            {"dependency_graph_hash", "ruleset_baseline_id", "metadata"},
            "ruleset_plan",
        )
        metadata_payload = plan_payload["metadata"]
        if not isinstance(metadata_payload, Mapping):
            raise TypeError("ruleset plan metadata must be an object")
        _require_exact_fields(
            metadata_payload,
            {"metadata_contract_version", "ordered_rules"},
            "ruleset plan metadata",
        )
        if metadata_payload["metadata_contract_version"] != RulesetPlan.CONTRACT_VERSION:
            raise ValueError("unsupported RulesetPlan metadata contract version")
        raw_rules = metadata_payload["ordered_rules"]
        if not isinstance(raw_rules, list):
            raise TypeError("ruleset ordered_rules must be a list")
        plan = RulesetPlan.build(
            RulePlanningMetadata.from_dict(item) for item in raw_rules
        )
        if plan_payload["dependency_graph_hash"] != plan.dependency_graph_hash:
            raise PlannedExecutionInputError("fabricated plan dependency graph hash")
        if plan_payload["ruleset_baseline_id"] != plan.ruleset_baseline_id:
            raise PlannedExecutionInputError("fabricated plan ruleset baseline ID")
        if plan.canonical_dict() != dict(metadata_payload):
            raise PlannedExecutionInputError("non-canonical RulesetPlan metadata")

        try:
            facts = AssessmentFacts.from_dict(dict(payload["facts"]))
            baseline = AssessmentBaseline.from_dict(dict(payload["baseline"]))
            invocations = tuple(
                RuleInvocation.from_dict(item) for item in payload["invocations"]
            )
            authorizations = tuple(
                AuthorizedRuleInvocation.from_dict(item)
                for item in payload["authorizations"]
            )
            evidence_requirements = tuple(
                InvocationEvidenceRequirement.from_dict(item)
                for item in payload["evidence_requirements"]
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise PlannedExecutionInputError(
                "invalid canonical fingerprint payload"
            ) from exc
        instance = cls(
            facts_snapshot=facts,
            plan=plan,
            invocations=invocations,
            authorizations=authorizations,
            baseline=baseline,
            evidence_requirements=evidence_requirements,
        )
        if instance.canonical_payload() != dict(payload):
            raise PlannedExecutionInputError(
                "fingerprint payload is not in canonical form"
            )
        return instance

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PlannedExecutionFingerprintInput is immutable")
        object.__setattr__(self, name, value)

    def canonical_payload(self) -> dict[str, Any]:
        return to_primitive(self._payload)

    def digest(self) -> str:
        return self._digest

    @property
    def facts_schema_version(self) -> str:
        return self._validated.facts.schema_version

    @property
    def ruleset_plan(self) -> RulesetPlan:
        return RulesetPlan.build(self._validated.plan.ordered_rules)

    @property
    def baseline(self) -> AssessmentBaseline:
        return AssessmentBaseline.from_dict(self._validated.baseline.to_dict())

    @property
    def ordered_invocations(self) -> tuple[RuleInvocation, ...]:
        return tuple(
            RuleInvocation.from_dict(item.to_dict())
            for item in self._validated.ordered_invocations
        )


class PlannedAssessmentResult(SerializableModel):
    """Immutable F2B1 execution result; it is not AssessmentRun 3.0."""

    CONTRACT_VERSION = "1.0.0"
    ENGINE_VERSION = "3.0.0"

    __slots__ = (
        "_fingerprint_input",
        "_records",
        "_finding_payloads",
        "_associations",
        "_diagnostics",
        "_sealed",
    )

    def __init__(
        self,
        *,
        fingerprint_input: PlannedExecutionFingerprintInput,
        execution_records: Iterable[RuleExecutionRecord],
        findings: Iterable[Finding],
        finding_associations: Iterable[InvocationFindingAssociation],
        diagnostics: Iterable[ExecutionDiagnostic],
    ) -> None:
        if not isinstance(fingerprint_input, PlannedExecutionFingerprintInput):
            raise TypeError(
                "fingerprint_input must be a PlannedExecutionFingerprintInput"
            )
        fingerprint = PlannedExecutionFingerprintInput.from_canonical_payload(
            fingerprint_input.canonical_payload()
        )
        baseline_copy = fingerprint.baseline
        invocations = fingerprint.ordered_invocations
        records = tuple(
            RuleExecutionRecord.from_dict(item.to_dict())
            for item in _typed_tuple(
                execution_records,
                RuleExecutionRecord,
                field_name="execution_records",
            )
        )
        findings_values = tuple(findings)
        if any(not isinstance(item, Finding) for item in findings_values):
            raise TypeError("findings must contain Finding instances")
        finding_payloads = tuple(
            freeze_value(Finding.from_dict(item.to_dict()).to_dict())
            for item in findings_values
        )
        associations = tuple(
            InvocationFindingAssociation.from_dict(item.to_dict())
            for item in finding_associations
        )
        diagnostic_values = tuple(
            ExecutionDiagnostic.from_dict(item.to_dict())
            for item in _typed_tuple(
                diagnostics,
                ExecutionDiagnostic,
                field_name="diagnostics",
            )
        )
        if len(records) != len(invocations):
            raise ValueError("every ordered invocation requires one execution record")
        invocation_ids = tuple(item.invocation_id for item in invocations)
        if len(set(invocation_ids)) != len(invocation_ids):
            raise ValueError("ordered invocation IDs must be unique")
        if tuple(item.invocation for item in records) != invocations:
            raise ValueError("execution record order must match invocation order")
        record_ids = tuple(item.invocation.invocation_id for item in records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("execution record invocations must be unique")
        for record in records:
            if record.schema_version != baseline_copy.execution_record_version:
                raise ValueError("execution record schema version differs from baseline")
            if record.ruleset_baseline_id != baseline_copy.ruleset_baseline_id:
                raise ValueError("execution record ruleset baseline ID differs from result")
            if record.evidence_baseline_id != baseline_copy.legal_source_baseline_id:
                raise ValueError("execution record Evidence baseline ID differs from result")
        association_ids = [item.finding_id for item in associations]
        finding_ids = [str(item["finding_id"]) for item in finding_payloads]
        if len(set(association_ids)) != len(association_ids):
            raise ValueError("Finding associations must not contain duplicate IDs")
        if association_ids != finding_ids:
            raise ValueError(
                "Finding associations must exactly cover Findings in result order"
            )
        record_links = [
            item.finding_id for item in records if item.finding_id is not None
        ]
        if record_links != association_ids:
            raise ValueError(
                "completed execution record Finding links must match associations"
            )
        invocation_links = [
            item.invocation.invocation_id
            for item in records
            if item.finding_id is not None
        ]
        if [item.invocation_id for item in associations] != invocation_links:
            raise ValueError(
                "Finding associations must preserve the producing invocation scope"
            )
        records_by_id = {
            record.invocation.invocation_id: record for record in records
        }
        for association, payload in zip(
            associations,
            finding_payloads,
            strict=True,
        ):
            record = records_by_id.get(association.invocation_id)
            if record is None:
                raise ValueError("Finding association references unknown invocation")
            if record.status is not RuleExecutionStatus.COMPLETED:
                raise ValueError("only COMPLETED records may bind Findings")
            if record.finding_id != association.finding_id:
                raise ValueError("Finding association differs from execution record")
            if str(payload["finding_id"]) != association.finding_id:
                raise ValueError("formal Finding identity differs from association")
            if payload.get("rule_id") != record.invocation.rule_id:
                raise ValueError("Finding rule_id differs from producing invocation")
            if payload.get("rule_version") != record.invocation.rule_version:
                raise ValueError("Finding rule_version differs from producing invocation")
            if payload.get("assessment_run_id") is not None:
                raise ValueError("F2B1 Findings cannot reference an AssessmentRun")

        unresolved = tuple(
            record
            for record in records
            if record.status is not RuleExecutionStatus.COMPLETED
        )
        if len(diagnostic_values) != len(unresolved):
            raise ValueError(
                "every unresolved record requires exactly one diagnostic"
            )
        diagnostic_ids = tuple(item.invocation_id for item in diagnostic_values)
        if len(set(diagnostic_ids)) != len(diagnostic_ids):
            raise ValueError("diagnostic invocation IDs must be unique")
        expected_diagnostics = tuple(
            _expected_diagnostic(record) for record in unresolved
        )
        if diagnostic_values != expected_diagnostics:
            raise ValueError(
                "diagnostics must exactly match unresolved execution records"
            )

        object.__setattr__(self, "_fingerprint_input", fingerprint)
        object.__setattr__(self, "_records", records)
        object.__setattr__(self, "_finding_payloads", finding_payloads)
        object.__setattr__(self, "_associations", associations)
        object.__setattr__(self, "_diagnostics", diagnostic_values)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PlannedAssessmentResult is immutable")
        object.__setattr__(self, name, value)

    @property
    def engine_version(self) -> str:
        return self.ENGINE_VERSION

    @property
    def execution_contract_version(self) -> str:
        return self.CONTRACT_VERSION

    @property
    def facts_schema_version(self) -> str:
        return self._fingerprint_input.facts_schema_version

    @property
    def ruleset_baseline(self) -> AssessmentBaseline:
        return self._fingerprint_input.baseline

    @property
    def ordered_invocations(self) -> tuple[RuleInvocation, ...]:
        return self._fingerprint_input.ordered_invocations

    @property
    def execution_records(self) -> tuple[RuleExecutionRecord, ...]:
        return tuple(
            RuleExecutionRecord.from_dict(item.to_dict()) for item in self._records
        )

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(
            Finding.from_dict(to_primitive(item)) for item in self._finding_payloads
        )

    @property
    def finding_associations(self) -> tuple[InvocationFindingAssociation, ...]:
        return tuple(
            InvocationFindingAssociation.from_dict(item.to_dict())
            for item in self._associations
        )

    @property
    def diagnostics(self) -> tuple[ExecutionDiagnostic, ...]:
        return tuple(
            ExecutionDiagnostic.from_dict(item.to_dict())
            for item in self._diagnostics
        )

    @property
    def input_fingerprint(self) -> str:
        return self._fingerprint_input.digest()

    @property
    def fingerprint_payload(self) -> dict[str, Any]:
        return self._fingerprint_input.canonical_payload()

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": self.ENGINE_VERSION,
            "execution_contract_version": self.CONTRACT_VERSION,
            "fingerprint_payload": self._fingerprint_input.canonical_payload(),
            "input_fingerprint": self._fingerprint_input.digest(),
            "execution_records": [item.to_dict() for item in self._records],
            "findings": [to_primitive(item) for item in self._finding_payloads],
            "finding_associations": [
                item.to_dict() for item in self._associations
            ],
            "diagnostics": [item.to_dict() for item in self._diagnostics],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlannedAssessmentResult:
        if not isinstance(payload, dict):
            raise TypeError("PlannedAssessmentResult payload must be an object")
        expected = {
            "engine_version",
            "execution_contract_version",
            "fingerprint_payload",
            "input_fingerprint",
            "execution_records",
            "findings",
            "finding_associations",
            "diagnostics",
        }
        unknown = sorted(set(payload).difference(expected))
        missing = sorted(expected.difference(payload))
        if unknown:
            raise ValueError(
                "unknown PlannedAssessmentResult fields: " + ", ".join(unknown)
            )
        if missing:
            raise ValueError(
                "PlannedAssessmentResult payload is missing fields: "
                + ", ".join(missing)
            )
        if payload["engine_version"] != cls.ENGINE_VERSION:
            raise ValueError("unsupported planned engine version")
        if payload["execution_contract_version"] != cls.CONTRACT_VERSION:
            raise ValueError("unsupported planned execution contract version")
        fingerprint = PlannedExecutionFingerprintInput.from_canonical_payload(
            payload["fingerprint_payload"]
        )
        supplied_digest = payload["input_fingerprint"]
        if (
            not isinstance(supplied_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", supplied_digest) is None
        ):
            raise ValueError("malformed planned input fingerprint")
        if supplied_digest != fingerprint.digest():
            raise ValueError("planned input fingerprint digest mismatch")
        return cls(
            fingerprint_input=fingerprint,
            execution_records=tuple(
                RuleExecutionRecord.from_dict(item)
                for item in payload["execution_records"]
            ),
            findings=tuple(Finding.from_dict(item) for item in payload["findings"]),
            finding_associations=tuple(
                InvocationFindingAssociation.from_dict(item)
                for item in payload["finding_associations"]
            ),
            diagnostics=tuple(
                ExecutionDiagnostic.from_dict(item)
                for item in payload["diagnostics"]
            ),
        )


class LegacyScreeningRuleAdapter:
    """Explicitly adapt dependency-free screening rules to AssessmentContext."""

    @staticmethod
    def supports(rule: AssessmentRule, plan: RulesetPlan) -> bool:
        metadata = plan.metadata_for(rule.rule_id)
        return (
            metadata.phase is RulePhase.SCREENING
            and not metadata.dependencies
        )

    @classmethod
    def evaluate(
        cls,
        rule: AssessmentRule,
        context: AssessmentContext,
        plan: RulesetPlan,
    ) -> Finding | None:
        if isinstance(rule, ContextAwareAssessmentRule):
            return rule.evaluate_context(context)
        if not cls.supports(rule, plan):
            raise PlannedExecutionInputError(
                f"Rule {rule.rule_id!r} requires evaluate_context() for planned "
                "dependency execution"
            )
        return rule.evaluate(context.facts_snapshot)


class PlannedAssessmentEngine:
    """Execute explicit scoped invocations without affecting v0.5 callers."""

    VERSION = "3.0.0"
    CONTRACT_VERSION = "1.0.0"

    def __init__(
        self,
        registry: RuleRegistry,
        *,
        requirement_validator: FactRequirementValidator | None = None,
    ) -> None:
        if not isinstance(registry, RuleRegistry):
            raise TypeError("registry must be a RuleRegistry")
        if requirement_validator is not None and not isinstance(
            requirement_validator,
            FactRequirementValidator,
        ):
            raise TypeError(
                "requirement_validator must be a FactRequirementValidator"
            )
        self._registry = registry
        self._requirement_validator = (
            requirement_validator or FactRequirementValidator()
        )

    @property
    def engine_version(self) -> str:
        return self.VERSION

    def run(
        self,
        *,
        facts_snapshot: AssessmentFacts,
        plan: RulesetPlan,
        invocations: Iterable[RuleInvocation],
        authorizations: Iterable[AuthorizedRuleInvocation],
        baseline: AssessmentBaseline,
        evidence_requirements: Iterable[InvocationEvidenceRequirement] = (),
    ) -> PlannedAssessmentResult:
        """Validate all inputs, then execute each invocation exactly once."""

        inputs = _validate_planned_execution_inputs(
            facts_snapshot=facts_snapshot,
            plan=plan,
            invocations=invocations,
            authorizations=authorizations,
            baseline=baseline,
            evidence_requirements=evidence_requirements,
        )
        self._validate_registry(inputs.plan)
        facts = inputs.facts
        plan_snapshot = inputs.plan
        ordered_invocations = inputs.ordered_invocations
        authorization_by_invocation = inputs.authorization_by_invocation
        baseline_snapshot = inputs.baseline
        evidence_by_invocation = inputs.evidence_by_invocation
        fingerprint_input = PlannedExecutionFingerprintInput._from_validated(inputs)

        records: list[RuleExecutionRecord] = []
        findings: list[Finding] = []
        associations: list[InvocationFindingAssociation] = []
        diagnostics: list[ExecutionDiagnostic] = []
        record_by_id: dict[InvocationId, RuleExecutionRecord] = {}
        finding_by_invocation: dict[InvocationId, Finding] = {}

        for invocation in ordered_invocations:
            authorization = authorization_by_invocation.get(
                invocation.invocation_id
            )
            if authorization is None:
                record = self._record(
                    invocation,
                    RuleExecutionStatus.NOT_AUTHORIZED,
                    baseline_snapshot,
                )
                diagnostics.append(
                    self._diagnostic(record, "authorization.not_explicit")
                )
                records.append(record)
                record_by_id[invocation.invocation_id] = record
                continue

            prerequisites, dependency_reason = self._prerequisite_summaries(
                invocation,
                record_by_id,
                finding_by_invocation,
            )
            if dependency_reason is not None:
                record = self._record(
                    invocation,
                    RuleExecutionStatus.BLOCKED_BY_DEPENDENCY,
                    baseline_snapshot,
                    dependency_reason=dependency_reason,
                )
                diagnostics.append(
                    self._diagnostic(record, "dependency.not_satisfied")
                )
                records.append(record)
                record_by_id[invocation.invocation_id] = record
                continue

            evidence_requirement = evidence_by_invocation.get(
                invocation.invocation_id
            )
            if (
                evidence_requirement is not None
                and not evidence_requirement.available
            ):
                record = self._record(
                    invocation,
                    RuleExecutionStatus.BLOCKED_BY_EVIDENCE,
                    baseline_snapshot,
                    evidence_block_reason=(
                        f"{evidence_requirement.requirement_id}:"
                        f"{evidence_requirement.unavailable_reason}"
                    ),
                )
                diagnostics.append(
                    self._diagnostic(record, "evidence.required_unavailable")
                )
                records.append(record)
                record_by_id[invocation.invocation_id] = record
                continue

            rule = self._registry.get(invocation.rule_id)
            try:
                requirement = self._requirement_validator.validate(rule, facts)
            except Exception as exc:
                failure_type = _stable_failure_type(exc)
                record = self._record(
                    invocation,
                    RuleExecutionStatus.FAILED,
                    baseline_snapshot,
                    failure_type=failure_type,
                    failure_message=(
                        f"Required-fact validation raised {failure_type}."
                    ),
                )
                diagnostics.append(_expected_diagnostic(record))
                records.append(record)
                record_by_id[invocation.invocation_id] = record
                continue
            if not requirement.is_satisfied:
                record = self._record(
                    invocation,
                    RuleExecutionStatus.MISSING_FACTS,
                    baseline_snapshot,
                    missing_fact_paths=tuple(
                        item.fact_path for item in requirement.missing_facts
                    ),
                )
                diagnostics.append(
                    self._diagnostic(record, "facts.required_missing")
                )
                records.append(record)
                record_by_id[invocation.invocation_id] = record
                continue

            try:
                context = AssessmentContext(
                    facts_snapshot=facts,
                    invocation=invocation,
                    prerequisite_findings=prerequisites,
                    authorization=authorization,
                    baseline=baseline_snapshot,
                )
                output = LegacyScreeningRuleAdapter.evaluate(
                    rule,
                    context,
                    plan_snapshot,
                )
                finding = self._prepare_finding(rule, invocation, output)
            except Exception as exc:  # isolated by invocation by contract
                failure_type = _stable_failure_type(exc)
                record = self._record(
                    invocation,
                    RuleExecutionStatus.FAILED,
                    baseline_snapshot,
                    failure_type=failure_type,
                    failure_message=f"Rule execution raised {failure_type}.",
                )
                diagnostics.append(
                    self._diagnostic(record, "rule.execution_failed")
                )
                records.append(record)
                record_by_id[invocation.invocation_id] = record
                continue

            record = self._record(
                invocation,
                RuleExecutionStatus.COMPLETED,
                baseline_snapshot,
                finding_id=finding.finding_id if finding is not None else None,
            )
            records.append(record)
            record_by_id[invocation.invocation_id] = record
            if finding is not None:
                findings.append(finding)
                finding_by_invocation[invocation.invocation_id] = finding
                associations.append(
                    InvocationFindingAssociation(
                        invocation_id=invocation.invocation_id,
                        finding_id=finding.finding_id,
                    )
                )

        return PlannedAssessmentResult(
            fingerprint_input=fingerprint_input,
            execution_records=records,
            findings=findings,
            finding_associations=associations,
            diagnostics=diagnostics,
        )

    def _validate_registry(self, plan: RulesetPlan) -> None:
        try:
            registry_plan = self._registry.build_ruleset_plan()
        except Exception as exc:
            raise PlannedExecutionInputError(
                "registered rules cannot reproduce the supplied RulesetPlan"
            ) from exc
        if (
            registry_plan.dependency_graph_hash != plan.dependency_graph_hash
            or registry_plan.canonical_dict() != plan.canonical_dict()
        ):
            raise PlannedExecutionInputError(
                "RuleRegistry does not match the supplied RulesetPlan"
            )
        for rule in self._registry:
            metadata = plan.metadata_for(rule.rule_id)
            if (
                metadata.dependencies
                and not isinstance(rule, ContextAwareAssessmentRule)
            ):
                raise PlannedExecutionInputError(
                    f"Dependent rule {rule.rule_id!r} must implement "
                    "evaluate_context()"
                )

    @staticmethod
    def _prerequisite_summaries(
        invocation: RuleInvocation,
        records: Mapping[InvocationId, RuleExecutionRecord],
        findings: Mapping[InvocationId, Finding],
    ) -> tuple[tuple[PrerequisiteFindingSummary, ...], str | None]:
        summaries: list[PrerequisiteFindingSummary] = []
        for prerequisite_id in invocation.prerequisite_invocation_ids:
            record = records.get(prerequisite_id)
            if record is None:
                return (), f"missing_prerequisite:{prerequisite_id}"
            if record.status is not RuleExecutionStatus.COMPLETED:
                return (), (
                    f"prerequisite_not_completed:{prerequisite_id}:"
                    f"{record.status.value}"
                )
            finding = findings.get(prerequisite_id)
            if finding is None or record.finding_id is None:
                return (), f"prerequisite_finding_missing:{prerequisite_id}"
            if record.finding_id != finding.finding_id:
                return (), f"prerequisite_finding_mismatch:{prerequisite_id}"
            if (
                finding.rule_id != record.invocation.rule_id
                or finding.rule_version != record.invocation.rule_version
            ):
                return (), f"prerequisite_rule_mismatch:{prerequisite_id}"
            accepted = invocation.accepted_upstream_statuses[
                str(prerequisite_id)
            ]
            if finding.status.value not in accepted:
                return (), f"prerequisite_status_not_accepted:{prerequisite_id}"
            summaries.append(
                PrerequisiteFindingSummary(
                    finding_id=finding.finding_id,
                    prerequisite_invocation_id=prerequisite_id,
                    scope=record.invocation.scope,
                    framework=finding.framework,
                    rule_id=finding.rule_id,
                    rule_version=finding.rule_version,
                    status=finding.status.value,
                    reason_codes=tuple(finding.reason_codes),
                    trace_references=tuple(finding.fact_refs),
                )
            )
        return tuple(summaries), None

    @staticmethod
    def _prepare_finding(
        rule: AssessmentRule,
        invocation: RuleInvocation,
        output: Finding | None,
    ) -> Finding | None:
        if output is None:
            return None
        if not isinstance(output, Finding):
            raise PlannedRuleOutputError(
                "planned rules may return at most one Finding or None"
            )
        if hasattr(output, "scope"):
            claimed_scope = getattr(output, "scope")
            if not isinstance(claimed_scope, AssessmentScope):
                raise PlannedRuleOutputError(
                    "Finding scope claim must be an AssessmentScope"
                )
            if claimed_scope != invocation.scope:
                raise PlannedRuleOutputError(
                    "Finding scope does not match invocation"
                )
        finding = Finding.from_dict(deepcopy(output.to_dict()))
        if finding.category is not rule.category:
            raise PlannedRuleOutputError("Finding category does not match rule")
        if finding.rule_id is not None and finding.rule_id != invocation.rule_id:
            raise PlannedRuleOutputError("Finding rule_id does not match invocation")
        if (
            finding.rule_version is not None
            and finding.rule_version != invocation.rule_version
        ):
            raise PlannedRuleOutputError(
                "Finding rule_version does not match invocation"
            )
        if (
            finding.framework is not RegulatoryFramework.UNKNOWN
            and finding.framework is not rule.framework
        ):
            raise PlannedRuleOutputError("Finding framework does not match rule")
        if finding.assessment_run_id is not None:
            raise PlannedRuleOutputError(
                "F2B1 Findings cannot claim an AssessmentRun association"
            )
        finding.rule_id = invocation.rule_id
        finding.rule_version = invocation.rule_version
        finding.framework = rule.framework
        if not finding.legal_basis:
            finding.legal_basis = deepcopy(list(rule.legal_basis))
        finding.finding_id = (
            "finding:"
            + sha256(str(invocation.invocation_id).encode("utf-8")).hexdigest()[:32]
        )
        return finding

    @staticmethod
    def _record(
        invocation: RuleInvocation,
        status: RuleExecutionStatus,
        baseline: AssessmentBaseline,
        **fields: Any,
    ) -> RuleExecutionRecord:
        return RuleExecutionRecord(
            invocation=invocation,
            status=status,
            ruleset_baseline_id=baseline.ruleset_baseline_id,
            evidence_baseline_id=baseline.legal_source_baseline_id,
            **fields,
        )

    @staticmethod
    def _diagnostic(
        record: RuleExecutionRecord,
        code: str,
    ) -> ExecutionDiagnostic:
        messages = {
            RuleExecutionStatus.NOT_AUTHORIZED: (
                "No matching explicit authorization was supplied."
            ),
            RuleExecutionStatus.BLOCKED_BY_DEPENDENCY: (
                record.dependency_reason or "A declared dependency was not satisfied."
            ),
            RuleExecutionStatus.BLOCKED_BY_EVIDENCE: (
                record.evidence_block_reason
                or "A required Evidence proposition set is unavailable."
            ),
            RuleExecutionStatus.MISSING_FACTS: (
                "Required facts are missing: "
                + ", ".join(record.missing_fact_paths)
            ),
            RuleExecutionStatus.FAILED: (
                record.failure_message or "Rule execution failed."
            ),
        }
        return ExecutionDiagnostic(
            invocation_id=record.invocation.invocation_id,
            code=code,
            message=messages[record.status],
        )


def _validate_planned_execution_inputs(
    *,
    facts_snapshot: AssessmentFacts,
    plan: RulesetPlan,
    invocations: Iterable[RuleInvocation],
    authorizations: Iterable[AuthorizedRuleInvocation],
    baseline: AssessmentBaseline,
    evidence_requirements: Iterable[InvocationEvidenceRequirement],
) -> _ValidatedPlannedExecutionInputs:
    """Snapshot and validate every plan-based pre-execution invariant once."""

    if not isinstance(facts_snapshot, AssessmentFacts):
        raise TypeError("facts_snapshot must be AssessmentFacts")
    if not isinstance(plan, RulesetPlan):
        raise TypeError("plan must be a RulesetPlan")
    if not isinstance(baseline, AssessmentBaseline):
        raise TypeError("baseline must be an AssessmentBaseline")
    facts = AssessmentFacts.from_dict(facts_snapshot.to_dict())
    plan_snapshot = RulesetPlan.build(
        RulePlanningMetadata.from_dict(item.to_dict())
        for item in plan.ordered_rules
    )
    if (
        plan_snapshot.dependency_graph_hash != plan.dependency_graph_hash
        or plan_snapshot.ruleset_baseline_id != plan.ruleset_baseline_id
        or plan_snapshot.canonical_dict() != plan.canonical_dict()
    ):
        raise PlannedExecutionInputError(
            "RulesetPlan derived fields are inconsistent"
        )
    baseline_snapshot = AssessmentBaseline.from_dict(baseline.to_dict())
    invocation_values = tuple(
        RuleInvocation.from_dict(item.to_dict())
        for item in _typed_tuple(
            invocations,
            RuleInvocation,
            field_name="invocations",
        )
    )
    authorization_values = tuple(
        AuthorizedRuleInvocation.from_dict(item.to_dict())
        for item in _typed_tuple(
            authorizations,
            AuthorizedRuleInvocation,
            field_name="authorizations",
        )
    )
    evidence_values = tuple(
        InvocationEvidenceRequirement.from_dict(item.to_dict())
        for item in _typed_tuple(
            evidence_requirements,
            InvocationEvidenceRequirement,
            field_name="evidence_requirements",
        )
    )
    _validate_baseline(facts, plan_snapshot, baseline_snapshot)
    by_id = _validate_invocations(plan_snapshot, invocation_values)
    authorization_map = _validate_authorizations(
        plan_snapshot,
        invocation_values,
        authorization_values,
    )
    evidence_map = _validate_evidence_requirements(by_id, evidence_values)
    position = {
        metadata.rule_id: index
        for index, metadata in enumerate(plan_snapshot.ordered_rules)
    }
    ordered_invocations = tuple(
        sorted(
            invocation_values,
            key=lambda item: _invocation_sort_key(item, position),
        )
    )
    ordered_authorizations = tuple(
        sorted(authorization_values, key=_authorization_sort_key)
    )
    ordered_evidence = tuple(
        sorted(
            evidence_values,
            key=lambda item: (str(item.invocation_id), item.requirement_id),
        )
    )
    return _ValidatedPlannedExecutionInputs(
        facts=facts,
        plan=plan_snapshot,
        ordered_invocations=ordered_invocations,
        ordered_authorizations=ordered_authorizations,
        ordered_evidence_requirements=ordered_evidence,
        authorization_by_invocation=MappingProxyType(dict(authorization_map)),
        evidence_by_invocation=MappingProxyType(dict(evidence_map)),
        baseline=baseline_snapshot,
    )


def _validate_baseline(
    facts: AssessmentFacts,
    plan: RulesetPlan,
    baseline: AssessmentBaseline,
) -> None:
    expected_rules = tuple(
        RuleBaselineEntry(item.rule_id, item.rule_version)
        for item in plan.ordered_rules
    )
    checks = {
        "facts schema": (
            facts.schema_version == AssessmentFacts.V3_SCHEMA_VERSION
            and baseline.facts_schema_version == facts.schema_version
        ),
        "engine version": baseline.engine_version == "3.0.0",
        "questionnaire version": baseline.questionnaire_version == "3.0.0",
        "AssessmentContext version": (
            baseline.assessment_context_version
            == AssessmentContext.CONTRACT_VERSION
        ),
        "execution record version": baseline.execution_record_version == "1.0.0",
        "report schema version": baseline.report_schema_version == "2.0.0",
        "ordered rules": baseline.ordered_rules == expected_rules,
        "plan graph hash": (
            baseline.rule_dependency_graph_hash == plan.dependency_graph_hash
        ),
        "ruleset baseline ID": (
            baseline.ruleset_baseline_id == plan.ruleset_baseline_id
        ),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise PlannedExecutionInputError(
            "inconsistent AssessmentBaseline: " + ", ".join(failed)
        )


def _validate_invocations(
    plan: RulesetPlan,
    invocations: tuple[RuleInvocation, ...],
) -> dict[InvocationId, RuleInvocation]:
    by_id: dict[InvocationId, RuleInvocation] = {}
    identities: set[tuple[str, str, tuple[str | None, ...]]] = set()
    for invocation in invocations:
        if invocation.invocation_id in by_id:
            raise PlannedExecutionInputError("duplicate invocation_id")
        identity = (
            invocation.rule_id,
            invocation.rule_version,
            invocation.scope.canonical_tuple(),
        )
        if identity in identities:
            raise PlannedExecutionInputError("duplicate rule-and-scope invocation")
        identities.add(identity)
        by_id[invocation.invocation_id] = invocation

    positions = {
        item.rule_id: index for index, item in enumerate(plan.ordered_rules)
    }
    for invocation in invocations:
        try:
            metadata = plan.metadata_for(invocation.rule_id)
        except Exception as exc:
            raise PlannedExecutionInputError(
                f"unknown invocation rule {invocation.rule_id!r}"
            ) from exc
        if invocation.rule_version != metadata.rule_version:
            raise PlannedExecutionInputError(
                "invocation rule_version does not match RulesetPlan"
            )
        if invocation.phase is not None and invocation.phase != metadata.phase.value:
            raise PlannedExecutionInputError(
                "invocation phase does not match RulesetPlan"
            )
        if (
            invocation.ordering_key is not None
            and invocation.ordering_key != metadata.ordering_key
        ):
            raise PlannedExecutionInputError(
                "invocation ordering_key does not match RulesetPlan"
            )
        if len(invocation.prerequisite_invocation_ids) != len(
            metadata.dependencies
        ):
            raise PlannedExecutionInputError(
                "invocation prerequisite count does not match RulesetPlan"
            )
        present_dependency_rules: list[str] = []
        for prerequisite_id in invocation.prerequisite_invocation_ids:
            if prerequisite_id == invocation.invocation_id:
                raise PlannedExecutionInputError(
                    "an invocation cannot depend on itself"
                )
            statuses = invocation.accepted_upstream_statuses[str(prerequisite_id)]
            try:
                normalized_statuses = {
                    FindingStatus(item).value for item in statuses
                }
            except ValueError as exc:
                raise PlannedExecutionInputError(
                    "invocation contains an invalid accepted Finding status"
                ) from exc
            prerequisite = by_id.get(prerequisite_id)
            if prerequisite is None:
                continue
            if prerequisite.scope != invocation.scope:
                raise PlannedExecutionInputError(
                    "cross-scope prerequisites are not permitted by F2B1"
                )
            try:
                prerequisite_metadata = plan.metadata_for(prerequisite.rule_id)
            except Exception as exc:
                raise PlannedExecutionInputError(
                    "present prerequisite rule is absent from RulesetPlan"
                ) from exc
            if prerequisite.rule_version != prerequisite_metadata.rule_version:
                raise PlannedExecutionInputError(
                    "prerequisite rule_version does not match RulesetPlan"
                )
            if prerequisite.rule_id not in metadata.dependencies:
                raise PlannedExecutionInputError(
                    "invocation prerequisite rule is not declared by RulesetPlan"
                )
            if positions[prerequisite.rule_id] >= positions[invocation.rule_id]:
                raise PlannedExecutionInputError(
                    "dependency invocation must precede its dependent rule"
                )
            present_dependency_rules.append(prerequisite.rule_id)
            allowed = set(
                metadata.accepted_upstream_statuses.get(
                    prerequisite.rule_id,
                    (),
                )
            )
            if not normalized_statuses.issubset(allowed):
                raise PlannedExecutionInputError(
                    "invocation accepted statuses exceed RulesetPlan metadata"
                )
        if len(set(present_dependency_rules)) != len(present_dependency_rules):
            raise PlannedExecutionInputError(
                "invocation has duplicate prerequisite rule scopes"
            )
    return by_id


def _validate_authorizations(
    plan: RulesetPlan,
    invocations: tuple[RuleInvocation, ...],
    authorizations: tuple[AuthorizedRuleInvocation, ...],
) -> dict[InvocationId, AuthorizedRuleInvocation]:
    by_authorization_id: dict[str, AuthorizedRuleInvocation] = {}
    for authorization in authorizations:
        if authorization.authorization_id in by_authorization_id:
            raise PlannedExecutionInputError("duplicate authorization_id")
        if not authorization.scopes:
            raise PlannedExecutionInputError(
                "selector-only authorization must be expanded before F2B1"
            )
        try:
            metadata = plan.metadata_for(authorization.rule_id)
        except Exception as exc:
            raise PlannedExecutionInputError(
                f"unknown authorization rule {authorization.rule_id!r}"
            ) from exc
        if authorization.rule_version != metadata.rule_version:
            raise PlannedExecutionInputError(
                "authorization rule_version does not match RulesetPlan"
            )
        matching_invocations = {
            invocation.scope
            for invocation in invocations
            if invocation.rule_id == authorization.rule_id
            and invocation.rule_version == authorization.rule_version
        }
        if set(authorization.scopes).difference(matching_invocations):
            raise PlannedExecutionInputError(
                "authorization contains a scope with no supplied invocation"
            )
        by_authorization_id[authorization.authorization_id] = authorization

    matches: dict[InvocationId, AuthorizedRuleInvocation] = {}
    for invocation in invocations:
        candidates = [
            authorization
            for authorization in authorizations
            if authorization.rule_id == invocation.rule_id
            and authorization.rule_version == invocation.rule_version
            and invocation.scope in authorization.scopes
        ]
        if invocation.authorization_reference is not None:
            referenced = by_authorization_id.get(invocation.authorization_reference)
            if referenced is None or referenced not in candidates:
                raise PlannedExecutionInputError(
                    "invocation authorization_reference is inconsistent"
                )
            candidates = [referenced]
        if len(candidates) > 1:
            raise PlannedExecutionInputError(
                "invocation has ambiguous explicit authorizations"
            )
        if candidates:
            matches[invocation.invocation_id] = candidates[0]
    return matches


def _validate_evidence_requirements(
    invocations: Mapping[InvocationId, RuleInvocation],
    requirements: tuple[InvocationEvidenceRequirement, ...],
) -> dict[InvocationId, InvocationEvidenceRequirement]:
    by_invocation: dict[InvocationId, InvocationEvidenceRequirement] = {}
    requirement_ids: set[str] = set()
    for requirement in requirements:
        if requirement.invocation_id not in invocations:
            raise PlannedExecutionInputError(
                "Evidence requirement references an unknown invocation"
            )
        if requirement.invocation_id in by_invocation:
            raise PlannedExecutionInputError(
                "duplicate Evidence requirement for invocation"
            )
        if requirement.requirement_id in requirement_ids:
            raise PlannedExecutionInputError("duplicate Evidence requirement_id")
        requirement_ids.add(requirement.requirement_id)
        by_invocation[requirement.invocation_id] = requirement
    return by_invocation


def _fingerprint_payload(
    validated: _ValidatedPlannedExecutionInputs,
) -> dict[str, Any]:
    baseline = validated.baseline
    return {
        "contract_version": PlannedExecutionFingerprintInput.CONTRACT_VERSION,
        "engine_version": PlannedExecutionFingerprintInput.ENGINE_VERSION,
        "assessment_context_version": AssessmentContext.CONTRACT_VERSION,
        "execution_record_version": baseline.execution_record_version,
        "report_schema_version": baseline.report_schema_version,
        "facts": canonicalize_v3_facts(validated.facts.to_dict()),
        "ruleset_plan": {
            "dependency_graph_hash": validated.plan.dependency_graph_hash,
            "ruleset_baseline_id": validated.plan.ruleset_baseline_id,
            "metadata": validated.plan.canonical_dict(),
        },
        "baseline": baseline.to_dict(),
        "invocations": [
            item.to_dict() for item in validated.ordered_invocations
        ],
        "authorizations": [
            item.to_dict() for item in validated.ordered_authorizations
        ],
        "evidence_requirements": [
            item.to_dict()
            for item in validated.ordered_evidence_requirements
        ],
    }


def _require_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    unknown = sorted(set(payload).difference(expected))
    missing = sorted(expected.difference(payload))
    if unknown:
        raise ValueError(f"unknown {label} fields: " + ", ".join(unknown))
    if missing:
        raise ValueError(f"{label} is missing fields: " + ", ".join(missing))


def _expected_diagnostic(record: RuleExecutionRecord) -> ExecutionDiagnostic:
    codes = {
        RuleExecutionStatus.NOT_AUTHORIZED: "authorization.not_explicit",
        RuleExecutionStatus.BLOCKED_BY_DEPENDENCY: "dependency.not_satisfied",
        RuleExecutionStatus.BLOCKED_BY_EVIDENCE: "evidence.required_unavailable",
        RuleExecutionStatus.MISSING_FACTS: "facts.required_missing",
    }
    if record.status is RuleExecutionStatus.FAILED:
        code = (
            "rule.requirement_validation_failed"
            if (record.failure_message or "").startswith(
                "Required-fact validation raised "
            )
            else "rule.execution_failed"
        )
    else:
        try:
            code = codes[record.status]
        except KeyError as exc:
            raise ValueError("COMPLETED records cannot have diagnostics") from exc
    return PlannedAssessmentEngine._diagnostic(record, code)


def _typed_tuple(
    values: Iterable[Any],
    expected_type: type[Any],
    *,
    field_name: str,
) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError(f"{field_name} must be an iterable of typed contracts")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be iterable") from exc
    if any(not isinstance(item, expected_type) for item in result):
        raise TypeError(
            f"{field_name} must contain {expected_type.__name__} instances"
        )
    return result


def _reject_duplicate_values(
    values: Iterable[str],
    *,
    field_name: str,
) -> None:
    materialized = tuple(values)
    if len(set(materialized)) != len(materialized):
        raise PlannedExecutionInputError(f"duplicate {field_name}")


def _invocation_sort_key(
    invocation: RuleInvocation,
    rule_positions: Mapping[str, int],
) -> tuple[int, str, str, str, str, str]:
    if invocation.rule_id not in rule_positions:
        raise PlannedExecutionInputError(
            f"invocation rule {invocation.rule_id!r} is absent from RulesetPlan"
        )
    scope = tuple(value or "" for value in invocation.scope.canonical_tuple())
    return (rule_positions[invocation.rule_id], *scope, str(invocation.invocation_id))


def _authorization_sort_key(
    authorization: AuthorizedRuleInvocation,
) -> tuple[str, str, tuple[tuple[str, ...], ...], str]:
    scopes = tuple(
        tuple(value or "" for value in scope.canonical_tuple())
        for scope in authorization.scopes
    )
    return (
        authorization.rule_id,
        authorization.rule_version,
        scopes,
        authorization.authorization_id,
    )


def _stable_failure_type(exc: Exception) -> str:
    name = type(exc).__name__
    if not name or not name.replace("_", "").isalnum():
        return "ExecutionError"
    return name
