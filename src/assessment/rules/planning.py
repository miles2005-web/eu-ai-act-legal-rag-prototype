"""Immutable rule-planning contracts for the inactive v0.6 execution path."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Any, ClassVar

from src.assessment.baselines import AssessmentBaseline, RuleBaselineEntry
from src.assessment.findings import FindingStatus
from src.assessment.invocations import RuleInvocation
from src.assessment.models import FrozenDict, SerializableModel


class RulePlanningError(ValueError):
    """Raised when a ruleset cannot form a coherent deterministic plan."""


class RulePhase(str, Enum):
    """Ordered phases approved for the v0.6 recruitment consequence chain."""

    SCREENING = "screening"
    ROLE_RELEVANCE = "role_relevance"
    OBLIGATION_RELEVANCE = "obligation_relevance"
    ARTEFACT_PROJECTION = "artefact_projection"

    @property
    def rank(self) -> int:
        return _PHASE_RANK[self]


_PHASE_RANK = {
    RulePhase.SCREENING: 0,
    RulePhase.ROLE_RELEVANCE: 1,
    RulePhase.OBLIGATION_RELEVANCE: 2,
    RulePhase.ARTEFACT_PROJECTION: 3,
}


def _non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


@dataclass(frozen=True, slots=True)
class RulePlanningMetadata(SerializableModel):
    """Versioned execution-significant metadata for one registered rule."""

    rule_id: str
    rule_version: str
    phase: RulePhase
    ordering_key: str
    dependencies: tuple[str, ...] = ()
    accepted_upstream_statuses: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    subject_selector: str = "legacy_case_scope"
    contract_version: str = "1.0.0"

    CONTRACT_VERSION: ClassVar[str] = "1.0.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rule_id",
            _non_empty_string(self.rule_id, field_name="rule_id"),
        )
        object.__setattr__(
            self,
            "rule_version",
            _non_empty_string(
                self.rule_version,
                field_name="rule_version",
            ),
        )
        if isinstance(self.phase, str):
            try:
                object.__setattr__(self, "phase", RulePhase(self.phase))
            except ValueError as exc:
                raise ValueError(
                    f"invalid rule planning phase: {self.phase!r}"
                ) from exc
        if not isinstance(self.phase, RulePhase):
            raise TypeError("phase must be a RulePhase")
        object.__setattr__(
            self,
            "ordering_key",
            _non_empty_string(
                self.ordering_key,
                field_name="ordering_key",
            ),
        )
        object.__setattr__(
            self,
            "subject_selector",
            _non_empty_string(
                self.subject_selector,
                field_name="subject_selector",
            ),
        )
        if self.contract_version != self.CONTRACT_VERSION:
            raise ValueError(
                "unsupported RulePlanningMetadata contract_version"
            )
        if not isinstance(self.dependencies, (list, tuple)):
            raise TypeError("dependencies must be a list or tuple")
        dependencies = tuple(
            _non_empty_string(item, field_name="dependency rule ID")
            for item in self.dependencies
        )
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependencies must not contain duplicates")
        dependencies = tuple(sorted(dependencies))
        object.__setattr__(self, "dependencies", dependencies)

        if not isinstance(self.accepted_upstream_statuses, Mapping):
            raise TypeError(
                "accepted_upstream_statuses must be a mapping"
            )
        normalized_statuses: dict[str, tuple[str, ...]] = {}
        for raw_dependency, raw_statuses in (
            self.accepted_upstream_statuses.items()
        ):
            dependency = _non_empty_string(
                raw_dependency,
                field_name="accepted status dependency",
            )
            if dependency in normalized_statuses:
                raise ValueError(
                    "accepted_upstream_statuses contains duplicate "
                    "normalized dependency IDs"
                )
            if not isinstance(raw_statuses, (list, tuple)):
                raise TypeError(
                    "accepted status collections must be a list or tuple"
                )
            statuses: list[str] = []
            for raw_status in raw_statuses:
                if isinstance(raw_status, FindingStatus):
                    status = raw_status
                elif isinstance(raw_status, str):
                    try:
                        status = FindingStatus(raw_status)
                    except ValueError as exc:
                        raise ValueError(
                            f"invalid accepted Finding status: {raw_status!r}"
                        ) from exc
                else:
                    raise TypeError(
                        "accepted statuses must be FindingStatus values "
                        "or strings"
                    )
                statuses.append(status.value)
            if not statuses:
                raise ValueError(
                    "accepted status collections must not be empty"
                )
            if len(set(statuses)) != len(statuses):
                raise ValueError(
                    "accepted status collections must not contain duplicates"
                )
            normalized_statuses[dependency] = tuple(sorted(statuses))

        declared = set(dependencies)
        configured = set(normalized_statuses)
        undeclared = configured.difference(declared)
        if undeclared:
            raise ValueError(
                "accepted statuses reference undeclared dependencies: "
                f"{sorted(undeclared)!r}"
            )
        missing = declared.difference(configured)
        if missing:
            raise ValueError(
                "every dependency requires accepted upstream statuses: "
                f"{sorted(missing)!r}"
            )
        object.__setattr__(
            self,
            "accepted_upstream_statuses",
            FrozenDict(
                {
                    dependency: normalized_statuses[dependency]
                    for dependency in sorted(normalized_statuses)
                }
            ),
        )

    def canonical_dict(self) -> dict[str, Any]:
        """Return only execution-significant, language-neutral metadata."""

        return {
            "contract_version": self.contract_version,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "phase": self.phase.value,
            "ordering_key": self.ordering_key,
            "dependencies": list(self.dependencies),
            "accepted_upstream_statuses": {
                dependency: list(self.accepted_upstream_statuses[dependency])
                for dependency in self.dependencies
            },
            "subject_selector": self.subject_selector,
        }


@dataclass(frozen=True, slots=True)
class _DerivedRulesetPlan:
    """Canonical fields derived from validated planning metadata."""

    ordered_rules: tuple[RulePlanningMetadata, ...]
    dependencies: FrozenDict[str, tuple[str, ...]]
    reverse_dependencies: FrozenDict[str, tuple[str, ...]]
    dependency_graph_hash: str
    ruleset_baseline_id: str


@dataclass(frozen=True, slots=True)
class RulesetPlan:
    """Validated immutable DAG metadata; it never executes registered rules.

    F2A deliberately exposes no ``from_dict`` hydration API. Both supported
    construction paths, ``build`` and the public dataclass constructor, enforce
    the same strict derivation boundary.
    """

    ordered_rules: tuple[RulePlanningMetadata, ...]
    dependencies: FrozenDict[str, tuple[str, ...]]
    reverse_dependencies: FrozenDict[str, tuple[str, ...]]
    dependency_graph_hash: str
    ruleset_baseline_id: str
    metadata_contract_version: str = "1.0.0"

    CONTRACT_VERSION: ClassVar[str] = "1.0.0"

    def __post_init__(self) -> None:
        """Close direct-construction around the same validated derivation path."""

        if not isinstance(self.metadata_contract_version, str):
            raise TypeError("metadata_contract_version must be a string")
        if not self.metadata_contract_version.strip():
            raise ValueError(
                "metadata_contract_version must be a non-empty string"
            )
        if self.metadata_contract_version != self.CONTRACT_VERSION:
            raise ValueError(
                "unsupported RulesetPlan metadata_contract_version"
            )
        if not isinstance(self.ordered_rules, (list, tuple)):
            raise TypeError("ordered_rules must be a list or tuple")
        ordered_rules = tuple(self.ordered_rules)
        if not ordered_rules:
            raise RulePlanningError(
                "a ruleset plan requires at least one registered rule"
            )
        if any(
            not isinstance(item, RulePlanningMetadata)
            for item in ordered_rules
        ):
            raise TypeError(
                "ordered_rules must contain RulePlanningMetadata contracts"
            )

        derived = self._derive(ordered_rules)
        if ordered_rules != derived.ordered_rules:
            raise RulePlanningError(
                "ordered_rules must equal the deterministic topological order"
            )

        dependencies = self._normalize_adjacency_mapping(
            self.dependencies,
            field_name="dependencies",
        )
        if dependencies != derived.dependencies:
            raise RulePlanningError(
                "dependencies must exactly match rule planning metadata"
            )
        reverse_dependencies = self._normalize_adjacency_mapping(
            self.reverse_dependencies,
            field_name="reverse_dependencies",
        )
        if reverse_dependencies != derived.reverse_dependencies:
            raise RulePlanningError(
                "reverse_dependencies must exactly match derived graph edges"
            )

        if not isinstance(self.dependency_graph_hash, str):
            raise TypeError("dependency_graph_hash must be a string")
        if re.fullmatch(r"[0-9a-f]{64}", self.dependency_graph_hash) is None:
            raise ValueError(
                "dependency_graph_hash must be 64 lowercase hexadecimal "
                "characters"
            )
        if self.dependency_graph_hash != derived.dependency_graph_hash:
            raise RulePlanningError(
                "dependency_graph_hash does not match canonical graph metadata"
            )
        if not isinstance(self.ruleset_baseline_id, str):
            raise TypeError("ruleset_baseline_id must be a string")
        if self.ruleset_baseline_id != derived.ruleset_baseline_id:
            raise RulePlanningError(
                "ruleset_baseline_id must equal 'ruleset:' plus the graph hash"
            )

        object.__setattr__(self, "ordered_rules", derived.ordered_rules)
        object.__setattr__(self, "dependencies", derived.dependencies)
        object.__setattr__(
            self,
            "reverse_dependencies",
            derived.reverse_dependencies,
        )
        object.__setattr__(
            self,
            "dependency_graph_hash",
            derived.dependency_graph_hash,
        )
        object.__setattr__(
            self,
            "ruleset_baseline_id",
            derived.ruleset_baseline_id,
        )

    @classmethod
    def build(
        cls,
        metadata: Iterable[RulePlanningMetadata],
    ) -> RulesetPlan:
        if isinstance(metadata, (str, bytes)):
            raise TypeError(
                "rule planning metadata must be an iterable of contracts"
            )
        entries = tuple(metadata)
        if any(
            not isinstance(item, RulePlanningMetadata)
            for item in entries
        ):
            raise TypeError(
                "ruleset planning requires RulePlanningMetadata contracts"
            )
        derived = cls._derive(entries)
        return cls(
            ordered_rules=derived.ordered_rules,
            dependencies=derived.dependencies,
            reverse_dependencies=derived.reverse_dependencies,
            dependency_graph_hash=derived.dependency_graph_hash,
            ruleset_baseline_id=derived.ruleset_baseline_id,
            metadata_contract_version=cls.CONTRACT_VERSION,
        )

    @classmethod
    def _derive(
        cls,
        entries: tuple[RulePlanningMetadata, ...],
    ) -> _DerivedRulesetPlan:
        """Validate metadata and derive every execution-significant field."""

        if not entries:
            raise RulePlanningError(
                "a ruleset plan requires at least one registered rule"
            )
        by_id: dict[str, RulePlanningMetadata] = {}
        for item in entries:
            if item.rule_id in by_id:
                raise RulePlanningError(
                    f"duplicate rule ID in ruleset plan: {item.rule_id!r}"
                )
            by_id[item.rule_id] = item

        dependencies = {
            rule_id: tuple(by_id[rule_id].dependencies)
            for rule_id in sorted(by_id)
        }
        for rule_id, declared in dependencies.items():
            missing = sorted(set(declared).difference(by_id))
            if missing:
                raise RulePlanningError(
                    f"Rule {rule_id!r} declares missing dependencies: "
                    + ", ".join(missing)
                )
            if rule_id in declared:
                raise RulePlanningError(
                    f"Rule {rule_id!r} cannot depend on itself"
                )

        cls._reject_cycles(dependencies)
        for dependent_id, declared in dependencies.items():
            dependent = by_id[dependent_id]
            for prerequisite_id in declared:
                prerequisite = by_id[prerequisite_id]
                if prerequisite.phase is dependent.phase:
                    raise RulePlanningError(
                        "same-phase dependencies are not supported in F2A: "
                        f"{prerequisite_id!r} -> {dependent_id!r}"
                    )
                if prerequisite.phase.rank > dependent.phase.rank:
                    raise RulePlanningError(
                        "a dependent rule cannot precede its prerequisite "
                        "phase: "
                        f"{prerequisite_id!r} -> {dependent_id!r}"
                    )

        reverse: dict[str, list[str]] = {
            rule_id: [] for rule_id in by_id
        }
        for dependent_id, declared in dependencies.items():
            for prerequisite_id in declared:
                reverse[prerequisite_id].append(dependent_id)
        reverse_dependencies = {
            rule_id: tuple(sorted(reverse[rule_id]))
            for rule_id in sorted(reverse)
        }

        ordered = cls._topological_order(
            by_id,
            dependencies,
            reverse_dependencies,
        )
        graph_payload = cls._canonical_graph_payload(
            ordered,
            metadata_contract_version=cls.CONTRACT_VERSION,
        )
        canonical = json.dumps(
            graph_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        graph_hash = sha256(canonical.encode("utf-8")).hexdigest()
        baseline_id = f"ruleset:{graph_hash}"
        return _DerivedRulesetPlan(
            ordered_rules=ordered,
            dependencies=FrozenDict(dependencies),
            reverse_dependencies=FrozenDict(reverse_dependencies),
            dependency_graph_hash=graph_hash,
            ruleset_baseline_id=baseline_id,
        )

    @staticmethod
    def _normalize_adjacency_mapping(
        value: object,
        *,
        field_name: str,
    ) -> FrozenDict[str, tuple[str, ...]]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
        normalized: dict[str, tuple[str, ...]] = {}
        for raw_rule_id, raw_edges in value.items():
            rule_id = _non_empty_string(
                raw_rule_id,
                field_name=f"{field_name} rule ID",
            )
            if rule_id in normalized:
                raise ValueError(
                    f"{field_name} contains duplicate normalized rule IDs"
                )
            if not isinstance(raw_edges, (list, tuple)):
                raise TypeError(
                    f"{field_name} values must be lists or tuples"
                )
            edges = tuple(
                _non_empty_string(
                    edge,
                    field_name=f"{field_name} dependency ID",
                )
                for edge in raw_edges
            )
            if len(set(edges)) != len(edges):
                raise ValueError(
                    f"{field_name} values must not contain duplicate IDs"
                )
            normalized[rule_id] = tuple(sorted(edges))
        return FrozenDict(
            {
                rule_id: normalized[rule_id]
                for rule_id in sorted(normalized)
            }
        )

    @staticmethod
    def _reject_cycles(
        dependencies: Mapping[str, tuple[str, ...]],
    ) -> None:
        state: dict[str, int] = {rule_id: 0 for rule_id in dependencies}
        stack: list[str] = []

        def visit(rule_id: str) -> None:
            state[rule_id] = 1
            stack.append(rule_id)
            for dependency in dependencies[rule_id]:
                if state[dependency] == 0:
                    visit(dependency)
                elif state[dependency] == 1:
                    start = stack.index(dependency)
                    cycle = stack[start:] + [dependency]
                    raise RulePlanningError(
                        "rule dependency cycle detected: "
                        + " -> ".join(cycle)
                    )
            stack.pop()
            state[rule_id] = 2

        for rule_id in sorted(dependencies):
            if state[rule_id] == 0:
                visit(rule_id)

    @staticmethod
    def _sort_key(
        item: RulePlanningMetadata,
    ) -> tuple[int, str, str, str]:
        return (
            item.phase.rank,
            item.ordering_key,
            item.rule_id,
            item.rule_version,
        )

    @classmethod
    def _topological_order(
        cls,
        by_id: Mapping[str, RulePlanningMetadata],
        dependencies: Mapping[str, tuple[str, ...]],
        reverse_dependencies: Mapping[str, tuple[str, ...]],
    ) -> tuple[RulePlanningMetadata, ...]:
        indegree = {
            rule_id: len(dependencies[rule_id]) for rule_id in by_id
        }
        ready = sorted(
            (
                by_id[rule_id]
                for rule_id, count in indegree.items()
                if count == 0
            ),
            key=cls._sort_key,
        )
        ordered: list[RulePlanningMetadata] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for dependent_id in reverse_dependencies[current.rule_id]:
                indegree[dependent_id] -= 1
                if indegree[dependent_id] == 0:
                    ready.append(by_id[dependent_id])
                    ready.sort(key=cls._sort_key)
        if len(ordered) != len(by_id):
            raise RulePlanningError(
                "ruleset topological planning did not consume every rule"
            )
        return tuple(ordered)

    @staticmethod
    def _canonical_graph_payload(
        ordered_rules: tuple[RulePlanningMetadata, ...],
        *,
        metadata_contract_version: str,
    ) -> dict[str, Any]:
        return {
            "metadata_contract_version": metadata_contract_version,
            "ordered_rules": [
                item.canonical_dict() for item in ordered_rules
            ],
        }

    def canonical_dict(self) -> dict[str, Any]:
        return self._canonical_graph_payload(
            self.ordered_rules,
            metadata_contract_version=self.metadata_contract_version,
        )

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def build_assessment_baseline(
        self,
        *,
        engine_version: str = "3.0.0",
    ) -> AssessmentBaseline:
        engine_version = _non_empty_string(
            engine_version,
            field_name="engine_version",
        )
        return AssessmentBaseline(
            engine_version=engine_version,
            ordered_rules=tuple(
                RuleBaselineEntry(item.rule_id, item.rule_version)
                for item in self.ordered_rules
            ),
            rule_dependency_graph_hash=self.dependency_graph_hash,
            ruleset_baseline_id=self.ruleset_baseline_id,
            evidence_packs=(),
            legal_source_baseline_id=None,
        )

    def metadata_for(self, rule_id: str) -> RulePlanningMetadata:
        for item in self.ordered_rules:
            if item.rule_id == rule_id:
                return item
        raise RulePlanningError(
            f"Rule ID {rule_id!r} is not present in the ruleset plan"
        )

    def validate_invocation(
        self,
        invocation: RuleInvocation,
        *,
        prerequisite_invocations: Iterable[RuleInvocation] = (),
    ) -> RulePlanningMetadata:
        """Validate invocation metadata without executing or expanding scope."""

        if not isinstance(invocation, RuleInvocation):
            raise TypeError("invocation must be a RuleInvocation")
        metadata = self.metadata_for(invocation.rule_id)
        self._validate_invocation_identity(invocation, metadata)
        if isinstance(prerequisite_invocations, (str, bytes)):
            raise TypeError(
                "prerequisite_invocations must be an iterable of "
                "RuleInvocation objects"
            )
        prerequisites = tuple(prerequisite_invocations)
        if any(
            not isinstance(item, RuleInvocation) for item in prerequisites
        ):
            raise TypeError(
                "prerequisite_invocations must contain RuleInvocation objects"
            )
        by_invocation_id: dict[str, RuleInvocation] = {}
        for item in prerequisites:
            identifier = str(item.invocation_id)
            if identifier in by_invocation_id:
                raise RulePlanningError(
                    "duplicate prerequisite invocation metadata"
                )
            by_invocation_id[identifier] = item

        declared_ids = {str(item) for item in invocation.prerequisite_invocation_ids}
        supplied_ids = set(by_invocation_id)
        if declared_ids != supplied_ids:
            raise RulePlanningError(
                "prerequisite invocation metadata must exactly match "
                "invocation.prerequisite_invocation_ids"
            )
        dependency_rule_ids: list[str] = []
        for prerequisite_id in sorted(supplied_ids):
            prerequisite = by_invocation_id[prerequisite_id]
            prerequisite_metadata = self.metadata_for(prerequisite.rule_id)
            self._validate_invocation_identity(
                prerequisite,
                prerequisite_metadata,
            )
            dependency_rule_ids.append(prerequisite.rule_id)
            allowed = set(
                metadata.accepted_upstream_statuses.get(
                    prerequisite.rule_id,
                    (),
                )
            )
            requested = set(
                invocation.accepted_upstream_statuses[prerequisite_id]
            )
            if not requested.issubset(allowed):
                raise RulePlanningError(
                    "invocation accepted statuses exceed rule planning "
                    f"metadata for dependency {prerequisite.rule_id!r}"
                )
        if tuple(sorted(dependency_rule_ids)) != metadata.dependencies:
            raise RulePlanningError(
                "invocation prerequisite rules do not match rule planning "
                "dependencies"
            )
        return metadata

    @staticmethod
    def _validate_invocation_identity(
        invocation: RuleInvocation,
        metadata: RulePlanningMetadata,
    ) -> None:
        """Validate rule-owned invocation fields without traversing the DAG."""

        if invocation.rule_version != metadata.rule_version:
            raise RulePlanningError(
                "invocation rule_version does not match ruleset metadata"
            )
        if invocation.phase is not None and invocation.phase != metadata.phase.value:
            raise RulePlanningError(
                "invocation phase does not match ruleset metadata"
            )
        if (
            invocation.ordering_key is not None
            and invocation.ordering_key != metadata.ordering_key
        ):
            raise RulePlanningError(
                "invocation ordering_key does not match ruleset metadata"
            )
