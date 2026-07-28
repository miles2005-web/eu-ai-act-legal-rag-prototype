"""Tests for the inactive v0.6 deterministic ruleset-planning path."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from src.assessment import (
    AssessmentEngine,
    AssessmentFacts,
    AssessmentScope,
    Finding,
    FindingCategory,
    FindingStatus,
    LegalBasis,
    RuleInvocation,
)
from src.assessment.demo.factory import create_assessment_workflow
from src.assessment.rules import (
    AIActHighRiskEmploymentRule,
    AIActHighRiskProductSafetyRule,
    AssessmentRule,
    DuplicateRuleError,
    EUDataActRelevanceRule,
    GDPRArticle22RelevanceRule,
    RulePhase,
    RulePlanningError,
    RulePlanningMetadata,
    RuleRegistry,
    RulesetPlan,
)


class _PlanningRule(AssessmentRule):
    category = FindingCategory.INFORMATION_GAP
    required_fact_paths = ()
    legal_basis = (
        LegalBasis(
            instrument="TEST_INSTRUMENT",
            citation="Reference 1",
            anchor="reference:1",
        ),
    )

    def __init__(
        self,
        rule_id: str,
        *,
        version: str = "1.0.0",
        phase: RulePhase | str = RulePhase.SCREENING,
        ordering_key: str = "100",
        dependencies: tuple[str, ...] = (),
        accepted_statuses: dict[str, tuple[str, ...]] | None = None,
        subject_selector: str = "test_scope",
        display_name: str = "Display label",
    ) -> None:
        self.rule_id = rule_id
        self.version = version
        self.planning_phase = phase
        self.planning_ordering_key = ordering_key
        self.planning_dependencies = dependencies
        self.planning_accepted_upstream_statuses = (
            accepted_statuses or {}
        )
        self.planning_subject_selector = subject_selector
        self.display_name = display_name

    def evaluate(self, facts: AssessmentFacts) -> Finding:
        return Finding(
            category=self.category,
            issue_code=self.rule_id,
            status=FindingStatus.NOT_ASSESSED,
            title=self.display_name,
            summary="Planning test rule.",
            finding_id=f"finding:{self.rule_id.lower()}",
        )


class _ContradictoryPlanningRule(_PlanningRule):
    def planning_metadata(self) -> RulePlanningMetadata:
        return _metadata("DIFFERENT_RULE")


def _metadata(
    rule_id: str,
    *,
    version: str = "1.0.0",
    phase: RulePhase | str = RulePhase.SCREENING,
    ordering_key: str = "100",
    dependencies: tuple[str, ...] = (),
    statuses: dict[str, tuple[str, ...]] | None = None,
    subject_selector: str = "test_scope",
    contract_version: str = "1.0.0",
) -> RulePlanningMetadata:
    return RulePlanningMetadata(
        rule_id=rule_id,
        rule_version=version,
        phase=phase,
        ordering_key=ordering_key,
        dependencies=dependencies,
        accepted_upstream_statuses=statuses or {},
        subject_selector=subject_selector,
        contract_version=contract_version,
    )


def _dependency_pair(
    *,
    accepted_statuses: tuple[str, ...] = (
        FindingStatus.POTENTIALLY_APPLIES.value,
    ),
) -> tuple[RulePlanningMetadata, RulePlanningMetadata]:
    screening = _metadata(
        "SCREEN",
        phase=RulePhase.SCREENING,
        ordering_key="900",
    )
    role = _metadata(
        "ROLE",
        phase=RulePhase.ROLE_RELEVANCE,
        ordering_key="001",
        dependencies=("SCREEN",),
        statuses={"SCREEN": accepted_statuses},
    )
    return screening, role


def _adjacency_from_metadata(
    metadata: tuple[RulePlanningMetadata, ...],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    forward = {
        item.rule_id: list(item.dependencies) for item in metadata
    }
    reverse = {item.rule_id: [] for item in metadata}
    for dependent_id, dependencies in forward.items():
        for dependency_id in dependencies:
            if dependency_id in reverse:
                reverse[dependency_id].append(dependent_id)
    return forward, reverse


def _direct_plan(
    ordered_rules: list[RulePlanningMetadata]
    | tuple[RulePlanningMetadata, ...],
    *,
    dependencies: dict[str, list[str]] | None = None,
    reverse_dependencies: dict[str, list[str]] | None = None,
    dependency_graph_hash: object = "0" * 64,
    ruleset_baseline_id: object = "ruleset:" + ("0" * 64),
    metadata_contract_version: object = "1.0.0",
) -> RulesetPlan:
    metadata = tuple(ordered_rules)
    derived_forward, derived_reverse = _adjacency_from_metadata(metadata)
    return RulesetPlan(
        ordered_rules=ordered_rules,
        dependencies=(
            derived_forward if dependencies is None else dependencies
        ),
        reverse_dependencies=(
            derived_reverse
            if reverse_dependencies is None
            else reverse_dependencies
        ),
        dependency_graph_hash=dependency_graph_hash,  # type: ignore[arg-type]
        ruleset_baseline_id=ruleset_baseline_id,  # type: ignore[arg-type]
        metadata_contract_version=metadata_contract_version,  # type: ignore[arg-type]
    )


class RulePlanningMetadataTests(unittest.TestCase):
    def test_valid_screening_and_cross_phase_metadata(self) -> None:
        screening, role = _dependency_pair()

        self.assertEqual(screening.dependencies, ())
        self.assertEqual(role.dependencies, ("SCREEN",))
        self.assertEqual(
            role.accepted_upstream_statuses["SCREEN"],
            ("potentially_applies",),
        )

    def test_invalid_phase_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _metadata("RULE", phase="invalid")

    def test_blank_ordering_key_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _metadata("RULE", ordering_key=" ")

    def test_empty_and_duplicate_dependencies_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _metadata(
                "RULE",
                dependencies=("",),
                statuses={"": ("potentially_applies",)},
            )
        with self.assertRaises(ValueError):
            _metadata(
                "RULE",
                dependencies=("UPSTREAM", "UPSTREAM"),
                statuses={"UPSTREAM": ("potentially_applies",)},
            )

    def test_accepted_status_declarations_must_match_dependencies(self) -> None:
        with self.assertRaises(ValueError):
            _metadata(
                "RULE",
                statuses={"UNDECLARED": ("potentially_applies",)},
            )
        with self.assertRaises(ValueError):
            _metadata("RULE", dependencies=("UPSTREAM",))

    def test_raw_status_string_and_unsupported_version_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            _metadata(
                "RULE",
                dependencies=("UPSTREAM",),
                statuses={"UPSTREAM": "potentially_applies"},  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            _metadata("RULE", contract_version="2.0.0")

    def test_duplicate_accepted_status_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _metadata(
                "RULE",
                dependencies=("UPSTREAM",),
                statuses={
                    "UPSTREAM": (
                        "potentially_applies",
                        "potentially_applies",
                    )
                },
            )

    def test_metadata_is_immutable_and_serializable(self) -> None:
        _, role = _dependency_pair()

        with self.assertRaises(FrozenInstanceError):
            role.ordering_key = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            role.accepted_upstream_statuses["SCREEN"] = ()  # type: ignore[index]
        self.assertEqual(
            RulePlanningMetadata.from_dict(role.to_dict()),
            role,
        )


class RuleRegistryGraphValidationTests(unittest.TestCase):
    def test_duplicate_rule_registration_is_rejected(self) -> None:
        registry = RuleRegistry([_PlanningRule("DUPLICATE")])

        with self.assertRaises(DuplicateRuleError):
            registry.register(_PlanningRule("DUPLICATE", version="2.0.0"))

    def test_planning_identity_must_match_registered_rule(self) -> None:
        with self.assertRaisesRegex(RulePlanningError, "identity"):
            RuleRegistry([_ContradictoryPlanningRule("REGISTERED_RULE")])

    def test_missing_and_self_dependencies_are_rejected_by_planning(self) -> None:
        missing = RuleRegistry(
            [
                _PlanningRule(
                    "DEPENDENT",
                    phase=RulePhase.ROLE_RELEVANCE,
                    dependencies=("ABSENT",),
                    accepted_statuses={
                        "ABSENT": ("potentially_applies",)
                    },
                )
            ]
        )
        with self.assertRaisesRegex(RulePlanningError, "missing dependencies"):
            missing.build_ruleset_plan()

        self_dependency = RuleRegistry(
            [
                _PlanningRule(
                    "SELF",
                    phase=RulePhase.ROLE_RELEVANCE,
                    dependencies=("SELF",),
                    accepted_statuses={
                        "SELF": ("potentially_applies",)
                    },
                )
            ]
        )
        with self.assertRaisesRegex(RulePlanningError, "depend on itself"):
            self_dependency.build_ruleset_plan()

    def test_direct_and_indirect_cycles_are_rejected(self) -> None:
        direct = RuleRegistry(
            [
                _PlanningRule(
                    "A",
                    phase=RulePhase.SCREENING,
                    dependencies=("B",),
                    accepted_statuses={"B": ("potentially_applies",)},
                ),
                _PlanningRule(
                    "B",
                    phase=RulePhase.ROLE_RELEVANCE,
                    dependencies=("A",),
                    accepted_statuses={"A": ("potentially_applies",)},
                ),
            ]
        )
        with self.assertRaisesRegex(RulePlanningError, "cycle"):
            direct.build_ruleset_plan()

        indirect = RuleRegistry(
            [
                _PlanningRule(
                    "A",
                    phase=RulePhase.SCREENING,
                    dependencies=("C",),
                    accepted_statuses={"C": ("potentially_applies",)},
                ),
                _PlanningRule(
                    "B",
                    phase=RulePhase.ROLE_RELEVANCE,
                    dependencies=("A",),
                    accepted_statuses={"A": ("potentially_applies",)},
                ),
                _PlanningRule(
                    "C",
                    phase=RulePhase.OBLIGATION_RELEVANCE,
                    dependencies=("B",),
                    accepted_statuses={"B": ("potentially_applies",)},
                ),
            ]
        )
        with self.assertRaisesRegex(RulePlanningError, "cycle"):
            indirect.build_ruleset_plan()

    def test_later_phase_and_same_phase_dependencies_are_rejected(self) -> None:
        later = RuleRegistry(
            [
                _PlanningRule(
                    "LATER",
                    phase=RulePhase.ROLE_RELEVANCE,
                ),
                _PlanningRule(
                    "EARLIER",
                    phase=RulePhase.SCREENING,
                    dependencies=("LATER",),
                    accepted_statuses={
                        "LATER": ("potentially_applies",)
                    },
                ),
            ]
        )
        with self.assertRaisesRegex(
            RulePlanningError,
            "cannot precede its prerequisite",
        ):
            later.build_ruleset_plan()

        same_phase = RuleRegistry(
            [
                _PlanningRule("FIRST"),
                _PlanningRule(
                    "SECOND",
                    dependencies=("FIRST",),
                    accepted_statuses={
                        "FIRST": ("potentially_applies",)
                    },
                ),
            ]
        )
        with self.assertRaisesRegex(
            RulePlanningError,
            "same-phase dependencies",
        ):
            same_phase.build_ruleset_plan()


class DeterministicRulesetPlanTests(unittest.TestCase):
    def test_registration_order_does_not_affect_plan_or_hash(self) -> None:
        rules = [
            _PlanningRule("B", ordering_key="100"),
            _PlanningRule("A", ordering_key="100"),
            _PlanningRule("C", ordering_key="200"),
        ]
        forward = RuleRegistry(rules).build_ruleset_plan()
        reverse = RuleRegistry(reversed(rules)).build_ruleset_plan()

        self.assertEqual(
            tuple(item.rule_id for item in forward.ordered_rules),
            ("A", "B", "C"),
        )
        self.assertEqual(forward.ordered_rules, reverse.ordered_rules)
        self.assertEqual(
            forward.dependency_graph_hash,
            reverse.dependency_graph_hash,
        )
        self.assertEqual(forward.canonical_json(), reverse.canonical_json())

    def test_dependency_precedes_lower_ordering_key_dependent(self) -> None:
        screening, role = _dependency_pair()
        plan = RulesetPlan.build((role, screening))

        self.assertEqual(
            tuple(item.rule_id for item in plan.ordered_rules),
            ("SCREEN", "ROLE"),
        )
        self.assertEqual(plan.dependencies["ROLE"], ("SCREEN",))
        self.assertEqual(plan.reverse_dependencies["SCREEN"], ("ROLE",))

    def test_plan_is_immutable_and_repeated_construction_is_stable(self) -> None:
        plan = RulesetPlan.build(_dependency_pair())
        repeated = RulesetPlan.build(_dependency_pair())

        with self.assertRaises(FrozenInstanceError):
            plan.ruleset_baseline_id = "changed"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            plan.dependencies["ROLE"] = ()  # type: ignore[index]
        self.assertEqual(plan.canonical_json(), repeated.canonical_json())
        self.assertEqual(plan.dependency_graph_hash, repeated.dependency_graph_hash)

    def test_registry_captures_metadata_snapshot_at_registration(self) -> None:
        rule = _PlanningRule("RULE", ordering_key="original")
        registry = RuleRegistry([rule])
        rule.planning_ordering_key = "mutated-after-registration"

        self.assertEqual(
            registry.build_ruleset_plan().ordered_rules[0].ordering_key,
            "original",
        )

    def test_execution_significant_changes_alter_hash_or_baseline(self) -> None:
        screening, role = _dependency_pair()
        linked = RulesetPlan.build((screening, role))
        unlinked = RulesetPlan.build(
            (
                screening,
                _metadata(
                    "ROLE",
                    phase=RulePhase.ROLE_RELEVANCE,
                    ordering_key="001",
                ),
            )
        )
        different_status = RulesetPlan.build(
            _dependency_pair(
                accepted_statuses=(FindingStatus.APPLIES.value,)
            )
        )
        different_phase = RulesetPlan.build(
            (
                screening,
                _metadata(
                    "ROLE",
                    phase=RulePhase.OBLIGATION_RELEVANCE,
                    ordering_key="001",
                    dependencies=("SCREEN",),
                    statuses={
                        "SCREEN": ("potentially_applies",)
                    },
                ),
            )
        )
        different_version = RulesetPlan.build(
            (
                screening,
                _metadata(
                    "ROLE",
                    version="2.0.0",
                    phase=RulePhase.ROLE_RELEVANCE,
                    ordering_key="001",
                    dependencies=("SCREEN",),
                    statuses={
                        "SCREEN": ("potentially_applies",)
                    },
                ),
            )
        )

        for changed in (unlinked, different_status, different_phase):
            self.assertNotEqual(
                linked.dependency_graph_hash,
                changed.dependency_graph_hash,
            )
        self.assertNotEqual(
            linked.ruleset_baseline_id,
            different_version.ruleset_baseline_id,
        )

    def test_display_only_metadata_does_not_affect_hash(self) -> None:
        first = RuleRegistry(
            [_PlanningRule("RULE", display_name="English")]
        ).build_ruleset_plan()
        second = RuleRegistry(
            [_PlanningRule("RULE", display_name="中文")]
        ).build_ruleset_plan()

        self.assertEqual(
            first.dependency_graph_hash,
            second.dependency_graph_hash,
        )

    def test_assessment_baseline_contains_only_available_plan_metadata(self) -> None:
        plan = RulesetPlan.build(_dependency_pair())
        baseline = plan.build_assessment_baseline(
            engine_version="3.0.0-f2a"
        )

        self.assertEqual(baseline.engine_version, "3.0.0-f2a")
        self.assertEqual(
            tuple(item.rule_id for item in baseline.ordered_rules),
            ("SCREEN", "ROLE"),
        )
        self.assertEqual(
            baseline.rule_dependency_graph_hash,
            plan.dependency_graph_hash,
        )
        self.assertEqual(
            baseline.ruleset_baseline_id,
            plan.ruleset_baseline_id,
        )
        self.assertEqual(baseline.evidence_packs, ())
        self.assertIsNone(baseline.legal_source_baseline_id)


class RulesetPlanConstructorBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.built = RulesetPlan.build(_dependency_pair())

    def _canonical_direct(
        self,
        **overrides: object,
    ) -> RulesetPlan:
        values: dict[str, object] = {
            "ordered_rules": list(self.built.ordered_rules),
            "dependencies": {
                key: list(value)
                for key, value in self.built.dependencies.items()
            },
            "reverse_dependencies": {
                key: list(value)
                for key, value in self.built.reverse_dependencies.items()
            },
            "dependency_graph_hash": self.built.dependency_graph_hash,
            "ruleset_baseline_id": self.built.ruleset_baseline_id,
            "metadata_contract_version": (
                self.built.metadata_contract_version
            ),
        }
        values.update(overrides)
        return RulesetPlan(**values)  # type: ignore[arg-type]

    def test_direct_constructor_rejects_invalid_contract_versions(self) -> None:
        for value, error_type in (
            ("9.9.9", ValueError),
            ("", ValueError),
            (None, TypeError),
        ):
            with self.subTest(value=value):
                with self.assertRaises(error_type):
                    self._canonical_direct(
                        metadata_contract_version=value
                    )

    def test_direct_constructor_rejects_fabricated_hashes(self) -> None:
        for graph_hash, error_type in (
            ("not-a-hash", ValueError),
            (self.built.dependency_graph_hash.upper(), ValueError),
            ("f" * 64, RulePlanningError),
            (42, TypeError),
        ):
            with self.subTest(graph_hash=graph_hash):
                with self.assertRaises(error_type):
                    self._canonical_direct(
                        dependency_graph_hash=graph_hash
                    )

    def test_direct_constructor_rejects_fabricated_baseline_ids(self) -> None:
        for baseline_id, error_type in (
            ("arbitrary", RulePlanningError),
            ("ruleset:" + ("f" * 64), RulePlanningError),
            (None, TypeError),
        ):
            with self.subTest(baseline_id=baseline_id):
                with self.assertRaises(error_type):
                    self._canonical_direct(
                        ruleset_baseline_id=baseline_id
                    )

    def test_direct_constructor_rejects_inconsistent_dependencies(self) -> None:
        canonical = {
            key: list(value)
            for key, value in self.built.dependencies.items()
        }
        cases = {
            "missing key": {"SCREEN": []},
            "extra key": {**canonical, "EXTRA": []},
            "missing edge": {"SCREEN": [], "ROLE": []},
            "unknown edge": {
                "SCREEN": [],
                "ROLE": ["UNKNOWN"],
            },
        }
        for label, dependencies in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(RulePlanningError):
                    self._canonical_direct(
                        dependencies=dependencies
                    )

    def test_direct_constructor_rejects_inconsistent_reverse_edges(self) -> None:
        canonical = {
            key: list(value)
            for key, value in self.built.reverse_dependencies.items()
        }
        cases = {
            "missing key": {"SCREEN": ["ROLE"]},
            "extra key": {**canonical, "EXTRA": []},
            "missing edge": {"SCREEN": [], "ROLE": []},
            "extra edge": {
                "SCREEN": ["ROLE"],
                "ROLE": ["SCREEN"],
            },
        }
        for label, reverse_dependencies in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(RulePlanningError):
                    self._canonical_direct(
                        reverse_dependencies=reverse_dependencies
                    )

    def test_direct_constructor_rejects_duplicate_adjacency_edges(self) -> None:
        with self.assertRaises(ValueError):
            self._canonical_direct(
                reverse_dependencies={
                    "SCREEN": ["ROLE", "ROLE"],
                    "ROLE": [],
                }
            )

    def test_direct_constructor_rejects_non_topological_sequences(self) -> None:
        with self.assertRaisesRegex(
            RulePlanningError,
            "topological order",
        ):
            self._canonical_direct(
                ordered_rules=list(reversed(self.built.ordered_rules))
            )

        registration_order = (
            _metadata("B", ordering_key="100"),
            _metadata("A", ordering_key="100"),
        )
        with self.assertRaisesRegex(
            RulePlanningError,
            "topological order",
        ):
            _direct_plan(registration_order)

    def test_direct_constructor_revalidates_every_graph_invariant(self) -> None:
        graph_cases = {
            "direct cycle": (
                _metadata(
                    "A",
                    phase=RulePhase.SCREENING,
                    dependencies=("B",),
                    statuses={"B": ("potentially_applies",)},
                ),
                _metadata(
                    "B",
                    phase=RulePhase.ROLE_RELEVANCE,
                    dependencies=("A",),
                    statuses={"A": ("potentially_applies",)},
                ),
            ),
            "indirect cycle": (
                _metadata(
                    "A",
                    phase=RulePhase.SCREENING,
                    dependencies=("C",),
                    statuses={"C": ("potentially_applies",)},
                ),
                _metadata(
                    "B",
                    phase=RulePhase.ROLE_RELEVANCE,
                    dependencies=("A",),
                    statuses={"A": ("potentially_applies",)},
                ),
                _metadata(
                    "C",
                    phase=RulePhase.OBLIGATION_RELEVANCE,
                    dependencies=("B",),
                    statuses={"B": ("potentially_applies",)},
                ),
            ),
            "same phase": (
                _metadata("A"),
                _metadata(
                    "B",
                    dependencies=("A",),
                    statuses={"A": ("potentially_applies",)},
                ),
            ),
            "later prerequisite": (
                _metadata(
                    "ROLE",
                    phase=RulePhase.ROLE_RELEVANCE,
                ),
                _metadata(
                    "SCREEN",
                    phase=RulePhase.SCREENING,
                    dependencies=("ROLE",),
                    statuses={"ROLE": ("potentially_applies",)},
                ),
            ),
        }
        for label, metadata in graph_cases.items():
            with self.subTest(label=label):
                with self.assertRaises(RulePlanningError):
                    _direct_plan(metadata)

        duplicate_ids = (
            _metadata("DUPLICATE"),
            _metadata("DUPLICATE", version="2.0.0"),
        )
        with self.assertRaisesRegex(RulePlanningError, "duplicate"):
            _direct_plan(duplicate_ids)

    def test_empty_plan_and_invalid_container_types_are_rejected(self) -> None:
        with self.assertRaises(RulePlanningError):
            RulesetPlan.build(())
        with self.assertRaises(TypeError):
            self._canonical_direct(ordered_rules="SCREEN,ROLE")
        with self.assertRaises(TypeError):
            self._canonical_direct(dependencies=())
        with self.assertRaises(TypeError):
            self._canonical_direct(reverse_dependencies=())

    def test_valid_direct_construction_matches_build_and_baseline(self) -> None:
        direct = self._canonical_direct()

        self.assertEqual(direct.canonical_json(), self.built.canonical_json())
        self.assertEqual(
            direct.dependency_graph_hash,
            self.built.dependency_graph_hash,
        )
        self.assertEqual(
            direct.ruleset_baseline_id,
            self.built.ruleset_baseline_id,
        )
        baseline = direct.build_assessment_baseline(
            engine_version="3.0.0-direct"
        )
        self.assertEqual(
            tuple(
                (item.rule_id, item.rule_version)
                for item in baseline.ordered_rules
            ),
            tuple(
                (item.rule_id, item.rule_version)
                for item in direct.ordered_rules
            ),
        )
        self.assertEqual(
            baseline.rule_dependency_graph_hash,
            direct.dependency_graph_hash,
        )
        self.assertEqual(
            baseline.ruleset_baseline_id,
            direct.ruleset_baseline_id,
        )
        self.assertEqual(baseline.evidence_packs, ())
        self.assertIsNone(baseline.legal_source_baseline_id)

    def test_direct_construction_defensively_copies_source_containers(self) -> None:
        ordered_rules = list(self.built.ordered_rules)
        dependencies = {
            key: list(value)
            for key, value in self.built.dependencies.items()
        }
        reverse_dependencies = {
            key: list(value)
            for key, value in self.built.reverse_dependencies.items()
        }
        direct = RulesetPlan(
            ordered_rules=ordered_rules,
            dependencies=dependencies,
            reverse_dependencies=reverse_dependencies,
            dependency_graph_hash=self.built.dependency_graph_hash,
            ruleset_baseline_id=self.built.ruleset_baseline_id,
        )
        before = direct.canonical_json()

        ordered_rules.reverse()
        dependencies["ROLE"].clear()
        reverse_dependencies["SCREEN"].clear()

        self.assertEqual(direct.canonical_json(), before)
        self.assertEqual(direct.dependencies["ROLE"], ("SCREEN",))
        self.assertEqual(
            direct.reverse_dependencies["SCREEN"],
            ("ROLE",),
        )

    def test_ruleset_plan_has_no_hydration_api_in_f2a(self) -> None:
        self.assertFalse(hasattr(RulesetPlan, "from_dict"))


class RuleInvocationCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = RulesetPlan.build(_dependency_pair())
        self.scope = AssessmentScope(system_id="system:test")
        self.screening = RuleInvocation.create(
            rule_id="SCREEN",
            rule_version="1.0.0",
            scope=self.scope,
            phase=RulePhase.SCREENING.value,
            ordering_key="900",
        )

    def _role_invocation(
        self,
        *,
        version: str = "1.0.0",
        statuses: tuple[str, ...] = ("potentially_applies",),
        include_dependency: bool = True,
    ) -> RuleInvocation:
        if not include_dependency:
            return RuleInvocation.create(
                rule_id="ROLE",
                rule_version=version,
                scope=self.scope,
            )
        return RuleInvocation.create(
            rule_id="ROLE",
            rule_version=version,
            scope=self.scope,
            phase=RulePhase.ROLE_RELEVANCE.value,
            ordering_key="001",
            prerequisite_invocation_ids=(
                self.screening.invocation_id,
            ),
            accepted_upstream_statuses={
                self.screening.invocation_id: statuses
            },
        )

    def test_matching_invocations_are_accepted_and_phase_is_resolved(self) -> None:
        metadata = self.plan.validate_invocation(self.screening)
        role_metadata = self.plan.validate_invocation(
            self._role_invocation(),
            prerequisite_invocations=(self.screening,),
        )

        self.assertEqual(metadata.phase, RulePhase.SCREENING)
        self.assertEqual(role_metadata.phase, RulePhase.ROLE_RELEVANCE)

    def test_unknown_rule_and_version_mismatch_are_rejected(self) -> None:
        unknown = RuleInvocation.create(
            rule_id="UNKNOWN",
            rule_version="1.0.0",
            scope=self.scope,
        )
        mismatch = self._role_invocation(version="2.0.0")

        with self.assertRaises(RulePlanningError):
            self.plan.validate_invocation(unknown)
        with self.assertRaisesRegex(RulePlanningError, "rule_version"):
            self.plan.validate_invocation(
                mismatch,
                prerequisite_invocations=(self.screening,),
            )

    def test_invocation_and_prerequisite_phase_mismatches_are_rejected(self) -> None:
        wrong_role_phase = RuleInvocation.create(
            rule_id="ROLE",
            rule_version="1.0.0",
            scope=self.scope,
            phase=RulePhase.OBLIGATION_RELEVANCE.value,
            prerequisite_invocation_ids=(
                self.screening.invocation_id,
            ),
            accepted_upstream_statuses={
                self.screening.invocation_id: ("potentially_applies",)
            },
        )
        wrong_screening_phase = RuleInvocation.create(
            rule_id="SCREEN",
            rule_version="1.0.0",
            scope=self.scope,
            phase=RulePhase.ROLE_RELEVANCE.value,
        )

        with self.assertRaisesRegex(RulePlanningError, "phase"):
            self.plan.validate_invocation(
                wrong_role_phase,
                prerequisite_invocations=(self.screening,),
            )
        with self.assertRaisesRegex(RulePlanningError, "phase"):
            self.plan.validate_invocation(
                self._role_invocation(),
                prerequisite_invocations=(wrong_screening_phase,),
            )

    def test_incompatible_dependency_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            RulePlanningError,
            "prerequisite rules",
        ):
            self.plan.validate_invocation(
                self._role_invocation(include_dependency=False)
            )

    def test_status_subset_is_compatible_but_excess_status_is_rejected(self) -> None:
        self.plan.validate_invocation(
            self._role_invocation(),
            prerequisite_invocations=(self.screening,),
        )
        with self.assertRaisesRegex(
            RulePlanningError,
            "accepted statuses exceed",
        ):
            self.plan.validate_invocation(
                self._role_invocation(
                    statuses=(
                        "potentially_applies",
                        "does_not_apply",
                    )
                ),
                prerequisite_invocations=(self.screening,),
            )


class LegacyPlanningCompatibilityTests(unittest.TestCase):
    def test_current_rules_keep_versions_phase_dependencies_and_order(self) -> None:
        rules = (
            AIActHighRiskEmploymentRule(),
            AIActHighRiskProductSafetyRule(),
            GDPRArticle22RelevanceRule(),
            EUDataActRelevanceRule(),
        )
        registry = RuleRegistry(rules)
        plan = registry.build_ruleset_plan()

        self.assertEqual(
            registry.ids(),
            (
                "AI_ACT_HIGH_RISK_EMPLOYMENT",
                "AI_ACT_HIGH_RISK_PRODUCT_SAFETY",
                "GDPR_ARTICLE22_RELEVANCE",
                "EU_DATA_ACT_RELEVANCE",
            ),
        )
        self.assertTrue(
            all(item.rule_version == "2026.1" for item in plan.ordered_rules)
        )
        self.assertTrue(
            all(item.phase is RulePhase.SCREENING for item in plan.ordered_rules)
        )
        self.assertTrue(
            all(not item.dependencies for item in plan.ordered_rules)
        )
        self.assertEqual(
            tuple(item.rule_id for item in plan.ordered_rules),
            registry.ids(),
        )

    def test_engine_still_executes_registration_order_not_planned_order(self) -> None:
        registry = RuleRegistry(
            [
                _PlanningRule("B", ordering_key="200"),
                _PlanningRule("A", ordering_key="100"),
            ]
        )
        plan = registry.build_ruleset_plan()
        result = AssessmentEngine(registry).run(AssessmentFacts())

        self.assertEqual(
            tuple(item.rule_id for item in plan.ordered_rules),
            ("A", "B"),
        )
        self.assertEqual(result.executed_rule_ids, ["B", "A"])

    def test_factory_composition_and_legacy_engine_contract_are_unchanged(self) -> None:
        bundle = create_assessment_workflow()

        self.assertEqual(
            bundle.engine.registered_rule_ids,
            bundle.rule_registry.ids(),
        )
        self.assertEqual(bundle.engine.engine_version, "2.0.0")


if __name__ == "__main__":
    unittest.main()
