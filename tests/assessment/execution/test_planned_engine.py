"""Focused tests for the isolated v0.6 F2B1 planned execution path."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from src.assessment import (
    AssessmentBaseline,
    AssessmentContext,
    AssessmentEngine,
    AssessmentFacts,
    AssessmentFactsCompatibilityAdapter,
    AssessmentScope,
    AuthorizedRuleInvocation,
    ExecutionDiagnostic,
    EvidencePackBaseline,
    FactRequirementValidator,
    Finding,
    FindingCategory,
    FindingStatus,
    InvocationEvidenceRequirement,
    InvocationFindingAssociation,
    LegalBasis,
    PlannedAssessmentEngine,
    PlannedAssessmentResult,
    PlannedExecutionFingerprintInput,
    PlannedExecutionInputError,
    RegulatoryFramework,
    RoleHypothesis,
    RuleExecutionRecord,
    RuleExecutionStatus,
    RuleInvocation,
    RulesetPlan,
    TriState,
)
from src.assessment.facts import UseDomain
from src.assessment.product_regulation import load_annex_i_instrument_catalog
from src.assessment.rules import (
    AIActHighRiskEmploymentRule,
    AIActHighRiskProductSafetyRule,
    AssessmentRule,
    EUDataActRelevanceRule,
    GDPRArticle22RelevanceRule,
    RulePhase,
    RuleRegistry,
)


class _ContextRule(AssessmentRule):
    framework = RegulatoryFramework.EU_AI_ACT
    category = FindingCategory.INFORMATION_GAP
    legal_basis = (
        LegalBasis("EU_AI_ACT", "Test reference", "test:reference"),
    )

    def __init__(
        self,
        rule_id: str,
        *,
        phase: RulePhase = RulePhase.SCREENING,
        ordering_key: str = "100",
        dependencies: tuple[str, ...] = (),
        accepted_statuses: dict[str, tuple[str, ...]] | None = None,
        required_paths: tuple[str, ...] = (),
        status: FindingStatus = FindingStatus.POTENTIALLY_APPLIES,
        output: str = "finding",
    ) -> None:
        self.rule_id = rule_id
        self.version = "1.0.0"
        self.planning_phase = phase
        self.planning_ordering_key = ordering_key
        self.planning_dependencies = dependencies
        self.planning_accepted_upstream_statuses = accepted_statuses or {}
        self.planning_subject_selector = "explicit_test_scope"
        self.required_fact_paths = required_paths
        self.status = status
        self.output = output
        self.calls = 0
        self.contexts: list[AssessmentContext] = []

    def evaluate(self, facts: AssessmentFacts) -> Finding:
        raise AssertionError("context-aware rule used legacy evaluate")

    def evaluate_context(self, context: AssessmentContext) -> Finding | None:
        self.calls += 1
        self.contexts.append(context)
        if self.output == "none":
            return None
        if self.output == "raise":
            raise RuntimeError("temporary path /tmp/private at 0xABC123")
        if self.output == "wrong_rule":
            return self._finding(rule_id="OTHER_RULE")
        if self.output == "wrong_version":
            finding = self._finding()
            finding.rule_version = "9.9.9"
            return finding
        if self.output == "wrong_scope":
            finding = self._finding()
            finding.scope = _scope(actor="actor:other")  # type: ignore[attr-defined]
            return finding
        if self.output == "many":
            return [self._finding(), self._finding()]  # type: ignore[return-value]
        if self.output == "role_hypothesis":
            return RoleHypothesis(  # type: ignore[return-value]
                hypothesis_id="hypothesis:test-role",
                framework=self.framework,
                scope=context.invocation.scope,
                hypothesis_type="deployer_like",
            )
        return self._finding()

    def _finding(self, *, rule_id: str | None = None) -> Finding:
        return Finding(
            finding_id=f"finding:source-{self.rule_id.lower()}",
            framework=self.framework,
            category=self.category,
            issue_code=self.rule_id,
            status=self.status,
            title=f"Finding for {self.rule_id}",
            summary="Deterministic test Finding.",
            rule_id=rule_id or self.rule_id,
            rule_version=self.version,
            reason_codes=["TEST_REASON"],
            fact_refs=["case.title"],
        )


class _LegacyCountingRule(AssessmentRule):
    framework = RegulatoryFramework.EU_AI_ACT
    rule_id = "LEGACY_SCREEN"
    version = "1.0.0"
    category = FindingCategory.INFORMATION_GAP
    required_fact_paths: tuple[str, ...] = ()
    legal_basis = (
        LegalBasis("EU_AI_ACT", "Test reference", "test:reference"),
    )
    planning_ordering_key = "010"

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, facts: AssessmentFacts) -> Finding:
        self.calls += 1
        return Finding(
            finding_id="finding:legacy-source",
            framework=self.framework,
            category=self.category,
            issue_code=self.rule_id,
            status=FindingStatus.POTENTIALLY_APPLIES,
            title="Legacy result",
            summary="Legacy facts-only evaluation.",
        )


def _facts() -> AssessmentFacts:
    facts = AssessmentFacts.new_v3()
    facts.case.title = "F2B1 case"
    return facts


def _scope(
    *,
    actor: str = "actor:employer",
    system: str = "system:ranker",
    workflow: str = "workflow:hiring",
    operation: str = "operation:screening",
) -> AssessmentScope:
    return AssessmentScope(
        actor_id=actor,
        system_id=system,
        workflow_id=workflow,
        processing_operation_id=operation,
    )


def _invocation(
    rule: AssessmentRule,
    *,
    scope: AssessmentScope | None = None,
    prerequisites: tuple[RuleInvocation, ...] = (),
    authorization_reference: str | None = None,
) -> RuleInvocation:
    metadata = rule.planning_metadata()
    return RuleInvocation.create(
        rule_id=rule.rule_id,
        rule_version=rule.version,
        scope=scope or _scope(),
        phase=metadata.phase.value,
        ordering_key=metadata.ordering_key,
        prerequisite_invocation_ids=tuple(
            item.invocation_id for item in prerequisites
        ),
        accepted_upstream_statuses={
            item.invocation_id: metadata.accepted_upstream_statuses[item.rule_id]
            for item in prerequisites
        },
        authorization_reference=authorization_reference,
    )


def _authorization(
    invocation: RuleInvocation,
    *,
    authorization_id: str | None = None,
) -> AuthorizedRuleInvocation:
    return AuthorizedRuleInvocation(
        authorization_id=(
            authorization_id
            or f"authorization:{str(invocation.invocation_id).split(':')[-1]}"
        ),
        rule_id=invocation.rule_id,
        rule_version=invocation.rule_version,
        scopes=(invocation.scope,),
        authorization_source="test_confirmation",
    )


def _setup(
    *rules: AssessmentRule,
) -> tuple[RuleRegistry, RulesetPlan, AssessmentBaseline, PlannedAssessmentEngine]:
    registry = RuleRegistry(rules)
    plan = registry.build_ruleset_plan()
    baseline = plan.build_assessment_baseline()
    return registry, plan, baseline, PlannedAssessmentEngine(registry)


def _run(
    engine: PlannedAssessmentEngine,
    plan: RulesetPlan,
    baseline: AssessmentBaseline,
    invocations: list[RuleInvocation] | tuple[RuleInvocation, ...],
    authorizations: list[AuthorizedRuleInvocation]
    | tuple[AuthorizedRuleInvocation, ...],
    *,
    facts: AssessmentFacts | None = None,
    evidence: tuple[InvocationEvidenceRequirement, ...] = (),
) -> PlannedAssessmentResult:
    return engine.run(
        facts_snapshot=facts or _facts(),
        plan=plan,
        invocations=invocations,
        authorizations=authorizations,
        baseline=baseline,
        evidence_requirements=evidence,
    )


class PlannedInputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = _ContextRule("SCREEN")
        self.registry, self.plan, self.baseline, self.engine = _setup(self.rule)
        self.invocation = _invocation(self.rule)
        self.authorization = _authorization(self.invocation)

    def test_facts_and_baseline_schema_mismatch_rejected_before_execution(self) -> None:
        with self.assertRaisesRegex(PlannedExecutionInputError, "facts schema"):
            self.engine.run(
                facts_snapshot=AssessmentFacts(),
                plan=self.plan,
                invocations=(self.invocation,),
                authorizations=(self.authorization,),
                baseline=self.baseline,
            )
        self.assertEqual(self.rule.calls, 0)

    def test_plan_hash_and_baseline_id_mismatch_rejected(self) -> None:
        for baseline in (
            replace(self.baseline, rule_dependency_graph_hash="0" * 64),
            replace(self.baseline, ruleset_baseline_id="ruleset:" + "0" * 64),
        ):
            with self.assertRaisesRegex(PlannedExecutionInputError, "inconsistent"):
                _run(
                    self.engine,
                    self.plan,
                    baseline,
                    [self.invocation],
                    [self.authorization],
                )
        self.assertEqual(self.rule.calls, 0)

    def test_unknown_rule_and_version_mismatch_rejected(self) -> None:
        unknown = RuleInvocation.create(
            rule_id="UNKNOWN",
            rule_version="1.0.0",
            scope=_scope(),
        )
        with self.assertRaisesRegex(PlannedExecutionInputError, "unknown"):
            _run(self.engine, self.plan, self.baseline, [unknown], [])
        changed = self.invocation.to_dict()
        changed["rule_version"] = "2.0.0"
        changed["invocation_id"] = str(
            RuleInvocation.create(
                rule_id="SCREEN",
                rule_version="2.0.0",
                scope=_scope(),
            ).invocation_id
        )
        mismatched = RuleInvocation.from_dict(changed)
        with self.assertRaisesRegex(PlannedExecutionInputError, "rule_version"):
            _run(self.engine, self.plan, self.baseline, [mismatched], [])

    def test_duplicate_invocation_and_rule_scope_rejected(self) -> None:
        with self.assertRaisesRegex(PlannedExecutionInputError, "duplicate"):
            _run(
                self.engine,
                self.plan,
                self.baseline,
                [self.invocation, self.invocation],
                [self.authorization],
            )

    def test_duplicate_authorization_id_rejected(self) -> None:
        with self.assertRaisesRegex(PlannedExecutionInputError, "authorization_id"):
            _run(
                self.engine,
                self.plan,
                self.baseline,
                [self.invocation],
                [self.authorization, self.authorization],
            )

    def test_selector_only_authorization_rejected(self) -> None:
        selector = AuthorizedRuleInvocation(
            authorization_id="authorization:selector",
            rule_id=self.rule.rule_id,
            rule_version=self.rule.version,
            subject_selector="all_actors",
        )
        with self.assertRaisesRegex(PlannedExecutionInputError, "selector-only"):
            _run(
                self.engine,
                self.plan,
                self.baseline,
                [self.invocation],
                [selector],
            )

    def test_authorization_scope_without_invocation_rejected(self) -> None:
        invalid = AuthorizedRuleInvocation(
            authorization_id="authorization:other",
            rule_id=self.rule.rule_id,
            rule_version=self.rule.version,
            scopes=(_scope(actor="actor:other"),),
            authorization_source="test",
        )
        with self.assertRaisesRegex(PlannedExecutionInputError, "no supplied"):
            _run(
                self.engine,
                self.plan,
                self.baseline,
                [self.invocation],
                [invalid],
            )
        self.assertEqual(self.rule.calls, 0)

    def test_inconsistent_authorization_reference_rejected(self) -> None:
        invocation = _invocation(
            self.rule,
            authorization_reference="authorization:missing",
        )
        authorization = _authorization(invocation, authorization_id="authorization:real")
        with self.assertRaisesRegex(PlannedExecutionInputError, "reference"):
            _run(
                self.engine,
                self.plan,
                self.baseline,
                [invocation],
                [authorization],
            )


class PlannedOrderingAndStateTests(unittest.TestCase):
    def test_caller_and_authorization_order_do_not_change_output(self) -> None:
        first = _ContextRule("FIRST", ordering_key="010")
        second = _ContextRule("SECOND", ordering_key="020")
        _, plan, baseline, engine = _setup(second, first)
        invocations = [_invocation(second), _invocation(first)]
        authorizations = [_authorization(item) for item in invocations]
        a = _run(engine, plan, baseline, invocations, authorizations)
        b = _run(
            engine,
            plan,
            baseline,
            list(reversed(invocations)),
            list(reversed(authorizations)),
        )
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(
            [item.rule_id for item in a.ordered_invocations],
            ["FIRST", "SECOND"],
        )

    def test_same_rule_invocations_sort_by_scope_then_invocation_id(self) -> None:
        rule = _ContextRule("SCREEN")
        _, plan, baseline, engine = _setup(rule)
        zulu = _invocation(rule, scope=_scope(actor="actor:zulu"))
        alpha = _invocation(rule, scope=_scope(actor="actor:alpha"))
        result = _run(
            engine,
            plan,
            baseline,
            [zulu, alpha],
            [_authorization(zulu), _authorization(alpha)],
        )
        self.assertEqual(
            [item.scope.actor_id for item in result.ordered_invocations],
            ["actor:alpha", "actor:zulu"],
        )

    def test_completed_with_and_without_finding(self) -> None:
        with_finding = _ContextRule("WITH", ordering_key="010")
        without_finding = _ContextRule("WITHOUT", ordering_key="020", output="none")
        _, plan, baseline, engine = _setup(with_finding, without_finding)
        invocations = [_invocation(without_finding), _invocation(with_finding)]
        result = _run(
            engine,
            plan,
            baseline,
            invocations,
            [_authorization(item) for item in invocations],
        )
        self.assertEqual(
            [item.status for item in result.execution_records],
            [RuleExecutionStatus.COMPLETED, RuleExecutionStatus.COMPLETED],
        )
        self.assertEqual(len(result.findings), 1)
        self.assertIsNone(result.execution_records[1].finding_id)

    def test_not_authorized_never_calls_rule(self) -> None:
        rule = _ContextRule("SCREEN")
        _, plan, baseline, engine = _setup(rule)
        result = _run(engine, plan, baseline, [_invocation(rule)], [])
        self.assertEqual(result.execution_records[0].status, RuleExecutionStatus.NOT_AUTHORIZED)
        self.assertEqual(rule.calls, 0)
        self.assertFalse(result.findings)

    def test_missing_facts_never_calls_rule(self) -> None:
        rule = _ContextRule("SCREEN", required_paths=("use_context.task",))
        _, plan, baseline, engine = _setup(rule)
        invocation = _invocation(rule)
        result = _run(engine, plan, baseline, [invocation], [_authorization(invocation)])
        record = result.execution_records[0]
        self.assertEqual(record.status, RuleExecutionStatus.MISSING_FACTS)
        self.assertEqual(record.missing_fact_paths, ("use_context.task",))
        self.assertEqual(rule.calls, 0)

    def test_explicit_unavailable_evidence_blocks_without_retrieval_inference(self) -> None:
        rule = _ContextRule("SCREEN")
        _, plan, baseline, engine = _setup(rule)
        invocation = _invocation(rule)
        gate = InvocationEvidenceRequirement(
            invocation.invocation_id,
            "evidence:test-set",
            False,
            "not_reviewed",
        )
        result = _run(
            engine,
            plan,
            baseline,
            [invocation],
            [_authorization(invocation)],
            evidence=(gate,),
        )
        self.assertEqual(
            result.execution_records[0].status,
            RuleExecutionStatus.BLOCKED_BY_EVIDENCE,
        )
        self.assertEqual(rule.calls, 0)

    def test_available_legal_evidence_baseline_is_preserved_in_record(self) -> None:
        rule = _ContextRule("SCREEN")
        _, plan, baseline, engine = _setup(rule)
        baseline = replace(
            baseline,
            evidence_packs=(
                EvidencePackBaseline(
                    instrument_id="EU_AI_ACT",
                    pack_version="2026.1",
                    manifest_hash="a" * 64,
                ),
            ),
            legal_source_baseline_id="legal-source:test-pack",
        )
        invocation = _invocation(rule)
        gate = InvocationEvidenceRequirement(
            invocation.invocation_id,
            "evidence:test-set",
            True,
        )
        result = _run(
            engine,
            plan,
            baseline,
            [invocation],
            [_authorization(invocation)],
            evidence=(gate,),
        )
        self.assertEqual(
            result.execution_records[0].evidence_baseline_id,
            "legal-source:test-pack",
        )
        self.assertEqual(
            result.execution_records[0].status,
            RuleExecutionStatus.COMPLETED,
        )

    def test_rule_exception_is_sanitized_and_isolated(self) -> None:
        failing = _ContextRule("FAIL", ordering_key="010", output="raise")
        passing = _ContextRule("PASS", ordering_key="020")
        _, plan, baseline, engine = _setup(failing, passing)
        invocations = [_invocation(failing), _invocation(passing)]
        result = _run(
            engine,
            plan,
            baseline,
            invocations,
            [_authorization(item) for item in invocations],
        )
        failed, completed = result.execution_records
        self.assertEqual(failed.status, RuleExecutionStatus.FAILED)
        self.assertEqual(failed.failure_type, "RuntimeError")
        self.assertEqual(failed.failure_message, "Rule execution raised RuntimeError.")
        self.assertNotIn("/tmp", failed.failure_message)
        self.assertNotIn("0x", failed.failure_message)
        self.assertEqual(completed.status, RuleExecutionStatus.COMPLETED)
        self.assertEqual(passing.calls, 1)

    def test_each_invocation_has_exactly_one_matrix_valid_record(self) -> None:
        complete = _ContextRule("COMPLETE", ordering_key="010")
        missing = _ContextRule(
            "MISSING",
            ordering_key="020",
            required_paths=("use_context.task",),
        )
        _, plan, baseline, engine = _setup(complete, missing)
        invocations = [_invocation(complete), _invocation(missing)]
        result = _run(
            engine,
            plan,
            baseline,
            invocations,
            [_authorization(invocations[0])],
        )
        self.assertEqual(len(result.execution_records), len(invocations))
        for record in result.execution_records:
            RuleExecutionRecord.from_dict(record.to_dict())


class PlannedDependencyTests(unittest.TestCase):
    def _chain(
        self,
        *,
        upstream_status: FindingStatus = FindingStatus.POTENTIALLY_APPLIES,
        upstream_output: str = "finding",
        downstream_accepted: tuple[str, ...] = ("potentially_applies",),
    ) -> tuple[
        _ContextRule,
        _ContextRule,
        RulesetPlan,
        AssessmentBaseline,
        PlannedAssessmentEngine,
        RuleInvocation,
        RuleInvocation,
    ]:
        upstream = _ContextRule(
            "UPSTREAM",
            phase=RulePhase.SCREENING,
            ordering_key="900",
            status=upstream_status,
            output=upstream_output,
        )
        downstream = _ContextRule(
            "DOWNSTREAM",
            phase=RulePhase.ROLE_RELEVANCE,
            ordering_key="001",
            dependencies=("UPSTREAM",),
            accepted_statuses={"UPSTREAM": downstream_accepted},
        )
        _, plan, baseline, engine = _setup(downstream, upstream)
        upstream_invocation = _invocation(upstream)
        downstream_invocation = _invocation(
            downstream,
            prerequisites=(upstream_invocation,),
        )
        return (
            upstream,
            downstream,
            plan,
            baseline,
            engine,
            upstream_invocation,
            downstream_invocation,
        )

    def test_accepted_upstream_finding_builds_exact_context(self) -> None:
        upstream, downstream, plan, baseline, engine, up, down = self._chain()
        result = _run(
            engine,
            plan,
            baseline,
            [down, up],
            [_authorization(down), _authorization(up)],
        )
        self.assertEqual(
            [record.invocation.rule_id for record in result.execution_records],
            ["UPSTREAM", "DOWNSTREAM"],
        )
        self.assertTrue(all(record.status is RuleExecutionStatus.COMPLETED for record in result.execution_records))
        self.assertEqual(downstream.calls, 1)
        summaries = downstream.contexts[0].prerequisite_findings
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].prerequisite_invocation_id, up.invocation_id)
        self.assertEqual(downstream.contexts[0].invocation, down)
        self.assertEqual(
            downstream.contexts[0].authorization,
            _authorization(down),
        )
        self.assertEqual(downstream.contexts[0].baseline, baseline)
        self.assertFalse(hasattr(downstream.contexts[0], "findings"))

    def test_unacceptable_status_blocks_dependent(self) -> None:
        _, downstream, plan, baseline, engine, up, down = self._chain(
            upstream_status=FindingStatus.DOES_NOT_APPLY
        )
        result = _run(
            engine,
            plan,
            baseline,
            [up, down],
            [_authorization(up), _authorization(down)],
        )
        self.assertEqual(
            result.execution_records[1].status,
            RuleExecutionStatus.BLOCKED_BY_DEPENDENCY,
        )
        self.assertEqual(downstream.calls, 0)

    def test_absent_prerequisite_invocation_blocks_dependent(self) -> None:
        _, downstream, plan, baseline, engine, up, down = self._chain()
        result = _run(
            engine,
            plan,
            baseline,
            [down],
            [_authorization(down)],
        )
        self.assertEqual(
            result.execution_records[0].dependency_reason,
            f"missing_prerequisite:{up.invocation_id}",
        )
        self.assertEqual(downstream.calls, 0)

    def test_unauthorized_failed_and_missing_prerequisite_block_dependent(self) -> None:
        cases = (
            ("unauthorized", "finding", (), _facts()),
            ("failed", "raise", ("up",), _facts()),
            ("missing", "finding", ("up",), _facts()),
        )
        for label, output, auth_tokens, facts in cases:
            with self.subTest(label=label):
                up_rule, down_rule, plan, baseline, engine, up, down = self._chain(
                    upstream_output=output
                )
                if label == "missing":
                    up_rule.required_fact_paths = ("use_context.task",)
                authorizations = [_authorization(down)]
                if auth_tokens:
                    authorizations.append(_authorization(up))
                result = _run(
                    engine,
                    plan,
                    baseline,
                    [up, down],
                    authorizations,
                    facts=facts,
                )
                self.assertEqual(
                    result.execution_records[1].status,
                    RuleExecutionStatus.BLOCKED_BY_DEPENDENCY,
                )
                self.assertEqual(down_rule.calls, 0)

    def test_completed_prerequisite_without_finding_blocks_dependent(self) -> None:
        _, downstream, plan, baseline, engine, up, down = self._chain(
            upstream_output="none"
        )
        result = _run(
            engine,
            plan,
            baseline,
            [up, down],
            [_authorization(up), _authorization(down)],
        )
        self.assertIn("finding_missing", result.execution_records[1].dependency_reason)
        self.assertEqual(downstream.calls, 0)

    def test_role_hypothesis_cannot_satisfy_formal_dependency(self) -> None:
        _, downstream, plan, baseline, engine, up, down = self._chain(
            upstream_output="role_hypothesis"
        )
        result = _run(
            engine,
            plan,
            baseline,
            [up, down],
            [_authorization(up), _authorization(down)],
        )
        self.assertEqual(
            result.execution_records[0].status,
            RuleExecutionStatus.FAILED,
        )
        self.assertEqual(
            result.execution_records[1].status,
            RuleExecutionStatus.BLOCKED_BY_DEPENDENCY,
        )
        self.assertEqual(downstream.calls, 0)

    def test_second_execution_never_reuses_stale_upstream_finding(self) -> None:
        upstream, downstream, plan, baseline, engine, up, down = self._chain()
        authorizations = [_authorization(up), _authorization(down)]
        first = _run(engine, plan, baseline, [up, down], authorizations)
        self.assertEqual(first.execution_records[1].status, RuleExecutionStatus.COMPLETED)
        upstream.status = FindingStatus.DOES_NOT_APPLY
        second = _run(engine, plan, baseline, [up, down], authorizations)
        self.assertEqual(
            second.execution_records[1].status,
            RuleExecutionStatus.BLOCKED_BY_DEPENDENCY,
        )
        self.assertEqual(downstream.calls, 1)

    def test_cross_scope_prerequisite_rejected_before_execution(self) -> None:
        upstream, downstream, plan, baseline, engine, up, _ = self._chain()
        down = _invocation(
            downstream,
            scope=_scope(actor="actor:other"),
            prerequisites=(up,),
        )
        with self.assertRaisesRegex(PlannedExecutionInputError, "cross-scope"):
            _run(
                engine,
                plan,
                baseline,
                [up, down],
                [_authorization(up), _authorization(down)],
            )
        self.assertEqual(upstream.calls, 0)

    def test_independent_branch_continues_when_other_branch_fails(self) -> None:
        failing = _ContextRule("FAIL", ordering_key="010", output="raise")
        independent = _ContextRule("INDEPENDENT", ordering_key="020")
        dependent = _ContextRule(
            "DEPENDENT",
            phase=RulePhase.ROLE_RELEVANCE,
            ordering_key="010",
            dependencies=("FAIL",),
            accepted_statuses={"FAIL": ("potentially_applies",)},
        )
        _, plan, baseline, engine = _setup(dependent, independent, failing)
        fail_inv = _invocation(failing)
        independent_inv = _invocation(independent)
        dependent_inv = _invocation(dependent, prerequisites=(fail_inv,))
        invocations = [dependent_inv, independent_inv, fail_inv]
        result = _run(
            engine,
            plan,
            baseline,
            invocations,
            [_authorization(item) for item in invocations],
        )
        statuses = {item.invocation.rule_id: item.status for item in result.execution_records}
        self.assertEqual(statuses["FAIL"], RuleExecutionStatus.FAILED)
        self.assertEqual(statuses["INDEPENDENT"], RuleExecutionStatus.COMPLETED)
        self.assertEqual(statuses["DEPENDENT"], RuleExecutionStatus.BLOCKED_BY_DEPENDENCY)
        self.assertEqual(independent.calls, 1)
        self.assertEqual(dependent.calls, 0)


class FindingAndResultContractTests(unittest.TestCase):
    def test_matching_finding_is_associated_with_invocation_scope(self) -> None:
        rule = _ContextRule("SCREEN")
        _, plan, baseline, engine = _setup(rule)
        invocation = _invocation(rule)
        result = _run(engine, plan, baseline, [invocation], [_authorization(invocation)])
        association = result.finding_associations[0]
        self.assertEqual(association.invocation_id, invocation.invocation_id)
        self.assertEqual(association.finding_id, result.findings[0].finding_id)

    def test_conflicting_rule_version_scope_and_multiple_findings_fail(self) -> None:
        for output in (
            "wrong_rule",
            "wrong_version",
            "wrong_scope",
            "many",
            "role_hypothesis",
        ):
            with self.subTest(output=output):
                rule = _ContextRule("SCREEN", output=output)
                _, plan, baseline, engine = _setup(rule)
                invocation = _invocation(rule)
                result = _run(
                    engine,
                    plan,
                    baseline,
                    [invocation],
                    [_authorization(invocation)],
                )
                self.assertEqual(result.execution_records[0].status, RuleExecutionStatus.FAILED)
                self.assertFalse(result.findings)

    def test_conflicting_invocation_association_scope_is_rejected(self) -> None:
        rule = _ContextRule("SCREEN")
        _, plan, baseline, engine = _setup(rule)
        invocation = _invocation(rule)
        result = _run(engine, plan, baseline, [invocation], [_authorization(invocation)])
        other = _invocation(rule, scope=_scope(actor="actor:other"))
        with self.assertRaisesRegex(ValueError, "producing invocation scope"):
            PlannedAssessmentResult(
                fingerprint_input=PlannedExecutionFingerprintInput.from_canonical_payload(
                    result.fingerprint_payload
                ),
                execution_records=result.execution_records,
                findings=result.findings,
                finding_associations=(
                    InvocationFindingAssociation(
                        other.invocation_id,
                        result.findings[0].finding_id,
                    ),
                ),
                diagnostics=(),
            )

    def test_result_round_trip_and_defensive_collections(self) -> None:
        rule = _ContextRule("SCREEN")
        _, plan, baseline, engine = _setup(rule)
        invocation = _invocation(rule)
        result = _run(engine, plan, baseline, [invocation], [_authorization(invocation)])
        restored = PlannedAssessmentResult.from_dict(result.to_dict())
        self.assertEqual(restored.to_dict(), result.to_dict())
        returned = result.findings[0]
        returned.title = "Mutated"
        self.assertNotEqual(result.findings[0].title, "Mutated")
        self.assertIsInstance(result.execution_records, tuple)
        self.assertIsInstance(result.ordered_invocations, tuple)
        with self.assertRaisesRegex(AttributeError, "immutable"):
            result._input_fingerprint = "0" * 64  # type: ignore[misc]

    def test_caller_facts_and_collections_cannot_mutate_completed_result(self) -> None:
        rule = _ContextRule("SCREEN")
        _, plan, baseline, engine = _setup(rule)
        facts = _facts()
        invocation = _invocation(rule)
        invocations = [invocation]
        authorizations = [_authorization(invocation)]
        result = _run(
            engine,
            plan,
            baseline,
            invocations,
            authorizations,
            facts=facts,
        )
        canonical = result.to_dict()
        facts.case.title = "Mutated after execution"
        invocations.clear()
        authorizations.clear()
        self.assertEqual(result.to_dict(), canonical)

    def test_diagnostics_are_non_legal_and_stable(self) -> None:
        rule = _ContextRule("SCREEN")
        _, plan, baseline, engine = _setup(rule)
        result = _run(engine, plan, baseline, [_invocation(rule)], [])
        self.assertEqual(
            result.diagnostics,
            (
                ExecutionDiagnostic(
                    result.ordered_invocations[0].invocation_id,
                    "authorization.not_explicit",
                    "No matching explicit authorization was supplied.",
                ),
            ),
        )
        self.assertFalse(result.findings)


class PlannedFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = _ContextRule("SCREEN")
        _, self.plan, self.baseline, _ = _setup(self.rule)
        self.first = _invocation(self.rule, scope=_scope(actor="actor:a"))
        self.second = _invocation(self.rule, scope=_scope(actor="actor:b"))
        self.auth_first = _authorization(self.first)
        self.auth_second = _authorization(self.second)

    def _fingerprint(
        self,
        *,
        facts: AssessmentFacts | None = None,
        invocations: tuple[RuleInvocation, ...] | None = None,
        authorizations: tuple[AuthorizedRuleInvocation, ...] | None = None,
        plan: RulesetPlan | None = None,
        baseline: AssessmentBaseline | None = None,
    ) -> PlannedExecutionFingerprintInput:
        return PlannedExecutionFingerprintInput(
            facts_snapshot=facts if facts is not None else _facts(),
            plan=plan if plan is not None else self.plan,
            invocations=(
                invocations
                if invocations is not None
                else (self.first, self.second)
            ),
            authorizations=(
                authorizations
                if authorizations is not None
                else (self.auth_first, self.auth_second)
            ),
            baseline=baseline if baseline is not None else self.baseline,
        )

    def test_invocation_and_authorization_order_independence(self) -> None:
        a = self._fingerprint()
        b = self._fingerprint(
            invocations=(self.second, self.first),
            authorizations=(self.auth_second, self.auth_first),
        )
        self.assertEqual(a.digest(), b.digest())
        self.assertEqual(a.canonical_payload(), b.canonical_payload())

    def test_facts_scope_authorization_and_graph_changes_change_digest(self) -> None:
        original = self._fingerprint().digest()
        changed_facts = _facts()
        changed_facts.case.title = "Changed"
        changed_scope = _invocation(self.rule, scope=_scope(actor="actor:c"))
        changed_auth = AuthorizedRuleInvocation(
            authorization_id="authorization:changed",
            rule_id=self.first.rule_id,
            rule_version=self.first.rule_version,
            scopes=(self.first.scope,),
            authorization_source="different_source",
        )
        other_rule = _ContextRule("OTHER", ordering_key="200")
        other_registry = RuleRegistry((self.rule, other_rule))
        other_plan = other_registry.build_ruleset_plan()
        other_baseline = other_plan.build_assessment_baseline()
        digests = {
            self._fingerprint(facts=changed_facts).digest(),
            self._fingerprint(
                invocations=(changed_scope,),
                authorizations=(_authorization(changed_scope),),
            ).digest(),
            self._fingerprint(
                invocations=(self.first,),
                authorizations=(changed_auth,),
            ).digest(),
            PlannedExecutionFingerprintInput(
                facts_snapshot=_facts(),
                plan=other_plan,
                invocations=(self.first,),
                authorizations=(self.auth_first,),
                baseline=other_baseline,
            ).digest(),
        }
        self.assertNotIn(original, digests)
        self.assertEqual(len(digests), 4)

    def test_prerequisite_declaration_changes_fingerprint(self) -> None:
        upstream = _ContextRule("UP", ordering_key="010")
        downstream = _ContextRule(
            "DOWN",
            phase=RulePhase.ROLE_RELEVANCE,
            dependencies=("UP",),
            accepted_statuses={"UP": ("potentially_applies",)},
        )
        _, plan, baseline, _ = _setup(upstream, downstream)
        up = _invocation(upstream)
        down = _invocation(downstream, prerequisites=(up,))
        base = PlannedExecutionFingerprintInput(
            facts_snapshot=_facts(),
            plan=plan,
            invocations=(up, down),
            authorizations=(_authorization(up), _authorization(down)),
            baseline=baseline,
        )
        changed_downstream = _ContextRule(
            "DOWN",
            phase=RulePhase.ROLE_RELEVANCE,
            dependencies=("UP",),
            accepted_statuses={"UP": ("does_not_apply",)},
        )
        _, changed_plan, changed_baseline, _ = _setup(
            upstream,
            changed_downstream,
        )
        changed = _invocation(changed_downstream, prerequisites=(up,))
        changed_fingerprint = PlannedExecutionFingerprintInput(
            facts_snapshot=_facts(),
            plan=changed_plan,
            invocations=(up, changed),
            authorizations=(_authorization(up), _authorization(changed)),
            baseline=changed_baseline,
        )
        self.assertNotEqual(base.digest(), changed_fingerprint.digest())

    def test_source_mutation_does_not_change_existing_fingerprint(self) -> None:
        facts = _facts()
        invocations = [self.first, self.second]
        authorizations = [self.auth_first, self.auth_second]
        fingerprint = self._fingerprint(
            facts=facts,
            invocations=tuple(invocations),
            authorizations=tuple(authorizations),
        )
        before = fingerprint.digest()
        facts.case.title = "Mutated after construction"
        invocations.reverse()
        authorizations.clear()
        self.assertEqual(fingerprint.digest(), before)

    def test_evidence_requirement_order_is_fingerprint_neutral(self) -> None:
        first_requirement = InvocationEvidenceRequirement(
            self.first.invocation_id,
            "evidence:first",
            True,
        )
        second_requirement = InvocationEvidenceRequirement(
            self.second.invocation_id,
            "evidence:second",
            True,
        )
        first = PlannedExecutionFingerprintInput(
            facts_snapshot=_facts(),
            plan=self.plan,
            invocations=(self.first, self.second),
            authorizations=(self.auth_first, self.auth_second),
            baseline=self.baseline,
            evidence_requirements=(first_requirement, second_requirement),
        )
        reversed_input = PlannedExecutionFingerprintInput(
            facts_snapshot=_facts(),
            plan=self.plan,
            invocations=(self.first, self.second),
            authorizations=(self.auth_first, self.auth_second),
            baseline=self.baseline,
            evidence_requirements=(second_requirement, first_requirement),
        )
        self.assertEqual(first.digest(), reversed_input.digest())
        self.assertEqual(first.canonical_payload(), reversed_input.canonical_payload())
        unavailable = PlannedExecutionFingerprintInput(
            facts_snapshot=_facts(),
            plan=self.plan,
            invocations=(self.first, self.second),
            authorizations=(self.auth_first, self.auth_second),
            baseline=self.baseline,
            evidence_requirements=(
                replace(
                    first_requirement,
                    available=False,
                    unavailable_reason="not_available",
                ),
                second_requirement,
            ),
        )
        self.assertNotEqual(first.digest(), unavailable.digest())


class StrictFingerprintBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = _ContextRule("SCREEN")
        _, self.plan, self.baseline, _ = _setup(self.rule)
        self.invocation = _invocation(self.rule)
        self.authorization = _authorization(self.invocation)

    def _build(self, **changes: object) -> PlannedExecutionFingerprintInput:
        values = {
            "facts_snapshot": _facts(),
            "plan": self.plan,
            "invocations": (self.invocation,),
            "authorizations": (self.authorization,),
            "baseline": self.baseline,
            "evidence_requirements": (),
        }
        values.update(changes)
        return PlannedExecutionFingerprintInput(**values)  # type: ignore[arg-type]

    def test_plan_incompatible_invocation_metadata_is_rejected(self) -> None:
        cases = {
            "rule_version": RuleInvocation.create(
                rule_id=self.rule.rule_id,
                rule_version="9.9.9",
                scope=self.invocation.scope,
                phase=self.invocation.phase,
                ordering_key=self.invocation.ordering_key,
            ),
            "phase": RuleInvocation.create(
                rule_id=self.rule.rule_id,
                rule_version=self.rule.version,
                scope=self.invocation.scope,
                phase=RulePhase.ROLE_RELEVANCE.value,
                ordering_key=self.invocation.ordering_key,
            ),
            "ordering_key": RuleInvocation.create(
                rule_id=self.rule.rule_id,
                rule_version=self.rule.version,
                scope=self.invocation.scope,
                phase=self.invocation.phase,
                ordering_key="999",
            ),
        }
        for label, invocation in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(PlannedExecutionInputError):
                    self._build(invocations=(invocation,), authorizations=())

    def test_duplicate_invocation_and_rule_scope_are_rejected(self) -> None:
        with self.assertRaisesRegex(PlannedExecutionInputError, "duplicate"):
            self._build(
                invocations=(self.invocation, self.invocation),
                authorizations=(self.authorization,),
            )

    def test_fingerprint_authorizations_are_fully_validated(self) -> None:
        selector = AuthorizedRuleInvocation(
            authorization_id="authorization:selector",
            rule_id=self.rule.rule_id,
            rule_version=self.rule.version,
            subject_selector="all_actors",
        )
        other = AuthorizedRuleInvocation(
            authorization_id="authorization:other",
            rule_id=self.rule.rule_id,
            rule_version=self.rule.version,
            scopes=(_scope(actor="actor:other"),),
            authorization_source="test",
        )
        wrong_version = replace(
            self.authorization,
            authorization_id="authorization:wrong-version",
            rule_version="9.9.9",
        )
        for authorization in (selector, other, wrong_version):
            with self.subTest(authorization=authorization.authorization_id):
                with self.assertRaises(PlannedExecutionInputError):
                    self._build(authorizations=(authorization,))
        with self.assertRaisesRegex(PlannedExecutionInputError, "ambiguous"):
            self._build(
                authorizations=(
                    self.authorization,
                    replace(
                        self.authorization,
                        authorization_id="authorization:second",
                    ),
                )
            )

    def test_fingerprint_evidence_requirements_are_fully_validated(self) -> None:
        other_invocation = _invocation(
            self.rule,
            scope=_scope(actor="actor:other"),
        )
        unknown = InvocationEvidenceRequirement(
            other_invocation.invocation_id,
            "evidence:unknown",
            True,
        )
        with self.assertRaisesRegex(PlannedExecutionInputError, "unknown invocation"):
            self._build(evidence_requirements=(unknown,))
        first = InvocationEvidenceRequirement(
            self.invocation.invocation_id,
            "evidence:first",
            True,
        )
        second = InvocationEvidenceRequirement(
            self.invocation.invocation_id,
            "evidence:second",
            True,
        )
        with self.assertRaisesRegex(PlannedExecutionInputError, "duplicate Evidence"):
            self._build(evidence_requirements=(first, second))

    def test_complete_baseline_version_surface_is_enforced(self) -> None:
        cases = {
            "facts_schema_version": "2.0.0",
            "engine_version": "2.0.0",
            "questionnaire_version": "2.0.0",
            "report_schema_version": "1.0.0",
            "assessment_context_version": "9.0.0",
            "execution_record_version": "9.0.0",
        }
        for field_name, value in cases.items():
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    PlannedExecutionInputError,
                    "inconsistent AssessmentBaseline",
                ):
                    self._build(
                        baseline=replace(
                            self.baseline,
                            **{field_name: value},
                        )
                    )
        for field_name, value in (
            ("rule_dependency_graph_hash", "0" * 64),
            ("ruleset_baseline_id", "ruleset:" + "0" * 64),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    PlannedExecutionInputError,
                    "inconsistent AssessmentBaseline",
                ):
                    self._build(
                        baseline=replace(
                            self.baseline,
                            **{field_name: value},
                        )
                    )

    def test_strict_hydration_rebuilds_plan_and_rejects_fabrication(self) -> None:
        fingerprint = self._build()
        restored = PlannedExecutionFingerprintInput.from_canonical_payload(
            fingerprint.canonical_payload()
        )
        self.assertEqual(restored.digest(), fingerprint.digest())
        self.assertEqual(restored.canonical_payload(), fingerprint.canonical_payload())
        for field_name in ("dependency_graph_hash", "ruleset_baseline_id"):
            payload = deepcopy(fingerprint.canonical_payload())
            payload["ruleset_plan"][field_name] = (
                "0" * 64
                if field_name == "dependency_graph_hash"
                else "ruleset:" + "0" * 64
            )
            with self.subTest(field_name=field_name):
                with self.assertRaises(PlannedExecutionInputError):
                    PlannedExecutionFingerprintInput.from_canonical_payload(payload)

    def test_hydration_rejects_duplicate_and_noncanonical_invocation_order(self) -> None:
        other = _invocation(self.rule, scope=_scope(actor="actor:other"))
        fingerprint = self._build(
            invocations=(self.invocation, other),
            authorizations=(self.authorization, _authorization(other)),
        )
        duplicate = fingerprint.canonical_payload()
        duplicate["invocations"] = [
            duplicate["invocations"][0],
            duplicate["invocations"][0],
        ]
        with self.assertRaisesRegex(PlannedExecutionInputError, "duplicate"):
            PlannedExecutionFingerprintInput.from_canonical_payload(duplicate)
        noncanonical = fingerprint.canonical_payload()
        noncanonical["invocations"].reverse()
        with self.assertRaisesRegex(PlannedExecutionInputError, "canonical"):
            PlannedExecutionFingerprintInput.from_canonical_payload(noncanonical)


class StrictPlannedResultBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = _ContextRule("SCREEN")
        _, self.plan, self.baseline, self.engine = _setup(self.rule)
        self.invocation = _invocation(self.rule)
        self.result = _run(
            self.engine,
            self.plan,
            self.baseline,
            [self.invocation],
            [_authorization(self.invocation)],
        )

    def test_strict_round_trip_uses_validated_fingerprint_payload(self) -> None:
        restored = PlannedAssessmentResult.from_dict(self.result.to_dict())
        self.assertEqual(restored.to_dict(), self.result.to_dict())
        self.assertEqual(restored.input_fingerprint, self.result.input_fingerprint)

    def test_fabricated_digest_and_plan_identity_are_rejected(self) -> None:
        payload = self.result.to_dict()
        payload["input_fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            PlannedAssessmentResult.from_dict(payload)
        for field_name in ("dependency_graph_hash", "ruleset_baseline_id"):
            payload = self.result.to_dict()
            payload["fingerprint_payload"]["ruleset_plan"][field_name] = (
                "0" * 64
                if field_name == "dependency_graph_hash"
                else "ruleset:" + "0" * 64
            )
            with self.subTest(field_name=field_name):
                with self.assertRaises(PlannedExecutionInputError):
                    PlannedAssessmentResult.from_dict(payload)

    def test_result_rejects_record_baseline_and_cardinality_conflicts(self) -> None:
        fingerprint = PlannedExecutionFingerprintInput.from_canonical_payload(
            self.result.fingerprint_payload
        )
        record = self.result.execution_records[0]
        cases = (
            (),
            (record, record),
            (replace(record, ruleset_baseline_id="ruleset:other"),),
            (replace(record, evidence_baseline_id="evidence:other"),),
            (
                replace(
                    record,
                    invocation=_invocation(
                        self.rule,
                        scope=_scope(actor="actor:other"),
                    ),
                ),
            ),
        )
        for records in cases:
            with self.subTest(records=len(records)):
                with self.assertRaises(ValueError):
                    PlannedAssessmentResult(
                        fingerprint_input=fingerprint,
                        execution_records=records,
                        findings=self.result.findings,
                        finding_associations=self.result.finding_associations,
                        diagnostics=(),
                    )

    def test_result_rejects_semantically_conflicting_findings(self) -> None:
        fingerprint = PlannedExecutionFingerprintInput.from_canonical_payload(
            self.result.fingerprint_payload
        )
        finding = self.result.findings[0]
        for changed in (
            replace(finding, rule_id="OTHER"),
            replace(finding, rule_version="9.9.9"),
        ):
            with self.subTest(rule_id=changed.rule_id, version=changed.rule_version):
                with self.assertRaises(ValueError):
                    PlannedAssessmentResult(
                        fingerprint_input=fingerprint,
                        execution_records=self.result.execution_records,
                        findings=(changed,),
                        finding_associations=self.result.finding_associations,
                        diagnostics=(),
                    )
        other = _invocation(self.rule, scope=_scope(actor="actor:other"))
        with self.assertRaises(ValueError):
            PlannedAssessmentResult(
                fingerprint_input=fingerprint,
                execution_records=self.result.execution_records,
                findings=self.result.findings,
                finding_associations=(
                    InvocationFindingAssociation(
                        other.invocation_id,
                        finding.finding_id,
                    ),
                ),
                diagnostics=(),
            )
        with self.assertRaises(ValueError):
            PlannedAssessmentResult(
                fingerprint_input=fingerprint,
                execution_records=(
                    replace(self.result.execution_records[0], finding_id=None),
                ),
                findings=self.result.findings,
                finding_associations=(),
                diagnostics=(),
            )

    def test_result_rejects_inconsistent_diagnostics(self) -> None:
        missing = _run(self.engine, self.plan, self.baseline, [self.invocation], [])
        fingerprint = PlannedExecutionFingerprintInput.from_canonical_payload(
            missing.fingerprint_payload
        )
        correct = missing.diagnostics[0]
        cases = (
            (),
            (correct, correct),
            (
                replace(
                    correct,
                    invocation_id=_invocation(
                        self.rule,
                        scope=_scope(actor="actor:other"),
                    ).invocation_id,
                ),
            ),
            (replace(correct, code="facts.required_missing"),),
        )
        for diagnostics in cases:
            with self.subTest(count=len(diagnostics)):
                with self.assertRaises(ValueError):
                    PlannedAssessmentResult(
                        fingerprint_input=fingerprint,
                        execution_records=missing.execution_records,
                        findings=(),
                        finding_associations=(),
                        diagnostics=diagnostics,
                    )
        completed_fingerprint = PlannedExecutionFingerprintInput.from_canonical_payload(
            self.result.fingerprint_payload
        )
        with self.assertRaises(ValueError):
            PlannedAssessmentResult(
                fingerprint_input=completed_fingerprint,
                execution_records=self.result.execution_records,
                findings=self.result.findings,
                finding_associations=self.result.finding_associations,
                diagnostics=(
                    ExecutionDiagnostic(
                        self.invocation.invocation_id,
                        "authorization.not_explicit",
                        "No matching explicit authorization was supplied.",
                    ),
                ),
            )


class _SelectiveRaisingRequirementValidator(FactRequirementValidator):
    def validate(self, rule: AssessmentRule, facts: AssessmentFacts):  # type: ignore[no-untyped-def]
        if rule.rule_id == "VALIDATION_FAIL":
            raise RuntimeError("private /tmp/path at 0xABC123")
        return super().validate(rule, facts)


class RequirementValidationFailureIsolationTests(unittest.TestCase):
    def test_validator_failure_is_isolated_and_blocks_only_dependents(self) -> None:
        failing = _ContextRule("VALIDATION_FAIL", ordering_key="010")
        independent = _ContextRule("INDEPENDENT", ordering_key="020")
        dependent = _ContextRule(
            "DEPENDENT",
            phase=RulePhase.ROLE_RELEVANCE,
            ordering_key="010",
            dependencies=("VALIDATION_FAIL",),
            accepted_statuses={"VALIDATION_FAIL": ("potentially_applies",)},
        )
        registry = RuleRegistry((dependent, independent, failing))
        plan = registry.build_ruleset_plan()
        baseline = plan.build_assessment_baseline()
        engine = PlannedAssessmentEngine(
            registry,
            requirement_validator=_SelectiveRaisingRequirementValidator(),
        )
        failing_invocation = _invocation(failing)
        independent_invocation = _invocation(independent)
        dependent_invocation = _invocation(
            dependent,
            prerequisites=(failing_invocation,),
        )
        invocations = (
            dependent_invocation,
            independent_invocation,
            failing_invocation,
        )
        result = _run(
            engine,
            plan,
            baseline,
            invocations,
            tuple(_authorization(item) for item in invocations),
        )
        by_rule = {
            record.invocation.rule_id: record
            for record in result.execution_records
        }
        self.assertEqual(by_rule["VALIDATION_FAIL"].status, RuleExecutionStatus.FAILED)
        self.assertEqual(by_rule["INDEPENDENT"].status, RuleExecutionStatus.COMPLETED)
        self.assertEqual(
            by_rule["DEPENDENT"].status,
            RuleExecutionStatus.BLOCKED_BY_DEPENDENCY,
        )
        self.assertEqual(len(result.execution_records), len(invocations))
        self.assertEqual(failing.calls, 0)
        self.assertEqual(independent.calls, 1)
        self.assertEqual(dependent.calls, 0)
        canonical = json.dumps(result.to_dict(), sort_keys=True)
        self.assertNotIn("/tmp", canonical)
        self.assertNotIn("0x", canonical)
        self.assertNotIn("Traceback", canonical)
        self.assertIn("rule.requirement_validation_failed", canonical)


class ExecutionPrecedenceTests(unittest.TestCase):
    def test_unauthorized_precedes_missing_facts(self) -> None:
        rule = _ContextRule("SCREEN", required_paths=("use_context.task",))
        _, plan, baseline, engine = _setup(rule)
        result = _run(engine, plan, baseline, [_invocation(rule)], [], facts=_facts())
        self.assertEqual(
            result.execution_records[0].status,
            RuleExecutionStatus.NOT_AUTHORIZED,
        )
        self.assertEqual(rule.calls, 0)

    def test_unauthorized_precedes_blocked_dependency(self) -> None:
        _, downstream, plan, baseline, engine, up, down = (
            PlannedDependencyTests()._chain()
        )
        result = _run(engine, plan, baseline, [up, down], [])
        statuses = {
            record.invocation.rule_id: record.status
            for record in result.execution_records
        }
        self.assertEqual(statuses["DOWNSTREAM"], RuleExecutionStatus.NOT_AUTHORIZED)
        self.assertEqual(downstream.calls, 0)

    def test_blocked_dependency_precedes_missing_facts(self) -> None:
        _, downstream, plan, baseline, engine, up, down = (
            PlannedDependencyTests()._chain(
                upstream_status=FindingStatus.DOES_NOT_APPLY
            )
        )
        downstream.required_fact_paths = ("use_context.task",)
        result = _run(
            engine,
            plan,
            baseline,
            [up, down],
            [_authorization(up), _authorization(down)],
            facts=_facts(),
        )
        self.assertEqual(
            result.execution_records[1].status,
            RuleExecutionStatus.BLOCKED_BY_DEPENDENCY,
        )
        self.assertEqual(downstream.calls, 0)

    def test_unavailable_evidence_precedes_missing_facts(self) -> None:
        rule = _ContextRule("SCREEN", required_paths=("use_context.task",))
        _, plan, baseline, engine = _setup(rule)
        invocation = _invocation(rule)
        requirement = InvocationEvidenceRequirement(
            invocation.invocation_id,
            "evidence:required",
            False,
            "not_available",
        )
        result = _run(
            engine,
            plan,
            baseline,
            [invocation],
            [_authorization(invocation)],
            facts=_facts(),
            evidence=(requirement,),
        )
        self.assertEqual(
            result.execution_records[0].status,
            RuleExecutionStatus.BLOCKED_BY_EVIDENCE,
        )
        self.assertEqual(rule.calls, 0)


class LegacyAdapterRegressionTests(unittest.TestCase):
    def _planned_and_legacy(
        self,
        rule: AssessmentRule,
        facts: AssessmentFacts,
    ) -> tuple[Finding, Finding]:
        registry = RuleRegistry((rule,))
        legacy = AssessmentEngine(registry).run(facts).findings[0]
        plan = registry.build_ruleset_plan()
        baseline = plan.build_assessment_baseline()
        invocation = _invocation(rule)
        planned = PlannedAssessmentEngine(registry).run(
            facts_snapshot=AssessmentFactsCompatibilityAdapter().derive_v3(facts),
            plan=plan,
            invocations=(invocation,),
            authorizations=(_authorization(invocation),),
            baseline=baseline,
        ).findings[0]
        return legacy, planned

    def test_all_four_existing_screening_rules_match_legacy_semantics(self) -> None:
        fixture = json.loads(
            Path("tests/fixtures/recruitment_ai_case.json").read_text()
        )
        recruitment = AssessmentFacts.from_dict(fixture["facts"])

        gdpr = AssessmentFacts.from_dict(recruitment.to_dict()).make_editable()
        gdpr.data_protection.personal_data_processed = TriState.YES
        gdpr.data_protection.automated_individual_decision = TriState.YES

        data_act = AssessmentFacts()
        data_act.data_act.connected_product = TriState.YES
        data_act.data_act.related_service = TriState.NO
        data_act.data_act.data_generated = TriState.YES

        product = AssessmentFacts()
        product.product_regulation.ai_is_product = TriState.YES
        product.product_regulation.ai_is_safety_component = TriState.NO
        catalogue = load_annex_i_instrument_catalog()
        product.product_regulation.annex_i_instrument = catalogue.all()[0].instrument_id
        product.product_regulation.annex_i_instrument_confirmed = TriState.YES
        product.product_regulation.third_party_conformity_required = TriState.YES

        pairs = (
            (AIActHighRiskEmploymentRule(), recruitment),
            (GDPRArticle22RelevanceRule(), gdpr),
            (EUDataActRelevanceRule(), data_act),
            (AIActHighRiskProductSafetyRule(), product),
        )
        for rule, facts in pairs:
            with self.subTest(rule=rule.rule_id):
                legacy, planned = self._planned_and_legacy(rule, facts)
                self.assertEqual(rule.version, "2026.1")
                for field in (
                    "framework",
                    "category",
                    "issue_code",
                    "status",
                    "title",
                    "summary",
                    "rule_id",
                    "rule_version",
                    "fact_refs",
                    "reason_codes",
                    "legal_basis",
                    "trace",
                ):
                    self.assertEqual(getattr(planned, field), getattr(legacy, field))

    def test_legacy_adapter_obeys_authorization_and_missing_fact_guards(self) -> None:
        rule = _LegacyCountingRule()
        _, plan, baseline, engine = _setup(rule)
        invocation = _invocation(rule)
        unauthorized = _run(engine, plan, baseline, [invocation], [])
        self.assertEqual(unauthorized.execution_records[0].status, RuleExecutionStatus.NOT_AUTHORIZED)
        self.assertEqual(rule.calls, 0)
        rule.required_fact_paths = ("use_context.task",)
        missing = _run(engine, plan, baseline, [invocation], [_authorization(invocation)])
        self.assertEqual(missing.execution_records[0].status, RuleExecutionStatus.MISSING_FACTS)
        self.assertEqual(rule.calls, 0)

    def test_production_engine_remains_unplanned_and_version_2(self) -> None:
        self.assertEqual(AssessmentEngine.VERSION, "2.0.0")
        engine_source = Path("src/assessment/engine.py").read_text()
        self.assertNotIn("RulesetPlan", engine_source)
        self.assertNotIn("PlannedAssessmentEngine", engine_source)


if __name__ == "__main__":
    unittest.main()
