"""Focused acceptance tests for the v0.6 F1 domain foundation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import unittest

from src.assessment import (
    ActorFacts,
    ActorId,
    ActorKind,
    AISystemFacts,
    ApplicabilityLimitation,
    ArtefactSupplyState,
    AssessmentBaseline,
    AssessmentCaseService,
    AssessmentContext,
    AssessmentFacts,
    AssessmentFactsCompatibilityAdapter,
    AssessmentScope,
    AuthorizedRuleInvocation,
    ClassificationReviewState,
    ComplianceArtefactMetadata,
    EvidencePackBaseline,
    FindingCategory,
    FindingStatus,
    InformationalGap,
    InvocationId,
    PrerequisiteFindingSummary,
    ProcessingOperationFacts,
    RecruitmentDecisionProcessFacts,
    RecruitmentWorkflowFacts,
    RegulatoryFramework,
    RoleHypothesis,
    RoleHypothesisStatus,
    RuleBaselineEntry,
    RuleExecutionRecord,
    RuleExecutionStatus,
    RuleInvocation,
    ScreeningCriterionFacts,
    SystemId,
    TemporalContextFacts,
    TerritorialContextFacts,
    TriState,
    V06FingerprintInput,
    WorkflowId,
)
from src.assessment.evidence.service import EvidenceServiceResult
from src.assessment.report import ReportBuilder
from src.assessment.results import AssessmentResult
from src.assessment.rules import (
    AIActHighRiskEmploymentRule,
    AIActHighRiskProductSafetyRule,
    EUDataActRelevanceRule,
    GDPRArticle22RelevanceRule,
)
from src.assessment.workflow import AssessmentWorkflowService


ROOT = Path(__file__).resolve().parents[3]


def _scope(
    actor: str = "actor:recruiter",
    operation: str = "operation:client-screening",
) -> AssessmentScope:
    return AssessmentScope(
        actor_id=ActorId(actor),
        system_id=SystemId("system:ranker"),
        workflow_id=WorkflowId("workflow:hiring"),
        processing_operation_id=operation,
    )


def _invocation(scope: AssessmentScope | None = None) -> RuleInvocation:
    return RuleInvocation.create(
        rule_id="AI_ACT_DEPLOYER_ROLE_RELEVANCE",
        rule_version="2026.2",
        scope=scope or _scope(),
        phase="role_relevance",
        ordering_key="20",
    )


def _upstream_invocation(
    scope: AssessmentScope | None = None,
) -> RuleInvocation:
    return RuleInvocation.create(
        rule_id="AI_ACT_HIGH_RISK_EMPLOYMENT",
        rule_version="2026.1",
        scope=scope or _scope(),
    )


def _dependent_invocation(
    scope: AssessmentScope | None = None,
) -> tuple[RuleInvocation, RuleInvocation]:
    upstream = _upstream_invocation(scope)
    invocation = replace(
        _invocation(scope),
        prerequisite_invocation_ids=(upstream.invocation_id,),
        accepted_upstream_statuses={
            str(upstream.invocation_id): (
                FindingStatus.POTENTIALLY_APPLIES.value,
            )
        },
    )
    return invocation, upstream


def _authorization(invocation: RuleInvocation) -> AuthorizedRuleInvocation:
    return AuthorizedRuleInvocation(
        authorization_id="authorization:one",
        rule_id=invocation.rule_id,
        rule_version=invocation.rule_version,
        scopes=(invocation.scope,),
        authorization_source="user_confirmation",
        authorized_at=datetime(2026, 7, 27, tzinfo=timezone.utc),
    )


def _baseline(
    *,
    rule_version: str = "2026.2",
    manifest_hash: str | None = None,
) -> AssessmentBaseline:
    return AssessmentBaseline(
        ordered_rules=(
            RuleBaselineEntry(
                rule_id="AI_ACT_DEPLOYER_ROLE_RELEVANCE",
                rule_version=rule_version,
            ),
        ),
        rule_dependency_graph_hash=None,
        ruleset_baseline_id="ruleset:recruitment-v0.6",
        evidence_packs=(
            EvidencePackBaseline(
                instrument_id="EU_AI_ACT",
                pack_version=None,
                manifest_hash=manifest_hash,
            ),
        ),
        legal_source_baseline_id="legal-source:2026-07",
    )


def _baseline_with_prerequisites(
    *prerequisites: RuleInvocation,
) -> AssessmentBaseline:
    baseline = _baseline()
    return replace(
        baseline,
        ordered_rules=baseline.ordered_rules
        + tuple(
            RuleBaselineEntry(item.rule_id, item.rule_version)
            for item in prerequisites
        ),
    )


def _v3_facts() -> AssessmentFacts:
    actors = [
        ActorFacts(
            actor_id=ActorId("actor:employer"),
            display_name="Client employer",
            actor_kind=ActorKind.EMPLOYER,
            establishment_locations=["DE"],
        ),
        ActorFacts(
            actor_id=ActorId("actor:recruiter"),
            display_name="Recruitment agency",
            actor_kind=ActorKind.RECRUITER,
            establishment_locations=["FR"],
        ),
    ]
    systems = [
        AISystemFacts(system_id=SystemId("system:ranker"), name="Ranker"),
        AISystemFacts(system_id=SystemId("system:matcher"), name="Matcher"),
    ]
    operations = [
        ProcessingOperationFacts(
            processing_operation_id="operation:client-screening",
            workflow_id="workflow:hiring",
            system_ids=["system:ranker"],
            participating_actor_ids=["actor:employer", "actor:recruiter"],
            reported_purpose="Screen applicants for the client vacancy",
        ),
        ProcessingOperationFacts(
            processing_operation_id="operation:talent-pool",
            workflow_id="workflow:hiring",
            system_ids=["system:matcher"],
            participating_actor_ids=["actor:recruiter"],
            reported_purpose="Maintain an independent talent pool",
            independent_reuse_purpose=TriState.YES,
        ),
    ]
    workflows = [
        RecruitmentWorkflowFacts(
            workflow_id="workflow:hiring",
            title="Client hiring",
            employer_actor_ids=["actor:employer"],
            recruiter_actor_ids=["actor:recruiter"],
            system_ids=["system:ranker", "system:matcher"],
            final_decision_actor_ids=["actor:employer"],
            processing_operation_ids=[
                "operation:client-screening",
                "operation:talent-pool",
            ],
        )
    ]
    criterion = ScreeningCriterionFacts(
        criterion_id="criterion:experience",
        category="work_experience",
        gdpr_special_category_data=TriState.UNKNOWN,
        employment_equality_protected_characteristic=TriState.UNKNOWN,
        proxy_for_protected_characteristic=TriState.UNKNOWN,
        classification_review_state=ClassificationReviewState.NOT_REVIEWED,
    )
    processes = [
        RecruitmentDecisionProcessFacts(
            process_id="process:screening",
            workflow_id="workflow:hiring",
            processing_operation_id="operation:client-screening",
            system_id="system:ranker",
            ranking=TriState.YES,
            filtering=TriState.YES,
            recommendation=TriState.YES,
            automatic_exclusion=TriState.UNKNOWN,
            human_review=TriState.YES,
            substantive_basis_review=TriState.UNKNOWN,
            genuine_override_authority=TriState.UNKNOWN,
            routine_following_of_ai_output=TriState.UNKNOWN,
            screening_criteria=[criterion],
            documented_instructions=TriState.YES,
            system_configuration_authority=TriState.UNKNOWN,
            final_decision_authority=TriState.YES,
        )
    ]
    return AssessmentFacts.new_v3(
        actors=actors,
        ai_systems=systems,
        recruitment_workflows=workflows,
        processing_operations=operations,
        recruitment_processes=processes,
        temporal_context=TemporalContextFacts(
            assessment_date=None,
            intended_use_dates={"workflow:hiring": None},
            put_into_service_dates={"system:ranker": None},
            legal_source_baseline_id="legal-source:2026-07",
        ),
        territorial_context=TerritorialContextFacts(
            actor_establishment_locations={
                "actor:employer": ["DE"],
                "actor:recruiter": ["FR"],
            },
            system_use_locations={"system:ranker": None},
            output_use_locations={"workflow:hiring": None},
            affected_person_locations={"workflow:hiring": None},
            processing_operation_context={
                "operation:client-screening": None,
                "operation:talent-pool": None,
            },
        ),
    )


class StableIdentityTests(unittest.TestCase):
    def test_identity_survives_edits_and_list_reordering(self) -> None:
        facts = _v3_facts()
        before = sorted(facts.active_entity_ids())
        facts.actors[0].display_name = "Renamed employer"
        facts.actors.reverse()
        facts.ai_systems.reverse()

        self.assertEqual(sorted(facts.active_entity_ids()), before)
        with self.assertRaisesRegex(AttributeError, "actor_id is immutable"):
            facts.actors[0].actor_id = ActorId("actor:replacement")
        self.assertEqual(
            AssessmentFacts.from_dict(facts.to_dict()).to_dict(),
            facts.to_dict(),
        )

    def test_blank_and_malformed_ids_are_rejected(self) -> None:
        for invalid in ("", " actor", "actor translated name", "/actor"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ActorId(invalid)
        with self.assertRaisesRegex(ValueError, "scoped subject"):
            RuleInvocation.create(
                rule_id="RULE",
                rule_version="1.0.0",
                scope=AssessmentScope(),
            )

    def test_deleted_identity_must_be_retired_and_cannot_be_reused(self) -> None:
        service = AssessmentCaseService()
        original = _v3_facts()
        service.create_case("v3", facts=original, case_id="case:v3")
        changed = deepcopy(original)
        changed.actors = changed.actors[1:]
        changed.recruitment_workflows[0].employer_actor_ids = []
        changed.recruitment_workflows[0].final_decision_actor_ids = []
        changed.processing_operations[0].participating_actor_ids = [
            "actor:recruiter"
        ]
        del changed.territorial_context.actor_establishment_locations[
            "actor:employer"
        ]

        with self.assertRaisesRegex(ValueError, "retired_entity_ids"):
            service.update_facts("case:v3", changed)

        changed.retired_entity_ids.append("actor:employer")
        service.update_facts("case:v3", changed)
        changed.actors.append(ActorFacts(actor_id="actor:employer"))
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            changed.validate_v3()


class AssessmentFactsV3Tests(unittest.TestCase):
    def test_multiple_scoped_records_and_unknown_context_serialize(self) -> None:
        facts = _v3_facts()
        payload = facts.to_dict()

        self.assertEqual(payload["schema_version"], "3.0.0")
        self.assertEqual(len(payload["actors"]), 2)
        self.assertEqual(len(payload["ai_systems"]), 2)
        self.assertEqual(len(payload["processing_operations"]), 2)
        self.assertIsNone(payload["temporal_context"]["assessment_date"])
        self.assertIsNone(
            payload["territorial_context"]["system_use_locations"][
                "system:ranker"
            ]
        )
        criterion = payload["recruitment_processes"][0]["screening_criteria"][0]
        self.assertEqual(criterion["gdpr_special_category_data"], "unknown")
        self.assertEqual(
            criterion["employment_equality_protected_characteristic"],
            "unknown",
        )
        self.assertEqual(
            criterion["proxy_for_protected_characteristic"], "unknown"
        )
        self.assertEqual(criterion["classification_review_state"], "not_reviewed")
        self.assertFalse(hasattr(facts.actors[0], "controller"))
        self.assertFalse(hasattr(facts.actors[0], "deployer"))


class BackwardCompatibilityTests(unittest.TestCase):
    def test_existing_v2_fixture_round_trips_without_silent_upgrade(self) -> None:
        fixture = json.loads(
            (ROOT / "tests/fixtures/recruitment_ai_case.json").read_text()
        )["facts"]
        restored = AssessmentFacts.from_dict(fixture)

        self.assertEqual(restored.schema_version, "2.0.0")
        self.assertEqual(restored.to_dict(), fixture)
        self.assertNotIn("actors", restored.to_dict())

    def test_explicit_adapter_preserves_source_version_without_entities(self) -> None:
        fixture = json.loads(
            (ROOT / "tests/fixtures/recruitment_ai_case.json").read_text()
        )["facts"]
        source = AssessmentFactsCompatibilityAdapter.read(fixture)
        source_snapshot = source.to_dict()

        derived = AssessmentFactsCompatibilityAdapter.derive_v3(source)

        self.assertEqual(source.to_dict(), source_snapshot)
        self.assertEqual(derived.schema_version, "3.0.0")
        self.assertEqual(derived.source_schema_version, "2.0.0")
        self.assertIsNone(derived.actors)
        self.assertIsNone(derived.ai_systems)
        self.assertIsNone(derived.processing_operations)

    def test_rule_versions_are_unchanged(self) -> None:
        self.assertEqual(
            {
                rule.rule_id: rule.version
                for rule in (
                    AIActHighRiskEmploymentRule(),
                    AIActHighRiskProductSafetyRule(),
                    GDPRArticle22RelevanceRule(),
                    EUDataActRelevanceRule(),
                )
            },
            {
                "AI_ACT_HIGH_RISK_EMPLOYMENT": "2026.1",
                "AI_ACT_HIGH_RISK_PRODUCT_SAFETY": "2026.1",
                "GDPR_ARTICLE22_RELEVANCE": "2026.1",
                "EU_DATA_ACT_RELEVANCE": "2026.1",
            },
        )

    def test_report_1_snapshot_round_trips_without_v2_fields(self) -> None:
        result = AssessmentResult(
            findings=[],
            executed_rule_ids=[],
            engine_version="2.0.0",
            timestamp=datetime(2026, 7, 27, tzinfo=timezone.utc),
        )
        report = ReportBuilder().build(result, EvidenceServiceResult([], []))
        payload = report.to_dict()
        restored = type(report).from_dict(payload)

        self.assertEqual(restored.report_version, "1.0.0")
        self.assertEqual(restored.to_dict(), payload)
        self.assertNotIn("rule_invocations", payload)

    def test_legacy_fingerprint_algorithm_is_unchanged(self) -> None:
        facts = AssessmentFacts()
        rule_ids = ("AI_ACT_HIGH_RISK_EMPLOYMENT",)
        expected = sha256(
            json.dumps(
                {
                    "facts": facts.to_dict(),
                    "authorized_rule_ids": list(rule_ids),
                    "engine_version": "2.0.0",
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        ).hexdigest()
        service = object.__new__(AssessmentWorkflowService)
        service._assessment_engine = type(
            "EngineVersion", (), {"engine_version": "2.0.0"}
        )()

        self.assertEqual(service._fingerprint(facts.to_dict(), rule_ids), expected)


class InvocationAndContextTests(unittest.TestCase):
    def test_scope_changes_create_distinct_deterministic_invocations(self) -> None:
        first = _invocation()
        second_actor = _invocation(_scope(actor="actor:employer"))
        second_operation = _invocation(_scope(operation="operation:talent-pool"))

        self.assertEqual(first, _invocation())
        self.assertNotEqual(first.invocation_id, second_actor.invocation_id)
        self.assertNotEqual(first.invocation_id, second_operation.invocation_id)
        self.assertEqual(
            json.dumps(first.to_dict(), sort_keys=True),
            json.dumps(_invocation().to_dict(), sort_keys=True),
        )

    def test_context_defensively_copies_and_exposes_no_services_or_ui(self) -> None:
        facts = _v3_facts()
        invocation, upstream = _dependent_invocation()
        authorization = _authorization(invocation)
        context = AssessmentContext(
            facts_snapshot=facts,
            invocation=invocation,
            prerequisite_findings=(
                PrerequisiteFindingSummary(
                    finding_id="finding:upstream",
                    prerequisite_invocation_id=upstream.invocation_id,
                    scope=upstream.scope,
                    framework=RegulatoryFramework.EU_AI_ACT,
                    rule_id=upstream.rule_id,
                    rule_version=upstream.rule_version,
                    status=FindingStatus.POTENTIALLY_APPLIES.value,
                ),
            ),
            authorization=authorization,
            baseline=_baseline_with_prerequisites(upstream),
        )
        facts.actors[0].display_name = "Mutated source"
        returned = context.facts_snapshot
        returned.actors[0].display_name = "Mutated returned copy"

        self.assertEqual(
            context.facts_snapshot.actors[0].display_name, "Client employer"
        )
        self.assertFalse(hasattr(context, "ui_state"))
        self.assertFalse(hasattr(context, "services"))
        self.assertFalse(hasattr(context, "findings"))
        self.assertEqual(context.to_dict()["contract_version"], "1.0.0")


class ExecutionAndReportTests(unittest.TestCase):
    def test_execution_state_is_separate_from_finding_status(self) -> None:
        invocation = _invocation()
        completed = RuleExecutionRecord(
            invocation=invocation,
            status=RuleExecutionStatus.COMPLETED,
            finding_id="finding:one",
        )
        self.assertEqual(completed.finding_id, "finding:one")
        with self.assertRaisesRegex(ValueError, "only completed"):
            RuleExecutionRecord(
                invocation=invocation,
                status=RuleExecutionStatus.NOT_AUTHORIZED,
                finding_id="finding:fake",
            )
        missing = RuleExecutionRecord(
            invocation=invocation,
            status=RuleExecutionStatus.MISSING_FACTS,
            missing_fact_paths=("actors[actor:recruiter].acts_in_own_name",),
        )
        self.assertNotIn("finding_id", {
            key for key, value in missing.to_dict().items() if value is not None
        })

    def test_report_2_preserves_scoped_and_informational_separation(self) -> None:
        invocation = _invocation()
        authorization = _authorization(invocation)
        artefact = ComplianceArtefactMetadata(
            artefact_id="artefact:instructions",
            artefact_type="instructions_for_use",
            scope=invocation.scope,
            file_reference="opaque:case-reference",
            supply_state=ArtefactSupplyState.SUPPLIED,
        )
        hypothesis = RoleHypothesis(
            hypothesis_id="hypothesis:deployer",
            framework=RegulatoryFramework.EU_AI_ACT,
            scope=invocation.scope,
            hypothesis_type="deployer_like",
            status=RoleHypothesisStatus.SUPPORTED,
            supporting_fact_paths=["actors[actor:recruiter].operates_system_ids"],
            reason_codes=["OBSERVED_OPERATION"],
        )
        result = AssessmentResult(
            findings=[],
            executed_rule_ids=[],
            engine_version="2.0.0",
            timestamp=datetime(2026, 7, 27, tzinfo=timezone.utc),
        )
        legacy = ReportBuilder().build(result, EvidenceServiceResult([], []))
        report = replace(
            legacy,
            report_version="2.0.0",
            rule_invocations=[invocation],
            authorized_rule_invocations=[authorization],
            rule_execution_records=[
                RuleExecutionRecord(
                    invocation=invocation,
                    status=RuleExecutionStatus.NOT_AUTHORIZED,
                )
            ],
            actor_references=[ActorId("actor:recruiter")],
            system_references=[SystemId("system:ranker")],
            workflow_references=[WorkflowId("workflow:hiring")],
            processing_operation_references=["operation:client-screening"],
            role_hypotheses=[hypothesis],
            compliance_artefacts=[artefact],
            unresolved_informational_gaps=[
                InformationalGap(
                    gap_id="gap:role",
                    scope_description="recruiter/client-screening",
                    missing_fact_paths=["decision_rights.criteria"],
                )
            ],
            applicability_limitation=ApplicabilityLimitation(
                explanation="Temporal and territorial applicability not assessed."
            ),
            assessment_baseline=_baseline(),
        )
        payload = report.to_dict()

        self.assertEqual(payload["report_version"], "2.0.0")
        self.assertEqual(payload["findings"], [])
        self.assertEqual(len(payload["role_hypotheses"]), 1)
        self.assertEqual(
            payload["rule_execution_records"][0]["status"], "not_authorized"
        )
        self.assertEqual(
            payload["compliance_artefacts"][0]["file_reference"],
            "opaque:case-reference",
        )
        self.assertEqual(type(report).from_dict(payload).to_dict(), payload)


class BaselineFingerprintTests(unittest.TestCase):
    def test_canonical_order_is_deterministic(self) -> None:
        first = _invocation()
        second = _invocation(_scope(actor="actor:employer"))
        a = V06FingerprintInput(
            facts=_v3_facts().to_dict(),
            invocations=(first, second),
            baseline=_baseline(),
        )
        b = V06FingerprintInput(
            facts=deepcopy(a.facts),
            invocations=(second, first),
            baseline=deepcopy(a.baseline),
        )
        self.assertEqual(a.digest(), b.digest())

    def test_version_manifest_and_operation_scope_change_digest(self) -> None:
        facts = _v3_facts().to_dict()
        original = V06FingerprintInput(
            facts=facts,
            invocations=(_invocation(),),
            baseline=_baseline(),
        ).digest()
        changed_invocation = RuleInvocation.create(
            rule_id="AI_ACT_DEPLOYER_ROLE_RELEVANCE",
            rule_version="2026.3",
            scope=_scope(),
            phase="role_relevance",
            ordering_key="20",
        )
        changed_version = V06FingerprintInput(
            facts=facts,
            invocations=(changed_invocation,),
            baseline=_baseline(rule_version="2026.3"),
        ).digest()
        changed_manifest = V06FingerprintInput(
            facts=facts,
            invocations=(_invocation(),),
            baseline=_baseline(manifest_hash="abc123"),
        ).digest()
        changed_operation = V06FingerprintInput(
            facts=facts,
            invocations=(
                _invocation(_scope(operation="operation:talent-pool")),
            ),
            baseline=_baseline(),
        ).digest()

        self.assertEqual(len({original, changed_version, changed_manifest, changed_operation}), 4)
        self.assertIsNone(_baseline().rule_dependency_graph_hash)
        self.assertIsNone(_baseline().evidence_packs[0].manifest_hash)


if __name__ == "__main__":
    unittest.main()
