"""Regression tests for the corrected v0.6 F1 persistence boundaries."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import unittest

from src.assessment import (
    ActorFacts,
    ActorId,
    ActorKind,
    AISystemFacts,
    AssessmentBaseline,
    AssessmentContext,
    AssessmentFacts,
    AssessmentFactsCompatibilityAdapter,
    AssessmentReport,
    AssessmentRun,
    AssessmentScope,
    AuthorizedRuleInvocation,
    AuthorityLevel,
    ArtefactSupplyState,
    ClassificationReviewState,
    ComplianceArtefactMetadata,
    Evidence,
    EvidencePackBaseline,
    Finding,
    FindingCategory,
    FindingEvidenceBinding,
    FindingStatus,
    PrerequisiteFindingSummary,
    ProcessingOperationFacts,
    RecruitmentDecisionProcessFacts,
    RegulatoryFramework,
    RoleHypothesis,
    RoleHypothesisStatus,
    RuleBaselineEntry,
    RuleExecutionRecord,
    RuleExecutionStatus,
    RuleInvocation,
    ScreeningCriterionFacts,
    SystemId,
    TriState,
    V06FingerprintInput,
)
from src.assessment.evidence.service import EvidenceServiceResult
from src.assessment.report import ReportBuilder
from src.assessment.results import AssessmentResult
from tests.assessment.foundation.test_v06_foundation import (
    _authorization,
    _baseline,
    _baseline_with_prerequisites,
    _dependent_invocation,
    _invocation,
    _upstream_invocation,
    _v3_facts,
)


ROOT = Path(__file__).resolve().parents[3]


def _legacy_report() -> AssessmentReport:
    result = AssessmentResult(
        findings=[],
        executed_rule_ids=[],
        engine_version="2.0.0",
        timestamp=datetime(2026, 7, 27, tzinfo=timezone.utc),
    )
    return ReportBuilder().build(result, EvidenceServiceResult([], []))


class FingerprintBoundaryTests(unittest.TestCase):
    def test_semantically_unordered_v3_records_and_references_are_canonical(self) -> None:
        first = _v3_facts().to_dict()
        second = deepcopy(first)
        for name in (
            "actors",
            "ai_systems",
            "recruitment_workflows",
            "processing_operations",
        ):
            second[name].reverse()
        second["recruitment_workflows"][0]["system_ids"].reverse()
        second["processing_operations"][0]["participating_actor_ids"].reverse()

        self.assertEqual(
            V06FingerprintInput(
                facts=first,
                invocations=(_invocation(),),
                baseline=_baseline(),
            ).digest(),
            V06FingerprintInput(
                facts=second,
                invocations=(_invocation(),),
                baseline=_baseline(),
            ).digest(),
        )

    def test_path_aware_unordered_values_and_ordered_stages(self) -> None:
        first = _v3_facts().to_dict()
        first["actors"][0]["uses_system_ids_on_behalf_of_actor_ids"] = {
            "system:ranker": ["actor:employer", "actor:recruiter"]
        }
        first["actors"][0]["establishment_locations"] = ["DE", "FR"]
        first["ai_systems"][0]["outputs"] = ["ranking", "score"]
        first["processing_operations"][0]["data_categories"] = [
            "experience",
            "skills",
        ]
        first["processing_operations"][0]["data_sources"] = [
            "application",
            "cv",
        ]
        first["recruitment_workflows"][0]["recruitment_stages"] = [
            "screen",
            "interview",
        ]
        second = deepcopy(first)
        second["actors"][0]["uses_system_ids_on_behalf_of_actor_ids"][
            "system:ranker"
        ].reverse()
        second["actors"][0]["establishment_locations"].reverse()
        second["ai_systems"][0]["outputs"].reverse()
        second["processing_operations"][0]["data_categories"].reverse()
        second["processing_operations"][0]["data_sources"].reverse()

        original = V06FingerprintInput(
            facts=first,
            invocations=(_invocation(),),
            baseline=_baseline(),
        ).digest()
        self.assertEqual(
            original,
            V06FingerprintInput(
                facts=second,
                invocations=(_invocation(),),
                baseline=_baseline(),
            ).digest(),
        )

        ordered_change = deepcopy(first)
        ordered_change["recruitment_workflows"][0][
            "recruitment_stages"
        ].reverse()
        self.assertNotEqual(
            original,
            V06FingerprintInput(
                facts=ordered_change,
                invocations=(_invocation(),),
                baseline=_baseline(),
            ).digest(),
        )

        duplicate = deepcopy(first)
        duplicate["actors"][0]["uses_system_ids_on_behalf_of_actor_ids"][
            "system:ranker"
        ] = ["actor:employer", "actor:employer"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            V06FingerprintInput(
                facts=duplicate,
                invocations=(_invocation(),),
                baseline=_baseline(),
            )

    def test_fingerprint_normalizes_typed_contracts_and_rejects_duplicate_ids(
        self,
    ) -> None:
        invocation = _invocation()
        fingerprint = V06FingerprintInput(
            facts=_v3_facts().to_dict(),
            invocations=[invocation.to_dict()],
            baseline=_baseline().to_dict(),
        )
        self.assertIsInstance(fingerprint.invocations[0], RuleInvocation)
        self.assertIsInstance(fingerprint.baseline, AssessmentBaseline)

        with self.assertRaisesRegex(ValueError, "duplicate invocation_id"):
            V06FingerprintInput(
                facts=_v3_facts().to_dict(),
                invocations=(invocation, invocation),
                baseline=_baseline(),
            )

    def test_fingerprint_requires_coherent_facts_invocations_and_baseline(
        self,
    ) -> None:
        facts = _v3_facts().to_dict()
        invocation = _invocation()
        with self.assertRaisesRegex(ValueError, "schema_version"):
            V06FingerprintInput(
                facts=facts,
                invocations=(invocation,),
                baseline=replace(
                    _baseline(),
                    facts_schema_version="2.0.0",
                ),
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            V06FingerprintInput(
                facts=facts,
                invocations=(invocation,),
                baseline=replace(_baseline(), ordered_rules=()),
            )
        with self.assertRaisesRegex(ValueError, "rule_version"):
            V06FingerprintInput(
                facts=facts,
                invocations=(invocation,),
                baseline=_baseline(rule_version="different"),
            )

        upstream = _upstream_invocation()
        baseline = _baseline_with_prerequisites(upstream)
        first = V06FingerprintInput(
            facts=facts,
            invocations=(invocation,),
            baseline=baseline,
        )
        second = V06FingerprintInput(
            facts=deepcopy(facts),
            invocations=(invocation.to_dict(),),
            baseline=baseline.to_dict(),
        )
        self.assertEqual(first.digest(), second.digest())
        self.assertEqual(
            [item.rule_id for item in first.baseline.ordered_rules],
            [invocation.rule_id, upstream.rule_id],
        )

        conflicting = invocation.to_dict()
        conflicting["phase"] = "different-phase"
        with self.assertRaisesRegex(ValueError, "conflicting invocation"):
            V06FingerprintInput(
                facts=_v3_facts().to_dict(),
                invocations=(invocation.to_dict(), conflicting),
                baseline=_baseline(),
            )

        malformed = invocation.to_dict()
        malformed.pop("invocation_id")
        with self.assertRaisesRegex(ValueError, "invocation_id"):
            V06FingerprintInput(
                facts=_v3_facts().to_dict(),
                invocations=(malformed,),
                baseline=_baseline(),
            )

    def test_substantive_and_scope_changes_change_digest(self) -> None:
        original = _v3_facts().to_dict()
        changed = deepcopy(original)
        changed["actors"][0]["display_name"] = "Different employer"
        changed_scope = deepcopy(original)
        changed_scope["processing_operations"][0]["system_ids"] = [
            "system:matcher"
        ]
        digests = {
            V06FingerprintInput(
                facts=item,
                invocations=(_invocation(),),
                baseline=_baseline(),
            ).digest()
            for item in (original, changed, changed_scope)
        }
        self.assertEqual(len(digests), 3)

    def test_constructed_input_is_isolated_from_source_mutation(self) -> None:
        facts = _v3_facts().to_dict()
        statuses = {"invocation:upstream": ("applies",)}
        invocation = replace(
            _invocation(),
            prerequisite_invocation_ids=("invocation:upstream",),
            accepted_upstream_statuses=statuses,
        )
        packs = [
            EvidencePackBaseline(
                instrument_id="EU_AI_ACT", manifest_hash="manifest:one"
            )
        ]
        baseline = AssessmentBaseline(
            ordered_rules=(
                RuleBaselineEntry(
                    invocation.rule_id,
                    invocation.rule_version,
                ),
                RuleBaselineEntry("RULE_A", "1.0.0"),
                RuleBaselineEntry("RULE_B", "1.0.0"),
            ),
            evidence_packs=tuple(packs),
        )
        fingerprint = V06FingerprintInput(
            facts=facts,
            invocations=(invocation,),
            baseline=baseline,
        )
        before = fingerprint.digest()
        facts["actors"][0]["display_name"] = "Mutated"
        statuses["invocation:upstream"] = ("does_not_apply",)
        packs[0] = EvidencePackBaseline(instrument_id="GDPR")
        self.assertEqual(fingerprint.digest(), before)
        with self.assertRaises(TypeError):
            fingerprint.facts["actors"] = ()

    def test_evidence_packs_sort_and_reject_duplicates_but_rules_stay_ordered(self) -> None:
        baseline = AssessmentBaseline(
            ordered_rules=(
                RuleBaselineEntry("RULE_B", "1"),
                RuleBaselineEntry("RULE_A", "1"),
            ),
            evidence_packs=(
                EvidencePackBaseline("GDPR"),
                EvidencePackBaseline("EU_AI_ACT"),
            ),
        )
        self.assertEqual(
            [item.rule_id for item in baseline.ordered_rules],
            ["RULE_B", "RULE_A"],
        )
        self.assertEqual(
            [item.instrument_id for item in baseline.evidence_packs],
            ["EU_AI_ACT", "GDPR"],
        )
        with self.assertRaisesRegex(ValueError, "duplicate instrument"):
            AssessmentBaseline(
                evidence_packs=(
                    EvidencePackBaseline("GDPR"),
                    EvidencePackBaseline("GDPR"),
                )
            )


class FactsSchemaBoundaryTests(unittest.TestCase):
    def test_post_construction_schema_and_v3_field_inconsistency_are_rejected(self) -> None:
        facts = AssessmentFacts()
        facts.actors = [ActorFacts(actor_id="actor:one")]
        with self.assertRaisesRegex(ValueError, "cannot contain v3 fields"):
            facts.to_dict()

        facts = AssessmentFacts()
        facts.schema_version = "3.0.0"
        with self.assertRaisesRegex(ValueError, "cannot be changed in place"):
            facts.to_dict()

    def test_source_mask_cannot_discard_new_data(self) -> None:
        fixture = json.loads(
            (ROOT / "tests/fixtures/recruitment_ai_case.json").read_text()
        )["facts"]
        restored = AssessmentFacts.from_dict(fixture)
        restored.data_protection.personal_data_processed = TriState.YES
        with self.assertRaisesRegex(ValueError, "absent from the source"):
            restored.to_dict()

    def test_explicit_v3_derivation_is_editable_and_source_is_unchanged(self) -> None:
        fixture = json.loads(
            (ROOT / "tests/fixtures/recruitment_ai_case.json").read_text()
        )["facts"]
        source = AssessmentFacts.from_dict(fixture)
        before = source.to_dict()
        draft = AssessmentFactsCompatibilityAdapter.derive_v3(source)
        draft.actors = [ActorFacts(actor_id="actor:new")]
        self.assertEqual(draft.to_dict()["actors"][0]["actor_id"], "actor:new")
        self.assertEqual(source.to_dict(), before)


class StrictIdentityDeserializationTests(unittest.TestCase):
    def test_missing_run_and_evidence_identity_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "id"):
            AssessmentRun.from_dict(
                {"case_id": "case:one", "facts_snapshot": {}}
            )
        evidence = Evidence(
            evidence_id="evidence:one",
            legal_source="EU AI Act",
            citation="Article 6",
            excerpt="Text",
            authority_level=AuthorityLevel.BINDING_LEGISLATION,
        ).to_dict()
        evidence.pop("evidence_id")
        with self.assertRaisesRegex(ValueError, "evidence_id"):
            Evidence.from_dict(evidence)

    def test_missing_nested_finding_identity_in_report_is_rejected(self) -> None:
        payload = _legacy_report().to_dict()
        missing_report_id = deepcopy(payload)
        missing_report_id.pop("report_id")
        with self.assertRaisesRegex(ValueError, "report_id"):
            AssessmentReport.from_dict(missing_report_id)
        payload["findings"] = [
            Finding(
                finding_id="finding:one",
                category=FindingCategory.HIGH_RISK,
                issue_code="TEST",
                status=FindingStatus.UNDETERMINED,
                title="Test",
                summary="Test",
            ).to_dict()
        ]
        payload["findings"][0].pop("finding_id")
        with self.assertRaisesRegex(ValueError, "finding_id"):
            AssessmentReport.from_dict(payload)

    def test_complete_historical_report_round_trip_is_exact(self) -> None:
        payload = _legacy_report().to_dict()
        self.assertEqual(AssessmentReport.from_dict(payload).to_dict(), payload)

    def test_other_persisted_identity_models_require_serialized_ids(self) -> None:
        invocation = _invocation()
        invocation_payload = invocation.to_dict()
        invocation_payload.pop("invocation_id")
        with self.assertRaisesRegex(ValueError, "invocation_id"):
            RuleInvocation.from_dict(invocation_payload)

        execution_payload = RuleExecutionRecord(
            invocation=invocation,
            status=RuleExecutionStatus.NOT_AUTHORIZED,
        ).to_dict()
        execution_payload["invocation"].pop("invocation_id")
        with self.assertRaisesRegex(ValueError, "invocation_id"):
            RuleExecutionRecord.from_dict(execution_payload)

        with self.assertRaisesRegex(ValueError, "finding_id"):
            FindingEvidenceBinding.from_dict({"evidence_refs": []})

        hypothesis = RoleHypothesis(
            hypothesis_id="hypothesis:one",
            framework=RegulatoryFramework.EU_AI_ACT,
            scope=invocation.scope,
            hypothesis_type="deployer_like",
            status=RoleHypothesisStatus.UNRESOLVED,
        ).to_dict()
        hypothesis.pop("hypothesis_id")
        with self.assertRaisesRegex(ValueError, "hypothesis_id"):
            RoleHypothesis.from_dict(hypothesis)

        artefact = ComplianceArtefactMetadata(
            artefact_id="artefact:one",
            artefact_type="notice",
        ).to_dict()
        artefact.pop("artefact_id")
        with self.assertRaisesRegex(ValueError, "artefact_id"):
            ComplianceArtefactMetadata.from_dict(artefact)


class DirectConstructionNormalizationTests(unittest.TestCase):
    def test_execution_record_normalizes_types_and_rejects_invalid_scalars(
        self,
    ) -> None:
        invocation = _invocation()
        record = RuleExecutionRecord(
            invocation=invocation.to_dict(),
            status="not_authorized",
            ruleset_baseline_id="ruleset:one",
            evidence_baseline_id="evidence-baseline:one",
        )
        self.assertIsInstance(record.invocation, RuleInvocation)
        self.assertIs(record.status, RuleExecutionStatus.NOT_AUTHORIZED)

        malformed = invocation.to_dict()
        malformed.pop("invocation_id")
        with self.assertRaisesRegex(ValueError, "invocation_id"):
            RuleExecutionRecord(
                invocation=malformed,
                status="not_authorized",
            )
        with self.assertRaisesRegex(TypeError, "missing_fact_paths"):
            RuleExecutionRecord(
                invocation=invocation,
                status="missing_facts",
                missing_fact_paths="facts.path",
            )
        with self.assertRaisesRegex(ValueError, "non-empty strings"):
            RuleExecutionRecord(
                invocation=invocation,
                status="missing_facts",
                missing_fact_paths=("",),
            )
        with self.assertRaisesRegex(ValueError, "dependency_reason"):
            RuleExecutionRecord(
                invocation=invocation,
                status="blocked_by_dependency",
                dependency_reason="",
            )
        with self.assertRaisesRegex(ValueError, "ruleset_baseline_id"):
            RuleExecutionRecord(
                invocation=invocation,
                status="not_authorized",
                ruleset_baseline_id="bad id",
            )

    def test_execution_record_enforces_status_specific_field_matrix(
        self,
    ) -> None:
        invocation = _invocation()
        valid_records = (
            RuleExecutionRecord(
                invocation=invocation,
                status=RuleExecutionStatus.COMPLETED,
                finding_id="finding:one",
            ),
            RuleExecutionRecord(
                invocation=invocation,
                status=RuleExecutionStatus.NOT_AUTHORIZED,
            ),
            RuleExecutionRecord(
                invocation=invocation,
                status=RuleExecutionStatus.BLOCKED_BY_DEPENDENCY,
                dependency_reason="upstream status not accepted",
            ),
            RuleExecutionRecord(
                invocation=invocation,
                status=RuleExecutionStatus.BLOCKED_BY_EVIDENCE,
                evidence_block_reason="required authority unavailable",
            ),
            RuleExecutionRecord(
                invocation=invocation,
                status=RuleExecutionStatus.MISSING_FACTS,
                missing_fact_paths=("actors[actor:a].acts_in_own_name",),
            ),
            RuleExecutionRecord(
                invocation=invocation,
                status=RuleExecutionStatus.FAILED,
                failure_type="RuntimeError",
            ),
        )
        for record in valid_records:
            with self.subTest(status=record.status):
                self.assertEqual(
                    RuleExecutionRecord.from_dict(record.to_dict()).to_dict(),
                    record.to_dict(),
                )

        invalid_records = (
            (
                "completed.*missing_fact_paths",
                {
                    "status": RuleExecutionStatus.COMPLETED,
                    "missing_fact_paths": ("facts.path",),
                },
            ),
            (
                "completed.*dependency_reason",
                {
                    "status": RuleExecutionStatus.COMPLETED,
                    "dependency_reason": "not allowed",
                },
            ),
            (
                "blocked_by_dependency requires dependency_reason",
                {"status": RuleExecutionStatus.BLOCKED_BY_DEPENDENCY},
            ),
            (
                "blocked_by_evidence requires evidence_block_reason",
                {"status": RuleExecutionStatus.BLOCKED_BY_EVIDENCE},
            ),
            (
                "failed requires failure_type or failure_message",
                {"status": RuleExecutionStatus.FAILED},
            ),
            (
                "not_authorized.*dependency_reason",
                {
                    "status": RuleExecutionStatus.NOT_AUTHORIZED,
                    "dependency_reason": "not allowed",
                },
            ),
            (
                "missing_facts.*failure_type",
                {
                    "status": RuleExecutionStatus.MISSING_FACTS,
                    "missing_fact_paths": ("facts.path",),
                    "failure_type": "not allowed",
                },
            ),
        )
        for pattern, fields in invalid_records:
            with self.subTest(fields=fields), self.assertRaisesRegex(
                ValueError,
                pattern,
            ):
                RuleExecutionRecord(invocation=invocation, **fields)

    def test_identifier_collections_reject_scalar_strings_and_mapping_values(
        self,
    ) -> None:
        with self.assertRaisesRegex(TypeError, "list or tuple"):
            ActorFacts(
                actor_id="actor:a",
                operates_system_ids="abc",
            )
        with self.assertRaisesRegex(TypeError, "list or tuple"):
            ActorFacts(
                actor_id="actor:a",
                uses_system_ids_on_behalf_of_actor_ids={
                    "system:one": "actor:a"
                },
            )

    def test_new_recruitment_models_normalize_declared_enums(self) -> None:
        actor = ActorFacts(
            actor_id="actor:a",
            actor_kind="employer",
            acts_in_own_name="yes",
        )
        self.assertIs(actor.actor_kind, ActorKind.EMPLOYER)
        self.assertIs(actor.acts_in_own_name, TriState.YES)

        operation = ProcessingOperationFacts(
            processing_operation_id="operation:one",
            within_documented_instructions="yes",
        )
        self.assertIs(
            operation.within_documented_instructions,
            TriState.YES,
        )
        criterion = ScreeningCriterionFacts(
            criterion_id="criterion:one",
            used_for_ranking="yes",
            classification_review_state="reviewed",
        )
        self.assertIs(criterion.used_for_ranking, TriState.YES)
        self.assertIs(
            criterion.classification_review_state,
            ClassificationReviewState.REVIEWED,
        )
        process = RecruitmentDecisionProcessFacts(
            process_id="process:one",
            ranking="yes",
        )
        self.assertIs(process.ranking, TriState.YES)

        with self.assertRaisesRegex(ValueError, "ActorKind"):
            ActorFacts(actor_id="actor:a", actor_kind="invalid")
        with self.assertRaisesRegex(ValueError, "TriState"):
            ProcessingOperationFacts(
                processing_operation_id="operation:one",
                independent_reuse_purpose="invalid",
            )

    def test_informational_models_validate_canonical_metadata(self) -> None:
        hypothesis = RoleHypothesis(
            hypothesis_id="hypothesis:one",
            framework="EU_AI_ACT",
            scope=AssessmentScope(actor_id="actor:a"),
            hypothesis_type="deployer_like",
            status="supported",
            supporting_fact_paths=["facts.b", "facts.a"],
            reason_codes=["B", "A"],
        )
        self.assertIs(
            hypothesis.framework, RegulatoryFramework.EU_AI_ACT
        )
        self.assertIs(hypothesis.status, RoleHypothesisStatus.SUPPORTED)
        self.assertEqual(
            hypothesis.supporting_fact_paths, ["facts.a", "facts.b"]
        )
        self.assertEqual(hypothesis.reason_codes, ["A", "B"])

        artefact = ComplianceArtefactMetadata(
            artefact_id="artefact:one",
            artefact_type=" notice ",
            file_reference=" opaque:reference ",
            supply_state="supplied",
            provenance={"source": "user"},
            descriptive_metadata={"title": "Notice"},
        )
        self.assertIs(artefact.supply_state, ArtefactSupplyState.SUPPLIED)
        self.assertEqual(artefact.artefact_type, "notice")
        self.assertEqual(artefact.file_reference, "opaque:reference")

        with self.assertRaisesRegex(TypeError, "list or tuple"):
            RoleHypothesis(
                hypothesis_id="hypothesis:one",
                framework=RegulatoryFramework.EU_AI_ACT,
                scope=AssessmentScope(actor_id="actor:a"),
                hypothesis_type="deployer_like",
                supporting_fact_paths="facts.path",
            )
        with self.assertRaisesRegex(ValueError, "projection_version"):
            replace(hypothesis, projection_version="2.0.0")
        with self.assertRaisesRegex(TypeError, "string-to-string"):
            ComplianceArtefactMetadata(
                artefact_id="artefact:one",
                artefact_type="notice",
                provenance={"source": 1},
            )


class ReferenceValidationTests(unittest.TestCase):
    def test_plain_references_normalize_and_malformed_values_fail(self) -> None:
        actor = ActorFacts(
            actor_id="actor:one",
            operates_system_ids=["system:two", "system:one"],
        )
        self.assertTrue(all(isinstance(item, SystemId) for item in actor.operates_system_ids))
        self.assertEqual(actor.operates_system_ids, ["system:one", "system:two"])
        with self.assertRaises(ValueError):
            ActorFacts(actor_id="actor:one", operates_system_ids=["bad id"])

    def test_all_implemented_dangling_reference_classes_are_rejected(self) -> None:
        cases: list[tuple[str, callable]] = [
            (
                "actor system",
                lambda facts: facts.actors[0].operates_system_ids.append(
                    SystemId("system:missing")
                ),
            ),
            (
                "system actor",
                lambda facts: facts.ai_systems[0].vendor_actor_ids.__class__,
            ),
        ]
        facts = _v3_facts()
        facts.actors[0].operates_system_ids = ["system:missing"]
        with self.assertRaisesRegex(ValueError, "actor system"):
            facts.validate_v3()

        facts = _v3_facts()
        facts.ai_systems[0].vendor_actor_ids = ["actor:missing"]
        with self.assertRaisesRegex(ValueError, "system actor"):
            facts.validate_v3()

        facts = _v3_facts()
        facts.recruitment_processes[0].screening_criteria[
            0
        ].selecting_actor_ids = ["actor:missing"]
        with self.assertRaisesRegex(ValueError, "criterion selecting actor"):
            facts.validate_v3()

        facts = _v3_facts()
        facts.recruitment_processes[0].workflow_id = "workflow:missing"
        with self.assertRaisesRegex(ValueError, "process workflow"):
            facts.validate_v3()

        facts = _v3_facts()
        facts.temporal_context.intended_use_dates = {"workflow:missing": None}
        with self.assertRaisesRegex(ValueError, "temporal workflow"):
            facts.validate_v3()

        facts = _v3_facts()
        facts.territorial_context.actor_establishment_locations = {
            "actor:missing": ["EU"]
        }
        with self.assertRaisesRegex(ValueError, "territorial actor"):
            facts.validate_v3()

        facts = _v3_facts()
        facts.compliance_artefacts = [
            ComplianceArtefactMetadata(
                artefact_id="artefact:one",
                artefact_type="notice",
                scope=AssessmentScope(actor_id="actor:missing"),
            )
        ]
        with self.assertRaisesRegex(ValueError, "compliance artefact actor"):
            facts.validate_v3()


class DependencyContractTests(unittest.TestCase):
    def test_rule_invocation_dependency_metadata_must_be_coherent(self) -> None:
        with self.assertRaisesRegex(ValueError, "undeclared prerequisite"):
            replace(
                _invocation(),
                accepted_upstream_statuses={
                    "invocation:undeclared": ("applies",)
                },
            )
        with self.assertRaisesRegex(ValueError, "requires accepted statuses"):
            replace(
                _invocation(),
                prerequisite_invocation_ids=("invocation:required",),
            )
        with self.assertRaisesRegex(TypeError, "status collections"):
            replace(
                _invocation(),
                prerequisite_invocation_ids=("invocation:required",),
                accepted_upstream_statuses={
                    "invocation:required": "applies"
                },
            )
        for statuses in ((), ("",), ("applies", "applies")):
            with self.subTest(statuses=statuses), self.assertRaises(
                (ValueError, TypeError)
            ):
                replace(
                    _invocation(),
                    prerequisite_invocation_ids=("invocation:required",),
                    accepted_upstream_statuses={
                        "invocation:required": statuses
                    },
                )

    def test_authorization_subject_selector_is_validated_and_normalized(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "subject_selector"):
            AuthorizedRuleInvocation(
                authorization_id="authorization:selector",
                rule_id="RULE",
                rule_version="1",
                subject_selector="   ",
            )
        with self.assertRaisesRegex(TypeError, "subject_selector"):
            AuthorizedRuleInvocation(
                authorization_id="authorization:selector",
                rule_id="RULE",
                rule_version="1",
                subject_selector=123,
            )
        selector = AuthorizedRuleInvocation(
            authorization_id="authorization:selector",
            rule_id="RULE",
            rule_version="1",
            subject_selector="  actor:all  ",
        )
        self.assertEqual(selector.subject_selector, "actor:all")

        explicit = _authorization(_invocation())
        self.assertEqual(explicit.scopes, (_invocation().scope,))
        self.assertIsNone(explicit.subject_selector)

    def test_context_binds_only_declared_accepted_prerequisites(self) -> None:
        invocation, upstream = _dependent_invocation()
        authorization = _authorization(invocation)
        baseline = _baseline_with_prerequisites(upstream)

        def summary(**changes):
            values = {
                "finding_id": "finding:upstream",
                "prerequisite_invocation_id": upstream.invocation_id,
                "scope": upstream.scope,
                "framework": RegulatoryFramework.EU_AI_ACT,
                "rule_id": upstream.rule_id,
                "rule_version": upstream.rule_version,
                "status": FindingStatus.POTENTIALLY_APPLIES.value,
            }
            values.update(changes)
            return PrerequisiteFindingSummary(**values)

        valid = summary()
        context = AssessmentContext(
            facts_snapshot=_v3_facts(),
            invocation=invocation,
            prerequisite_findings=(valid,),
            authorization=authorization,
            baseline=baseline,
        )
        self.assertEqual(
            context.prerequisite_findings[0].prerequisite_invocation_id,
            upstream.invocation_id,
        )

        undeclared = replace(
            valid,
            prerequisite_invocation_id="invocation:undeclared",
        )
        with self.assertRaisesRegex(ValueError, "undeclared"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                prerequisite_findings=(undeclared,),
                authorization=authorization,
                baseline=baseline,
            )
        with self.assertRaisesRegex(ValueError, "duplicate prerequisite"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                prerequisite_findings=(valid, valid),
                authorization=authorization,
                baseline=baseline,
            )
        with self.assertRaisesRegex(ValueError, "rule metadata"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                prerequisite_findings=(summary(rule_version="wrong"),),
                authorization=authorization,
                baseline=baseline,
            )
        with self.assertRaisesRegex(ValueError, "not accepted"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                prerequisite_findings=(summary(status="does_not_apply"),),
                authorization=authorization,
                baseline=baseline,
            )

    def test_context_requires_coherent_facts_and_baseline_versions(self) -> None:
        invocation = _invocation()
        authorization = _authorization(invocation)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            AssessmentContext(
                facts_snapshot=AssessmentFacts(),
                invocation=invocation,
                authorization=authorization,
                baseline=_baseline(),
            )
        with self.assertRaisesRegex(
            ValueError, "assessment_context_version"
        ):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                authorization=authorization,
                baseline=replace(
                    _baseline(),
                    assessment_context_version="2.0.0",
                ),
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                authorization=authorization,
                baseline=replace(_baseline(), ordered_rules=()),
            )
        with self.assertRaisesRegex(ValueError, "rule_version"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                authorization=authorization,
                baseline=replace(
                    _baseline(),
                    ordered_rules=(
                        RuleBaselineEntry(invocation.rule_id, "different"),
                    ),
                ),
            )

    def test_context_requires_complete_baselined_prerequisite_set(self) -> None:
        invocation, upstream = _dependent_invocation()
        authorization = _authorization(invocation)

        def summary(item: RuleInvocation) -> PrerequisiteFindingSummary:
            return PrerequisiteFindingSummary(
                finding_id=f"finding:{item.rule_id.lower()}",
                prerequisite_invocation_id=item.invocation_id,
                scope=item.scope,
                framework=RegulatoryFramework.EU_AI_ACT,
                rule_id=item.rule_id,
                rule_version=item.rule_version,
                status=FindingStatus.POTENTIALLY_APPLIES.value,
            )

        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                prerequisite_findings=(),
                authorization=authorization,
                baseline=_baseline_with_prerequisites(upstream),
            )

        second_upstream = RuleInvocation.create(
            rule_id="AI_ACT_SECOND_PREREQUISITE",
            rule_version="2026.1",
            scope=invocation.scope,
        )
        two_dependency_invocation = replace(
            invocation,
            prerequisite_invocation_ids=(
                upstream.invocation_id,
                second_upstream.invocation_id,
            ),
            accepted_upstream_statuses={
                str(upstream.invocation_id): (
                    FindingStatus.POTENTIALLY_APPLIES.value,
                ),
                str(second_upstream.invocation_id): (
                    FindingStatus.POTENTIALLY_APPLIES.value,
                ),
            },
        )
        two_dependency_authorization = _authorization(
            two_dependency_invocation
        )
        complete_baseline = _baseline_with_prerequisites(
            upstream,
            second_upstream,
        )
        with self.assertRaisesRegex(ValueError, "requires exactly one"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=two_dependency_invocation,
                prerequisite_findings=(summary(upstream),),
                authorization=two_dependency_authorization,
                baseline=complete_baseline,
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                prerequisite_findings=(summary(upstream),),
                authorization=authorization,
                baseline=_baseline(),
            )
        with self.assertRaisesRegex(ValueError, "rule_version"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                prerequisite_findings=(summary(upstream),),
                authorization=authorization,
                baseline=replace(
                    _baseline(),
                    ordered_rules=_baseline().ordered_rules
                    + (
                        RuleBaselineEntry(
                            upstream.rule_id,
                            "different",
                        ),
                    ),
                ),
            )

        coherent = AssessmentContext(
            facts_snapshot=_v3_facts(),
            invocation=two_dependency_invocation,
            prerequisite_findings=(
                summary(upstream),
                summary(second_upstream),
            ),
            authorization=two_dependency_authorization,
            baseline=complete_baseline,
        )
        self.assertEqual(
            AssessmentContext.from_dict(coherent.to_dict()).to_dict(),
            coherent.to_dict(),
        )


class ReportAndContextBoundaryTests(unittest.TestCase):
    def test_report_version_boundaries_and_v2_reference_round_trip(self) -> None:
        legacy = _legacy_report()
        with self.assertRaisesRegex(ValueError, "Report 2.0.0 fields"):
            replace(
                legacy,
                rule_execution_records=[
                    RuleExecutionRecord(
                        invocation=_invocation(),
                        status=RuleExecutionStatus.NOT_AUTHORIZED,
                    )
                ],
            )
        legacy.report_version = "2.0.0"
        with self.assertRaisesRegex(ValueError, "cannot be changed in place"):
            legacy.to_dict()

        report = replace(
            _legacy_report(),
            report_version="2.0.0",
            actor_references=["actor:two", "actor:one"],
            system_references=["system:one"],
            workflow_references=["workflow:one"],
            processing_operation_references=["operation:one"],
        )
        payload = report.to_dict()
        self.assertEqual(payload["actor_references"], ["actor:one", "actor:two"])
        self.assertEqual(AssessmentReport.from_dict(payload).to_dict(), payload)

    def test_context_round_trip_authorization_and_defensive_properties(self) -> None:
        dependent, upstream = _dependent_invocation()
        invocation = replace(
            dependent,
            authorization_reference="authorization:one",
        )
        authorization = _authorization(invocation)
        source = _v3_facts()
        context = AssessmentContext(
            facts_snapshot=source,
            invocation=invocation,
            prerequisite_findings=(
                PrerequisiteFindingSummary(
                    finding_id="finding:upstream",
                    prerequisite_invocation_id=upstream.invocation_id,
                    scope=upstream.scope,
                    framework=RegulatoryFramework.EU_AI_ACT,
                    rule_id=upstream.rule_id,
                    rule_version=upstream.rule_version,
                    status="potentially_applies",
                ),
            ),
            authorization=authorization,
            baseline=_baseline_with_prerequisites(upstream),
        )
        payload = context.to_dict()
        self.assertEqual(AssessmentContext.from_dict(payload).to_dict(), payload)
        source.actors[0].display_name = "mutated"
        returned = context.facts_snapshot
        returned.actors[0].display_name = "returned mutation"
        self.assertNotEqual(
            context.facts_snapshot.actors[0].display_name,
            "returned mutation",
        )
        with self.assertRaises(AttributeError):
            context._facts_snapshot = {}

    def test_context_rejects_mismatches_selector_only_and_versions(self) -> None:
        invocation = _invocation()
        authorization = _authorization(invocation)
        with self.assertRaisesRegex(ValueError, "rule_id"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                authorization=replace(authorization, rule_id="OTHER"),
                baseline=_baseline(),
            )
        with self.assertRaisesRegex(ValueError, "rule_version"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                authorization=replace(authorization, rule_version="OTHER"),
                baseline=_baseline(),
            )
        with self.assertRaisesRegex(ValueError, "explicitly expanded"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=invocation,
                authorization=AuthorizedRuleInvocation(
                    authorization_id="authorization:selector",
                    rule_id=invocation.rule_id,
                    rule_version=invocation.rule_version,
                    subject_selector="all",
                ),
                baseline=_baseline(),
            )
        with self.assertRaisesRegex(ValueError, "authorization_reference"):
            AssessmentContext(
                facts_snapshot=_v3_facts(),
                invocation=replace(
                    invocation,
                    authorization_reference="authorization:other",
                ),
                authorization=authorization,
                baseline=_baseline(),
            )
        payload = AssessmentContext(
            facts_snapshot=_v3_facts(),
            invocation=invocation,
            authorization=authorization,
            baseline=_baseline(),
        ).to_dict()
        payload["contract_version"] = "9.0.0"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            AssessmentContext.from_dict(payload)
        payload = AssessmentContext(
            facts_snapshot=_v3_facts(),
            invocation=invocation,
            authorization=authorization,
            baseline=_baseline(),
        ).to_dict()
        payload["unknown"] = True
        with self.assertRaisesRegex(ValueError, "unknown AssessmentContext"):
            AssessmentContext.from_dict(payload)

    def test_nested_canonical_contracts_resist_mutation_and_versions_are_exact(self) -> None:
        statuses = {"invocation:upstream": ["applies"]}
        invocation = replace(
            _invocation(),
            prerequisite_invocation_ids=("invocation:upstream",),
            accepted_upstream_statuses=statuses,
        )
        statuses["invocation:upstream"].append("undetermined")
        self.assertEqual(
            invocation.accepted_upstream_statuses["invocation:upstream"],
            ("applies",),
        )
        with self.assertRaises(TypeError):
            invocation.accepted_upstream_statuses["new"] = ()
        expansion = {"actors": ["actor:one"]}
        authorization = replace(
            _authorization(invocation),
            expansion_inputs=expansion,
        )
        expansion["actors"].append("actor:two")
        self.assertEqual(
            authorization.to_dict()["expansion_inputs"]["actors"],
            ["actor:one"],
        )
        for constructor in (
            lambda: replace(invocation, contract_version="2.0.0"),
            lambda: replace(authorization, contract_version="2.0.0"),
            lambda: replace(_baseline(), baseline_version="2.0.0"),
            lambda: RuleExecutionRecord(
                invocation=_invocation(),
                status=RuleExecutionStatus.NOT_AUTHORIZED,
                schema_version="2.0.0",
            ),
            lambda: V06FingerprintInput(
                facts=_v3_facts().to_dict(),
                invocations=(_invocation(),),
                baseline=_baseline(),
                contract_version="2.0.0",
            ),
        ):
            with self.assertRaises(ValueError):
                constructor()


if __name__ == "__main__":
    unittest.main()
