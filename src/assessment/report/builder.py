"""Deterministic construction of structured assessment reports."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json

from src.assessment.evidence.service import EvidenceServiceResult
from src.assessment.findings import Finding, FindingStatus
from src.assessment.frameworks import RegulatoryFramework
from src.assessment.report.models import (
    AssessmentReport,
    FrameworkFindings,
    MissingInformation,
    RuleVersionMetadata,
)
from src.assessment.results import AssessmentResult


class ReportBuildError(ValueError):
    """Raised when report inputs cannot form a traceable report."""


class ReportBuilder:
    """Project assessment and evidence results into one immutable snapshot."""

    VERSION = "1.0.0"

    _STATUS_ORDER = (
        FindingStatus.APPLIES,
        FindingStatus.POTENTIALLY_APPLIES,
        FindingStatus.DOES_NOT_APPLY,
        FindingStatus.UNDETERMINED,
        FindingStatus.NOT_ASSESSED,
    )
    _FRAMEWORK_ORDER = (
        RegulatoryFramework.EU_AI_ACT,
        RegulatoryFramework.GDPR,
        RegulatoryFramework.EU_DATA_ACT,
        RegulatoryFramework.UNKNOWN,
    )

    def __init__(self, *, report_version: str = VERSION) -> None:
        if not isinstance(report_version, str) or not report_version.strip():
            raise ValueError("report_version must be a non-empty string")
        self._report_version = report_version

    def build(
        self,
        assessment_result: AssessmentResult,
        evidence_result: EvidenceServiceResult,
    ) -> AssessmentReport:
        """Build a report without adding or changing legal conclusions."""

        if not isinstance(assessment_result, AssessmentResult):
            raise TypeError("assessment_result must be an AssessmentResult")
        if not isinstance(evidence_result, EvidenceServiceResult):
            raise TypeError("evidence_result must be an EvidenceServiceResult")

        self._validate_traceability(assessment_result, evidence_result)
        missing_information = self._build_missing_information(assessment_result)
        rule_versions = self._build_rule_versions(assessment_result)
        summary = self._build_summary(
            assessment_result=assessment_result,
            evidence_result=evidence_result,
            missing_information=missing_information,
        )
        recommendations = self._build_recommendations(
            assessment_result=assessment_result,
            evidence_result=evidence_result,
            missing_information=missing_information,
        )
        assessment_reference = self._assessment_reference(assessment_result)
        report_id = self._report_id(assessment_result, evidence_result)
        findings_by_framework = self._build_framework_findings(
            assessment_result.findings
        )

        return AssessmentReport(
            report_id=report_id,
            assessment_run_reference=assessment_reference,
            generated_at=deepcopy(assessment_result.timestamp),
            summary=summary,
            findings=deepcopy(assessment_result.findings),
            evidence=deepcopy(evidence_result.evidence),
            evidence_bindings=deepcopy(evidence_result.bindings),
            missing_information=missing_information,
            recommendations=recommendations,
            engine_version=assessment_result.engine_version,
            rule_versions=rule_versions,
            execution_failures=deepcopy(assessment_result.failures),
            report_version=self._report_version,
            findings_by_framework=findings_by_framework,
            assessed_frameworks=deepcopy(
                assessment_result.assessed_frameworks
            ),
        )

    def _build_framework_findings(
        self,
        findings: list[Finding],
    ) -> list[FrameworkFindings]:
        """Group copied findings in stable framework and execution order."""

        grouped = {framework: [] for framework in self._FRAMEWORK_ORDER}
        for finding in findings:
            grouped[finding.framework].append(deepcopy(finding))
        return [
            FrameworkFindings(
                framework=framework,
                findings=grouped[framework],
            )
            for framework in self._FRAMEWORK_ORDER
            if grouped[framework]
        ]

    def _validate_traceability(
        self,
        assessment_result: AssessmentResult,
        evidence_result: EvidenceServiceResult,
    ) -> None:
        finding_ids = [finding.finding_id for finding in assessment_result.findings]
        evidence_ids = [evidence.evidence_id for evidence in evidence_result.evidence]
        binding_finding_ids = [
            binding.finding_id for binding in evidence_result.bindings
        ]

        if len(set(finding_ids)) != len(finding_ids):
            raise ReportBuildError("assessment findings contain duplicate IDs")
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ReportBuildError("evidence result contains duplicate IDs")
        if len(set(binding_finding_ids)) != len(binding_finding_ids):
            raise ReportBuildError("a finding has more than one evidence binding")

        known_findings = set(finding_ids)
        known_evidence = set(evidence_ids)
        for binding in evidence_result.bindings:
            if binding.finding_id not in known_findings:
                raise ReportBuildError(
                    f"evidence binding references unknown finding {binding.finding_id!r}"
                )
            unknown_evidence = [
                evidence_id
                for evidence_id in binding.evidence_refs
                if evidence_id not in known_evidence
            ]
            if unknown_evidence:
                raise ReportBuildError(
                    f"evidence binding references unknown evidence {unknown_evidence!r}"
                )

    @staticmethod
    def _build_missing_information(
        assessment_result: AssessmentResult,
    ) -> list[MissingInformation]:
        missing_information: list[MissingInformation] = []
        for requirement in assessment_result.missing_fact_requirements:
            for missing_fact in requirement.missing_facts:
                missing_information.append(
                    MissingInformation(
                        rule_id=requirement.rule_id,
                        rule_version=requirement.rule_version,
                        fact_path=missing_fact.fact_path,
                        reason=missing_fact.reason,
                        framework=requirement.framework,
                    )
                )
        return missing_information

    def _build_summary(
        self,
        *,
        assessment_result: AssessmentResult,
        evidence_result: EvidenceServiceResult,
        missing_information: list[MissingInformation],
    ) -> str:
        status_counts = Counter(
            finding.status for finding in assessment_result.findings
        )
        status_parts = [
            f"{status_counts[status]} {status.value}"
            for status in self._STATUS_ORDER
            if status_counts[status]
        ]
        status_detail = ", ".join(status_parts) if status_parts else "none"
        return (
            f"Preliminary assessment produced {len(assessment_result.findings)} "
            f"finding(s) ({status_detail}). Missing information: "
            f"{len(missing_information)} item(s). Evidence bindings: "
            f"{len(evidence_result.bindings)}. Rule execution failures: "
            f"{len(assessment_result.failures)}."
        )

    @staticmethod
    def _build_recommendations(
        *,
        assessment_result: AssessmentResult,
        evidence_result: EvidenceServiceResult,
        missing_information: list[MissingInformation],
    ) -> list[str]:
        recommendations: list[str] = []

        seen_fact_paths: set[str] = set()
        for item in missing_information:
            if item.fact_path in seen_fact_paths:
                continue
            seen_fact_paths.add(item.fact_path)
            recommendations.append(
                f"Provide or confirm the missing fact '{item.fact_path}' before "
                "rerunning the affected assessment rules."
            )

        for finding in assessment_result.findings:
            if finding.requires_legal_review:
                recommendations.append(
                    f"Obtain legal review for finding '{finding.issue_code}' before "
                    "relying on the preliminary result."
                )

        bound_finding_ids = {
            binding.finding_id for binding in evidence_result.bindings
        }
        for finding in assessment_result.findings:
            if finding.legal_basis and finding.finding_id not in bound_finding_ids:
                recommendations.append(
                    f"Obtain supporting authority for finding '{finding.issue_code}'."
                )

        for failure in assessment_result.failures:
            recommendations.append(
                f"Review the execution failure for rule '{failure.rule_id}' before "
                "finalizing the assessment."
            )
        return recommendations

    @staticmethod
    def _build_rule_versions(
        assessment_result: AssessmentResult,
    ) -> list[RuleVersionMetadata]:
        versions_by_rule: dict[
            str,
            tuple[str, RegulatoryFramework],
        ] = {}
        for finding in assessment_result.findings:
            if finding.rule_id and finding.rule_version:
                versions_by_rule[finding.rule_id] = (
                    finding.rule_version,
                    finding.framework,
                )
        for failure in assessment_result.failures:
            versions_by_rule[failure.rule_id] = (
                failure.rule_version,
                failure.framework,
            )
        for requirement in assessment_result.missing_fact_requirements:
            versions_by_rule[requirement.rule_id] = (
                requirement.rule_version,
                requirement.framework,
            )

        metadata: list[RuleVersionMetadata] = []
        seen: set[str] = set()
        for rule_id in assessment_result.executed_rule_ids:
            version_metadata = versions_by_rule.get(rule_id)
            if version_metadata is None:
                raise ReportBuildError(
                    f"executed rule {rule_id!r} has no version metadata"
                )
            version, framework = version_metadata
            metadata.append(
                RuleVersionMetadata(
                    rule_id=rule_id,
                    version=version,
                    framework=framework,
                )
            )
            seen.add(rule_id)
        for requirement in assessment_result.missing_fact_requirements:
            if requirement.rule_id not in seen:
                metadata.append(
                    RuleVersionMetadata(
                        rule_id=requirement.rule_id,
                        version=requirement.rule_version,
                        framework=requirement.framework,
                    )
                )
                seen.add(requirement.rule_id)
        return metadata

    def _assessment_reference(self, assessment_result: AssessmentResult) -> str:
        if assessment_result.assessment_run_id is not None:
            if not assessment_result.assessment_run_id.strip():
                raise ReportBuildError(
                    "assessment_result contains an empty assessment_run_id"
                )
            mismatched_findings = [
                finding.finding_id
                for finding in assessment_result.findings
                if finding.assessment_run_id is not None
                and finding.assessment_run_id != assessment_result.assessment_run_id
            ]
            if mismatched_findings:
                raise ReportBuildError(
                    "findings reference a different assessment run: "
                    f"{mismatched_findings!r}"
                )
            return assessment_result.assessment_run_id

        explicit_run_ids = {
            finding.assessment_run_id
            for finding in assessment_result.findings
            if finding.assessment_run_id is not None
        }
        if len(explicit_run_ids) > 1:
            raise ReportBuildError("findings reference multiple assessment runs")
        if explicit_run_ids:
            return explicit_run_ids.pop()
        digest = self._digest(assessment_result.to_dict())
        return f"assessment:{digest[:24]}"

    def _report_id(
        self,
        assessment_result: AssessmentResult,
        evidence_result: EvidenceServiceResult,
    ) -> str:
        digest = self._digest(
            {
                "assessment_result": assessment_result.to_dict(),
                "evidence_result": evidence_result.to_dict(),
                "report_version": self._report_version,
            }
        )
        return f"report:{digest[:24]}"

    @staticmethod
    def _digest(value: object) -> str:
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()
