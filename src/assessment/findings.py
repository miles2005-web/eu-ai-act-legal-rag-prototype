"""Typed legal findings produced by future deterministic assessment rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.assessment.frameworks import RegulatoryFramework
from src.assessment.models import SerializableModel, new_identifier


class FindingStatus(str, Enum):
    APPLIES = "applies"
    DOES_NOT_APPLY = "does_not_apply"
    POTENTIALLY_APPLIES = "potentially_applies"
    UNDETERMINED = "undetermined"
    NOT_ASSESSED = "not_assessed"


class FindingCategory(str, Enum):
    AI_SYSTEM_STATUS = "ai_system_status"
    SCOPE = "scope"
    HIGH_RISK = "high_risk"
    ROLE_PROVIDER = "role_provider"
    ROLE_DEPLOYER = "role_deployer"
    ROLE_IMPORTER = "role_importer"
    ROLE_DISTRIBUTOR = "role_distributor"
    PROHIBITED_PRACTICE = "prohibited_practice"
    HIGH_RISK_ARTICLE_6_1 = "high_risk_article_6_1"
    HIGH_RISK_ARTICLE_6_2 = "high_risk_article_6_2"
    HIGH_RISK_ARTICLE_6_3_EXCEPTION = "high_risk_article_6_3_exception"
    OBLIGATION_GROUP = "obligation_group"
    INFORMATION_GAP = "information_gap"
    DATA_PROTECTION = "data_protection"
    DATA_GOVERNANCE = "data_governance"


@dataclass(slots=True)
class LegalBasis(SerializableModel):
    instrument: str
    citation: str
    anchor: str


@dataclass(slots=True)
class FindingTraceEntry(SerializableModel):
    description: str
    fact_refs: list[str] = field(default_factory=list)
    result: str | None = None


@dataclass(slots=True)
class Finding(SerializableModel):
    """A serializable result from a future rule or coordinated rule group."""

    category: FindingCategory
    issue_code: str
    status: FindingStatus
    title: str
    summary: str
    framework: RegulatoryFramework = RegulatoryFramework.UNKNOWN
    finding_id: str = field(default_factory=new_identifier)
    assessment_run_id: str | None = None
    rule_id: str | None = None
    rule_version: str | None = None
    fact_refs: list[str] = field(default_factory=list)
    missing_fact_refs: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    legal_basis: list[LegalBasis] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    requires_legal_review: bool = False
    trace: list[FindingTraceEntry] = field(default_factory=list)
